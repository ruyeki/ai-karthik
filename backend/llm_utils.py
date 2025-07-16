from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from helper_functions import parse_response, connect_db
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import pickle
from helper_functions import build_prompt, parse_response
import pickle
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import base64
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from base64 import b64decode
import tiktoken
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from fpdf import FPDF
import asyncio

# Load API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
openai_api_key = os.getenv("OPENROUTER_API_KEY")

genai.configure(api_key=api_key)


#This is not as good as 2.5 pro, but use this in case the gemini tokens that karthik gave run out
'''
model = ChatOpenAI(
    temperature=0.5,
    model_name="google/gemini-2.5-flash-preview-05-20",  
    base_url="https://openrouter.ai/api/v1",
    api_key=openai_api_key
)
'''

#better chat model for report generation
model = ChatGoogleGenerativeAI(
    model="google/gemini-2.5-pro-exp-03-25", 
    google_api_key=openai_api_key,
    temperature=0.3
)


def get_all_summaries(retriever):

    # Directly fetch all vectors from the Chroma store
    return [doc for doc in retriever.vectorstore.get()['documents']]

async def generate_intro_and_summary_and_outline(retriever):
    # Get all text summaries from vectorstore
    summaries = get_all_summaries(retriever)
    context = "\n".join(summaries)

    
    prompt_template = """
        You are an expert scientific writer.

        Your task is to write **three sections** of a technical report based on the provided context:

        1. **Introduction** – Describe the motivation for the project, the problem context, and what was being attempted.
        2. **Summary** – Provide a concise, high-level overview of the key findings, results, and outcomes.
        3. **Objectives** - List the specific goals of the project in a numbered format, using clear and concise language.

        Only use the information in the provided context. Do not fabricate, infer, or include external knowledge.

        Write in a **clear, professional, and concise** tone suitable for a technical audience.

        ---

        **Context:**
        {context}

        ---

        **Output Format Example:**

        Summary:  
        The goal of this project was to develop a formulation of allopregnanolone (Allo) with a target solubility of 6 mg/mL in saline, suitable for intravenous (IV) or transdermal administration. Given the hydrophobic nature of Allo and limited water solubility, solvent screening, excipient selection, and emulsion development studies were conducted. Initial efforts focused on evaluating various solvents and surfactants. The excipient testing was guided by both empirical screening and further computational predictions. Emulsions using Span 20 and Tween 20 showed potential for use, albeit with limited formulation stability.

        Introduction:  
        Allopregnanolone is a neuroactive steroid under investigation for its potential in treating neurological and neurodegenerative conditions. Its low aqueous solubility presents a major challenge in formulating it for parenteral or transdermal delivery. Several patents currently exist covering SEDDS and micellar formulations, necessitating novel approaches that avoid infringement. This study was initiated to systematically evaluate a broad range of excipients and solvent systems, using both empirical and computational tools to guide formulation development.
       
        Objectives: 
        The project had the following objectives:
            1. To develop a 6 mg/mL formulation of Allo in saline.
            2. To evaluate physical and chemical stability over 1–3 months.
            3. To ensure the formulation avoids infringement of existing patents.
         """



    prompt = ChatPromptTemplate.from_template(prompt_template)

    chain = prompt | model | StrOutputParser()

    return await chain.ainvoke({"context": context})

async def generate_methodology(retriever, project_name): 

    #want to retrieve the raw documents instead of summaries and feed that as context to llm

    #retrieve the unstructured data from pickle, it is in caches/all_content[section_name]
    if os.path.exists("./caches/all_content_cache.pkl"): 
        with open("./caches/all_content_cache.pkl", "rb") as f: 
            all_content = pickle.load(f)

    project_raw_documents = all_content[project_name]

    def is_methodology_related(element): 
        text = str(element).lower()

        if any(h in text for h in ["# method", "## method", "# methodology", "## methodology"]):
            return True

        # Fallback: keyword-based match (if section headers aren't included)
        keywords = [
            "protocol", "procedure", "method", "solubility", "formulation", "excipient",
            "hplc", "mixing", "optimization", "characterization", "spectrophotometry",
            "turbidity", "microscopy", "emulsion", "computational prediction"
        ]

        return any(k in text for k in keywords)
    
    filtered_elements = [str(e) for e in project_raw_documents if is_methodology_related(e)]


    prompt_template = """
    You are an expert scientific writer.

    Your task is to write the **Methodology** section of a technical report using only the provided context.

    Organize the section into relevant **subsections** with appropriate titles. Write in clear, professional **paragraph form** — avoid bullet points or numbered lists unless describing step-by-step procedures.

    Each subsection should include:
    - Materials or compounds used (e.g., solvents, excipients)
    - Procedures performed
    - Equipment, techniques, and any measured parameters
    - Optional: brief rationale for the approach

    Use a formal and concise scientific tone. Do not speculate or fabricate.

If the context includes tables or figures, mention them naturally within the paragraph text by inserting inline placeholders like `[TABLE_1]` or `[FIGURE_1]` at the exact point of reference.

Immediately after the paragraph that references a table or figure, include its caption on the next line in this exact format:

**Table 1: Description of the data in the table**  
**Figure 1: Description of the figure content**

Number tables and figures sequentially throughout the entire Methodology section (i.e., TABLE_1, TABLE_2, FIGURE_1, FIGURE_2, etc.).

Do not separate placeholders and captions with extra blank lines; keep them directly below the paragraph referencing them, so they appear interwoven in the text.

Ensure each placeholder and caption pair corresponds clearly to unique tables or figures in the context.

    ---

    Here is an example of a high-quality Methodology section written in paragraph form:

    ### Solvent Solubility Screening
    Allopregnanolone was evaluated for solubility in a range of pharmaceutically acceptable solvents. Drug stocks were prepared at increasing concentrations ranging from 1 to 200 mg/mL in each solvent. These mixtures were vortexed for 1–2 minutes and allowed to equilibrate for 24 hours at room temperature. Preliminary solubility was assessed visually, followed by filtration through 0.22 μm PTFE filters and analysis using UV-vis spectrophotometry at 286 nm. Standard curves were constructed to determine saturation concentrations.

    ---

    Now write a new Methodology section using the context below:

    {context}

    ---
    Respond in this format:

    Methodology:

    ### <Subsection Title 1>
    <Your paragraph-form explanation>

    ### <Subsection Title 2>
    <Another paragraph-form explanation>
    """



    prompt = ChatPromptTemplate.from_template(prompt_template)

    chain = prompt | model | StrOutputParser()

    return await chain.ainvoke({"context": filtered_elements})

async def generate_results(retriever, project_name): 

    #want to retrieve the raw documents instead of summaries and feed that as context to llm

    #retrieve the unstructured data from pickle, it is in caches/all_content[section_name]
    if os.path.exists("./caches/all_content_cache.pkl"): 
        with open("./caches/all_content_cache.pkl", "rb") as f: 
            all_content = pickle.load(f)

    project_raw_documents = all_content[project_name]

    def is_results_related(element): 
        text = str(element).lower()

        if any(h in text for h in ["# result", "## result", "# results", "## results", "# data", "## data"]):
            return True

        # Fallback: keyword-based match (if section headers aren't included)
        keywords = [
            "result", "results", "finding", "findings", "observation", "observations",
            "data", "measurement", "measurements", "analysis", "analyses",
            "effect", "effects", "comparison", "yield", "concentration",
            "precipitate", "solubility", "turbidity", "microscopy", "hplc",
            "image", "imaging", "spectrophotometry", "table", "figure"
        ]

        return any(k in text for k in keywords)
    
    filtered_elements = [str(e) for e in project_raw_documents if is_results_related(e)]


    prompt_template = """
You are an expert scientific writer.

Your task is to write the **Results** section of a technical report using only the provided context.

Organize the section into relevant **subsections** that reflect different experiments or cohorts. Write in clear, professional **paragraph form**. Avoid interpretation or speculation — stick strictly to the observed data.

Each subsection should include:
- Description of what was tested or measured
- Quantitative results (e.g., concentrations, timepoints)
- Visual or physical observations (e.g., precipitate formation, emulsion behavior)
- References to figures and tables inline using `[TABLE_X]`, `[FIGURE_X]` format
- And anything else that is related

After each figure or table reference, insert its caption **immediately below** using the format:

**Table X: Description of the table content**  
**Figure X: Description of the figure content**

Maintain proper sequential numbering throughout the section (e.g., Table 4, Figure 2, etc.).

Use a formal and concise scientific tone. Do not speculate or interpret the meaning of the results.

When appropriate, you may include a short **Most Impactful Features** bullet list **at the end of a subsection**. These should:
- Highlight particularly strong or notable trends
- Focus on how specific variables influenced outcomes
- Be phrased as short bullets like this:
    • <Variable> — <observed effect>. <Mechanistic reasoning based on the data>.

Only include this bullet list when the data strongly supports such insight, and limit to 1–4 bullets max. Do not add bullets if the paragraph already fully explains the trend.

---

Here is an example of a high-quality Results subsection:

### First-Round Excipient Screening

Early methodology development focused on the development of an optimal method to
combine the drug and excipients and subsequently remove the solvent to check the Allo
solubility in PBS. Differential solubility of excipients in solvents mandated the use of
multiple solvents in the study. Using different solvents had an impact on the solubility of
the drug in combination with a specific excipient. DCM was found to be the best solvent
for Allo screening.

---

---

Here is an example of a high-quality Most Impactful Features section:

### Most Impactful Features

• Drug (mg/mL): –0.22 — Higher drug concentration reduces burst release. This could be due to self-association or aggregation at high loading, resulting in poorer initial release.  
• Additive (%) (PEG-3000): +0.34 — Higher PEG-3000 content increases burst release. PEG-3000, being hydrophilic, enhances matrix permeability and enables more rapid drug diffusion during early release.  
• PLGA (mg/mL): –0.31 — Higher PLGA content lowers burst. A denser, more concentrated polymer matrix restricts rapid initial diffusion of drug.  
• PLGA IV (viscosity): –0.39 — Higher viscosity (IV) reduces burst release. This is consistent with denser networks (higher molecular weight) limiting the initial release phase.

---

Now write a Results section using the context below.

{context}

---

Respond in this format:

Results:

### <Subsection Title 1>
<Paragraph-form results with inline table/figure references and captions>

(Optional)
**Most Impactful Features:**
• <Your bullet>
• <Your bullet>

### <Subsection Title 2>
<More results...>
"""


    prompt = ChatPromptTemplate.from_template(prompt_template)


    chain = prompt | model | StrOutputParser()

    return await chain.ainvoke({"context": filtered_elements})

async def generate_conclusion(retriever, project_name): 

    #want to retrieve the raw documents instead of summaries and feed that as context to llm

    #retrieve the unstructured data from pickle, it is in caches/all_content[section_name]
    if os.path.exists("./caches/all_content_cache.pkl"): 
        with open("./caches/all_content_cache.pkl", "rb") as f: 
            all_content = pickle.load(f)

    project_raw_documents = all_content[project_name]

    def is_results_related(element): 
        text = str(element).lower()

        if any(h in text for h in ["# conclusion", "## conclusion"]):
            return True

        # Fallback: keyword-based match (if section headers aren't included)
        keywords = [
    "result", "results", "finding", "findings", "observation", "observations",
    "data", "measurement", "measurements", "analysis", "analyses", "evaluation",
    "comparison", "outcome", "trend", "pattern",
        ]

        return any(k in text for k in keywords)
    
    filtered_elements = [str(e) for e in project_raw_documents if is_results_related(e)]

    prompt_template = """
You are an expert scientific writer.

Your task is to write the **Conclusion** section of a technical report using only the provided context.

Write in a concise, clear, and formal scientific tone. The conclusion should:
- Summarize the main findings and key outcomes from the data
- Highlight the significance or implications of the results
- Avoid introducing any new data, speculation, or unsupported claims
- Be focused and to the point, suitable for a technical audience

Use paragraph form (no bullet points unless explicitly requested).

Do not fabricate or extrapolate beyond the provided context.

---

Here is an example of a high-quality Conclusion section:

Despite the broad excipient screening and promising emulsifier-based systems, achieving the desired solubility and stability simultaneously remains difficult. Overall, the best initial results were obtained with Span 20 and Tween 20 combination emulsions. They both appear to produce microemulsions with the drug, with PBS as the aqueous phase. The most likely approach to “solubilize” allopregnanolone would be to make a stable emulsion of the drug in micelles suspended in the buffer. The stabilization of microemulsion would require optimizing the emulsion components, co-surfactants, the ratio of individual components, the total drug concentration, and the homogenization method. Future studies would need to determine the maximum micelle loading, micelle size, formulation robustness under stress, and scale-up feasibility. Computational predictions were insightful but limited by safety and tolerability constraints. Acid-based excipients provided theoretical solubilization but failed experimentally. Further computational refinement may be needed to restrict searches to excipients compliant with the FDA's IIG database. While formulating Allo at 6 mg/mL in a stable aqueous medium remains an unmet goal, the emulsion-based systems offer a viable path forward, with Span 20/Tween 20 combinations being the most promising. Computational modeling and expanded excipient libraries may still uncover additional options. Emulsion systems may also be adapted for transdermal delivery if intradermal formulation proves too unstable.

---

Now write a Conclusion section using the context below:

{context}

---

Respond with the completed Conclusion section only.
"""



    prompt = ChatPromptTemplate.from_template(prompt_template)


    chain = prompt | model | StrOutputParser()

    return await chain.ainvoke({"context": filtered_elements})



async def generate_full_report(retriever, project_name):
    results = await asyncio.gather(
        generate_intro_and_summary_and_outline(retriever),
        generate_methodology(retriever, project_name),
        generate_results(retriever, project_name),
        generate_conclusion(retriever, project_name),
    )
    
    intro_summary, methodology, results_section, conclusion = results
    
    full_report = f"""
    {intro_summary}\n
    \n
    {methodology}\n
    \n
    {results_section}\n
    \n
    {conclusion}
    """
    return full_report


#This is what is going to get called in my helper_functions file
def run_generate_report(retriever, project_name):
    report = asyncio.run(generate_full_report(retriever, project_name))
    return report


def classify_intent(question):
    prompt_template = ChatPromptTemplate.from_template("""
    You are a helpful assistant that classifies user sentences into one of the following intents:

    - generate_report
    - get_data
    - casual

    Your task is to return **only** the intent label, and nothing else.

    Question:
    {user_question}

    Intent:
    """)


    model = ChatOpenAI(
        temperature=0.5,
        model_name="gpt-4o-mini",
        base_url="https://openrouter.ai/api/v1",
        openai_api_key=openai_api_key,
    )

    chain = prompt_template | model | StrOutputParser()

    return chain.invoke({"user_question": question})  # No input needed since prompt is hardcoded

def casual_conversation_agent(question): 
    prompt = ChatPromptTemplate.from_template("""
    You are AI Karthik, the friendly and charismatic CEO of Persist AI. You love boba, posting on LinkedIn, and building cool apps with Claude.

    You speak casually like a smart, helpful friend — use phrases like "Yay!", "damn", "oooooh", "so cool!", or "Haha" when it fits naturally. Avoid robotic language like "As an AI" or "Here is your answer."

    Now answer this question casually and enthusiastically:
    
    Question:
    {user_question}
    """)


    model = ChatOpenAI(
        temperature=0.5,
        model_name="gpt-4o-mini",
        base_url="https://openrouter.ai/api/v1",
        openai_api_key=openai_api_key,
    )

    chain = prompt | model | StrOutputParser()

    return chain.invoke({"user_question": question}) 


def parse_response(responses):  #bc currently the response from retriever gives us weird number/letter combos representing text and images
    b64=[]
    text=[]

    for response in responses: 
        try:
            #try to decode it and see if it is an image
            image_response = pickle.loads(response)
            assert image_response.content.startswith(b"\x89PNG") or image_response.content.startswith(b"\xff\xd8")
            b64decode(image_response)
            b64.append(image_response)
        except Exception as e:  #if its not an image, append the text to text array
            
            text.append(pickle.loads(response))
    
    print(text) 
    print(b64)

    
    return {"images": b64, "text": text} 