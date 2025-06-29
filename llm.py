from dotenv import load_dotenv
import os
from unstructured.partition.pdf import partition_pdf
from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFSyntaxError
import base64
from IPython.display import Image, display
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import ChatOpenAI
import uuid
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from base64 import b64decode
import streamlit as st


load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

file_path = './report.pdf'


#parses through pdfs, returns everything in pdf into a single vector (Images, tables, text)
chunks = partition_pdf(
    filename=file_path,
    infer_table_structure=True, #allows you to extract table from documents
    strategy="hi_res", #needed to extract tables

    extract_image_block_types=["Image", "Table"],
    extract_image_block_to_payload=True, #extract images and they will have metadata element that contains base64 of image, needed to send to LLM
    size = {
  "longest_edge": 1500,  # max pixels for longest side (width or height)
    },

    #enable chunking by title
    #all elements inside a title will be under a single chunk, useful for RAG
    chunking_strategy = "by_title",
    max_characters=10000, #max size of chunk is 10,000 characters
    combine_text_under_n_chars=2000, #
    new_after_n_chars=6000,

)

len(chunks) #how many content blocks (chunks) did we extract from the pdf

chunks[0].metadata.orig_elements #get the first chunk and show the metadata

texts = []
tables= []

for chunk in chunks: 
    if "Table" in str(type(chunk)): 
        tables.append(chunk)
    if "CompositeElement" in str(type(chunk)): 
        texts.append(chunk)

def get_images_base64(chunks): 
    images_b64=[]

    for chunk in chunks: 
        if "CompositeElement" in str(type(chunk)): 
            chunk_el = chunk.metadata.orig_elements
            chunk_images = [el for el in chunk_el if 'Image' in str(type(el))]
            for image in chunk_images: 
                images_b64.append(image.metadata.image_base64)
    
    return images_b64

images = get_images_base64(chunks)
print(images)

            #chunk_images = [el for el in elements if 'Image' in str(type(el))]
#chunk_images[0].to_dict() #image_base64 is what is going to get sent to our llm

def display_base64_image(base64_code): 
    image_data = base64.b64decode(base64_code)
    display(Image(data=image_data))

display_base64_image(images[2])

prompt_text = """
You are an assistant tasked with summarizing tables and text.
Give a concise summary of the table or text.

Respond only with the summary, no additional comment.
Do not start your message by saying "Here is a summary" or anything like that.
Just give the summary as it is.

Table or text chunk: {element}
"""

#element will be filled with the actual table/text content

prompt = ChatPromptTemplate.from_template(prompt_text) #turns the prompt into a LangChain prompt object so it can be used with the model

model = ChatOpenAI( #define the model we will use
    temperature=0.5,
    model_name="google/gemini-2.5-pro-preview",
    base_url="https://openrouter.ai/api/v1",
    openai_api_key = api_key
)

summarize_chain = prompt | model | StrOutputParser() #this creates a processing chain where the input data (prompt) gets sent to the model and the output gets extracted with StrOutputParser

#summarize the text
text_summaries = summarize_chain.batch(texts, {"max_concurrency": 3}) #texts is an array that contains chunks of text, basically you get a summary of each chunk and it does it 3 at a time

#summarize the tables
tables_html = [table.metadata.text_as_html for table in tables]
table_summaries = summarize_chain.batch(tables_html, {"max_concurrency": 3}) 


#This will be done differenlty than text and tables

#Change this template later to fit what Kenan wants (refer to Zelda report)
prompt_template = """
Describe the image in detail. For context, the image is part of an experiment that was conducted. Be specific about graphs, such as bar plots.
"""


image_model = ChatOpenAI( #define the model we will use
    temperature=0.5,
    model_name="gpt-4o-mini", #need to use gpt for this since it can take in image input
    base_url="https://openrouter.ai/api/v1",
    openai_api_key = api_key
)

messages = [
    (
        "user",
        [
            {"type": "text", "text": prompt_template},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image}"},},
        ],
    )
]

prompt = ChatPromptTemplate.from_messages(messages)

chain = prompt | model | StrOutputParser()

image_summaries = chain.batch(images)

#need to use openai api key to access embedding models, openrouter does not support embedding models
#using a free embedding model

model_name = "hkunlp/instructor-large"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}

hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


vectorstore=Chroma(collection_name="multi_modal_rag", embedding_function=hf) #to store summaries

store = InMemoryStore() #to store documents

id_key="doc_id"


retriever = MultiVectorRetriever( #MultiVectorRetriever will search the vector store for relevant results, read the doc_id, and returns the related documents from document store
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key
) 

text_ids = [str(uuid.uuid4()) for _ in text_summaries] #creates a unique id for every index in texts
summary_texts = [
    Document(page_content= summary, metadata={id_key:text_ids[i]}) for i, summary in enumerate(text_summaries)
] #creating a list of langchain document objects for each summary with a unique id

if summary_texts:     
    retriever.vectorstore.add_documents(summary_texts) #add summaries to the vectorstore
    retriever.docstore.mset(list(zip(text_ids, texts))) #stores full original texts, pairs each id with the full text, mset saves them all at once
else: 
    print("No texts found. Skipping text insertion.")

#add tables
table_ids = [str(uuid.uuid4()) for _ in table_summaries]
summary_tables = [
    Document(page_content = summary, metadata={id_key: table_ids[i]}) for i, summary in enumerate(table_summaries)
]

if summary_tables: 
    retriever.vectorstore.add_documents(summary_tables)
    retriever.docstore.mset(list(zip(table_ids, tables)))
else: 
    print("No tables found. Skipping table insertion.")

#add images
image_ids = [str(uuid.uuid4()) for _ in image_summaries]
summary_images = [
    Document(page_content = summary, metadata={id_key: image_ids[i]}) for i, summary in enumerate(image_summaries)
]

if summary_images: 
    retriever.vectorstore.add_documents(summary_images)
    retriever.docstore.mset(list(zip(image_ids, images)))
else: 
    print("No images found. Skipping image insertion.")


def parse_response(responses):  #bc currently the response from retriever gives us weird number/letter combos representing text and images
    b64=[]
    text=[]

    for response in responses: 
        try:
            #try to decode it and see if it is an image
            b64decode(response)
            b64.append(response)
        except Exception as e:  #if its not an image, append the text to text array
            text.append(response)
    
    return {"images": b64, "text": text}


def build_prompt(kwargs): #build the prompt that the llm will see, adds all the text and attaches images if needed
    docs_by_type = kwargs["context"]  # the output of parse_docs()
    user_question = kwargs["question"]  # the question being asked

    context_text = ""
    # Add all the text content together
    if len(docs_by_type["text"]) > 0:
        for text_element in docs_by_type["text"]:
            context_text += text_element.text  # assume these are Document objects

    # Now build the actual prompt string
    prompt_template = f"""
    Answer the question based only on the following context, which can include text, tables, and the below image.
    Context: {context_text}
    Question: {user_question}
    """

    prompt_content = [{"type": "text", "text": prompt_template}]  # first item: text

    # If there are images, add them too
    if len(docs_by_type["images"]) > 0:
        for image in docs_by_type["images"]:
            prompt_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            })

    # Wrap everything into a LangChain chat prompt
    return ChatPromptTemplate.from_messages(
        [HumanMessage(content=prompt_content)]
    )



model = ChatOpenAI( #define the model we will use
    temperature=0.5,
    model_name="gpt-4o-mini", #has to be gpt 4o mini since it can take in images and text
    base_url="https://openrouter.ai/api/v1",
    openai_api_key = api_key

)

#The chain being used
chain = (
    {
        "context": retriever | RunnableLambda(parse_response) , #runnable lambda allows for running functions with langchain

        "question": RunnablePassthrough(), #allows for questions to go through untouched, what goes in comes out the same way

    }
    | RunnableLambda(build_prompt) | model | StrOutputParser() 
     
    #build the prompt, send it to llm, and get the output as a string
)


chain_with_sources = {
    "context": retriever | RunnableLambda(parse_response),
    "question": RunnablePassthrough(),
} | RunnablePassthrough().assign(
    response=(
        RunnableLambda(build_prompt)
        | model
        | StrOutputParser()
    )
)


#really fast and simple streamlit frontend for testing

st.title("Final Report Generator")

query = st.text_input("Ask a question about your experiments or generate a report.")

if query:
    response = chain_with_sources.invoke(query)
    for i,text in response['context']['text']: 
        st.markdown(f"### Result {i+1}")
        st.write(text.text)
    
    for image in response['context']['images']:
        st.image(display_base64_image(image))
