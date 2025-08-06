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
from langchain_community.chat_models import ChatOpenAI
import uuid
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.storage.sql import SQLStore
import json
import pickle
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from base64 import b64decode
from fpdf import FPDF
from langchain_core.output_parsers import StrOutputParser
from unstructured.partition.html import partition_html
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

'''

This file contains the code to build the RAG pipeline.

This code is used for non-report-generation user queries. 

'''

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=api_key)

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro", 
    google_api_key=api_key,
    temperature=0.5
)

image_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro", 
    google_api_key=api_key,
    temperature=0.5
)

#for html files
def process_html(all_contents): 
    results = {}
    for key, value in all_contents.items(): 
        project_name = key
        elements = partition_html(
            text=value,
            infer_table_structure=True,
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
        results[project_name] = elements

    return results



#for pdf files
def process_pdf(file_path): 
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

    #so we sent the pdf file through the unstructured library, which chunked up the pdf into machine readable code 
    #it extracts things like images, tables, text, titles, etc. from the document

    len(chunks) #how many content blocks (chunks) did we extract from the pdf

    chunks[0].metadata.orig_elements #get the first chunk and show the metadata

    texts = []
    tables= []

    for chunk in chunks: 
        if "Table" in str(type(chunk)): 
            tables.append(chunk)
        if "CompositeElement" in str(type(chunk)): 
            texts.append(chunk)


    images_b64=[]

    for chunk in chunks: 
        if "CompositeElement" in str(type(chunk)): 
            chunk_el = chunk.metadata.orig_elements
            chunk_images = [el for el in chunk_el if 'Image' in str(type(el))]
            for image in chunk_images: 
                images_b64.append(image.metadata.image_base64)
    
    return texts, tables, images_b64

def display_base64_image(base64_code): 
    return f"data:image/png;base64,{base64_code}"


def summarize_content(texts, tables, images):

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

    chain = prompt | image_model | StrOutputParser()

    image_summaries = chain.batch(images)

    return text_summaries, table_summaries, image_summaries


#using a free embedding model in huggingface
model_name = "hkunlp/instructor-large"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}

hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

id_key="doc_id"


#This function checks if a project exists and if it doesn't it will create a new vectorstore and docstore for it, then return the retriever
#if it does it will simply connect to the existing ones and return the retriever

def create_new_db(project_name):
    vectorstore=Chroma(collection_name="multi_modal_rag", embedding_function=hf, persist_directory=f"./db/{project_name}/{project_name}_chroma_db") #to store summaries

    vectorstore.persist()

    sql_store = SQLStore(namespace=f"{project_name}_docstore", db_url=f"sqlite:///./db/{project_name}/{project_name}_docstore.db")
    sql_store.create_schema()
    docstore = sql_store


    retriever = MultiVectorRetriever( #MultiVectorRetriever will search the vector store for relevant results, read the doc_id, and returns the related documents from document store
            vectorstore=vectorstore,
            docstore=docstore,
            id_key=id_key
        ) 

    return retriever
    
retriever_cache = {}

def connect_db(project_name):

    if project_name in retriever_cache: #if the project already exists we dont have to connect to it again
        print("Project already in cache, loading from cache")
        return retriever_cache[project_name] #This will return the retriever for the project specified

    base = f"./db/{project_name}"

    if os.path.exists(base):   #if project exists, connect to it, if it doesn't ask the user if they want to create a new project and call create_new_db
        print("Project exists!")

        vectorstore_path = f"./db/{project_name}/{project_name}_chroma_db"
        
        vectorstore=Chroma(collection_name="multi_modal_rag", embedding_function=hf, persist_directory=f"{vectorstore_path}") #to store summaries

        vectorstore.persist()

        sql_store = SQLStore(namespace=f"{project_name}_docstore", db_url=f"sqlite:///./db/{project_name}/{project_name}_docstore.db")
        
        docstore = sql_store
        
        
        retriever = MultiVectorRetriever( #MultiVectorRetriever will search the vector store for relevant results, read the doc_id, and returns the related documents from document store
            vectorstore=vectorstore,
            docstore=docstore,
            id_key=id_key
        )
    
    retriever_cache[project_name] = retriever

    
    return retriever



def store_to_db(retriever, text_summaries, texts, image_summaries, images, tables, table_summaries):

    #add text
    text_ids = [str(uuid.uuid4()) for _ in text_summaries] #creates a unique id for every index in texts

    summary_texts = [
        Document(page_content= summary, metadata={id_key:text_ids[i]}) for i,summary in enumerate(text_summaries)
    ] #creating a list of langchain document objects for each summary with a unique id

    if summary_texts:     
        retriever.vectorstore.add_documents(summary_texts) #add summaries to the vectorstore
        text_pickle = [pickle.dumps(text) for text in texts]
        retriever.docstore.mset(list(zip(text_ids, text_pickle))) #stores full original texts, pairs each id with the full text, mset saves them all at once
    else: 
        print("No texts found. Skipping text insertion.")

    #add tables
    table_ids = [str(uuid.uuid4()) for _ in table_summaries]
    summary_tables = [
        Document(page_content = summary, metadata={id_key: table_ids[i]}) for i, summary in enumerate(table_summaries)
    ]

    if summary_tables: 
        retriever.vectorstore.add_documents(summary_tables)
        table_pickle = [pickle.dumps(table) for table in tables]
        retriever.docstore.mset(list(zip(table_ids, table_pickle)))

    else: 
        print("No tables found. Skipping table insertion.")

    #add images
    image_ids = [str(uuid.uuid4()) for _ in image_summaries]
    summary_images = [
        Document(page_content = summary, metadata={id_key: image_ids[i]}) for i, summary in enumerate(image_summaries)
    ]

    if summary_images: 
        retriever.vectorstore.add_documents(summary_images)
        image_pickle = [pickle.dumps(image) for image in images]
        retriever.docstore.mset(list(zip(image_ids, image_pickle)))
    else: 
        print("No images found. Skipping image insertion.")


def parse_response(responses):  #bc currently the response from retriever gives us weird number/letter combos representing text and images
    b64=[]
    text=[]

    for response in responses: 
        try:
            #try to decode it and see if it is an image
            image_response = pickle.loads(response)
            b64decode(image_response)
            b64.append(image_response)
        except Exception as e:  #if its not an image, append the text to text array
            
            text.append(pickle.loads(response))
    
    print(text) 
    print(b64)

    
    return {"images": b64, "text": text}

def build_prompt(kwargs): #build the prompt that the llm will see, adds all the text and attaches images if needed
    docs_by_type = kwargs["context"]  # the output of parse_docs()
    user_question = kwargs["question"]  # the question being asked


    context_text = ""


    if len(docs_by_type["text"]) > 0:
        for text_element in docs_by_type["text"]:
            context_text += text_element.text

    prompt_template = f"""
        You are a helpful assistant. Your job is to extract accurate, relevant information from the context provided below.

        Only use the provided context to answer the question. Do not guess or fabricate information. If the answer is not in the context, say: "Sorry, I couldn't find that information in the data provided."
        
        Provide images and tables when necessary.
    ---
    Context:
    {context_text}

    Question:
    {user_question}

    ---

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

def query_llm(retriever, question):

    #The chain being used (OLD ONE)
    chain = (
        {
            "context": retriever | RunnableLambda(parse_response) , #runnable lambda allows for running functions with langchain

            "question": RunnablePassthrough(), #allows for questions to go through untouched, what goes in comes out the same way

        }
        | RunnableLambda(build_prompt) | model | StrOutputParser() 
        
        #build the prompt, send it to llm, and get the output as a string
    )

    #BETTER CHAIN USE THIS ONE
    chain_with_sources = {
        "context": retriever | RunnableLambda(parse_response), #find relevant info (documents/tables/images) and run it through parse_response
        "question": RunnablePassthrough(),
    } | RunnablePassthrough().assign(
        response=(
            RunnableLambda(build_prompt) #runs build_pormpt function which combines the context and question into one string for the model
            | model
            | StrOutputParser() #outputs response in a string
        )
    )

    response = chain_with_sources.invoke(question)
            
        #for image in response['context']['images']:
            #image_response.append(display_base64_image(image))
        
    return {
        "text_response": response["response"],
        "image_response": response['context']['images']
    }
        