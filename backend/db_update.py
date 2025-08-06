import requests
import json
from dotenv import load_dotenv
import os
import json
from pathlib import Path
import requests
import time
import base64
from unstructured.partition.html import partition_html
from unstructured.documents.elements import Image
import pickle
import sqlite3
from models import Projects
from app import app
from urllib.parse import quote
import time
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import ChatOpenAI
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from utils import connect_db, create_new_db
from langchain_core.documents import Document

'''

This is the script that automatically updates the content cache and vector/docstore databases.

The content cache is what is used as context for the LLMs when generating a report. It contains the HTML content of each page in the OneNotebook.

The vector and docstores are used to retrieve data for non-report-generating user queries. 

'''

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=api_key)

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro", 
    google_api_key=api_key,
    temperature=0.5
)


with app.app_context(): 
    all_projects = Projects.query.all()

def add_project_to_db(project_name): 
    conn = sqlite3.connect('./db/project_db.sqlite')
    cursor = conn.cursor()

    cursor.execute('INSERT OR IGNORE INTO projects (name, file_path) VALUES (?, ?)', (project_name, '/db') )
    
    conn.commit()
    conn.close()

def refresh_access_token(token, token_path): 
    print("Refreshing access token...")
    scopes = ['User.Read']

    tenant_id = "4532a721-a36d-41e1-8367-80fd926092a2"
    client_id = "4d7dd08e-dc2d-4334-961f-93506b22e435"
    client_secret = "eRJ8Q~DzN7Rxkzyihq0mQedZuXFYzqFXETz4CaEB"
    refresh_token = token['refresh_token']

    token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"

    # Data sent to Microsoft to get a new access token
    data = {
        "grant_type": "refresh_token",      # Tells MS you want to refresh a token
        "client_id": client_id,              # Your app ID
        "refresh_token": refresh_token,      # The long-lived token you got earlier
        "scope": "https://graph.microsoft.com/.default"  # Permissions you want
    }

    response = requests.post(token_url, data)

    if response.ok: 
        new_token = response.json()
        print("Token successfully refreshed!")

        with open(token_path, "w") as f: 
            json.dump(new_token, f, indent=2)

        return new_token
    
    else: 
        raise Exception("Failed to refresh token", response.text)


#CHATGPT Generated function
def check_response(resp, context):
    if not resp.ok:
        print(f"[ERROR] Failed {context}: {resp.status_code} - {resp.text}")
        resp.raise_for_status()

# Access tokens that give authorization to use OneNote API
token_path = Path.home() / ".credentials" / "onenote_graph_token.json"
with open(token_path, "r") as f:
    token = json.load(f)

headers = {
    "Authorization": f"Bearer {token['access_token']}"
}

# Connect to SharePoint
search_url = "https://graph.microsoft.com/v1.0/sites?search=Research"
search_response = requests.get(search_url, headers=headers)

# If token is expired, refresh it
if search_response.status_code == 401:
    new_token = refresh_access_token(token, token_path)
    headers = {
        "Authorization": f"Bearer {new_token['access_token']}"
    }
    search_response = requests.get(search_url, headers=headers)

search_data = search_response.json()
site_id = "persistai.sharepoint.com,44d5d7d9-9894-437b-a938-37f26f5ad066,95dec0fb-afca-4a24-af80-faf5e2f9cc75"

# Get the notebook
notebook_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/onenote/notebooks"
notebook_response = requests.get(notebook_url, headers=headers)
check_response(notebook_response, "Get notebook")
notebook_data = notebook_response.json()

# Get the sections
sections_url = (
    "https://graph.microsoft.com/v1.0/sites/"
    "persistai.sharepoint.com,44d5d7d9-9894-437b-a938-37f26f5ad066,"
    "95dec0fb-afca-4a24-af80-faf5e2f9cc75/onenote/notebooks/"
    "1-091c59f7-0e8d-4ac4-9bb6-34e621412f2f/sections"
)

section_response = requests.get(sections_url, headers=headers)
check_response(search_response, "Get sections")
section_data = section_response.json()
section_id = section_data.get("value", "id")
sections = section_data.get("value", [])

page_url = []
content_url = []
all_content = {}

changed_pages = []

#WARNING: This code block takes a while to complete, around 5-10 minutes

all_projects_list = [p.name for p in all_projects]

with open("./caches/all_content_cache.pkl", "rb") as f:
    all_content = pickle.load(f)


#======Create new databases for new projects=======

for section in sections:
    section_name = section.get("displayName")

    cached_pages = {p["id"]: p for p in all_content[section_name]}
    #print(cached_pages)

    #check if project is in sqlite db
    if section_name not in all_projects_list: 
        print("New project detected, adding to database...")
        create_new_db(section_name)
        add_project_to_db(section_name) #this adds the db to sqlite


#===================================================

#======= Update all_content cache ==================

    pages_url = section.get("pagesUrl") #this will return all the page content for each section

    page_response = requests.get(pages_url, headers=headers)

    if page_response.status_code == 401: #if token expires, refresh token
            new_token = refresh_access_token(token, token_path)
            headers = {
                "Authorization": f"Bearer {new_token['access_token']}"
            }
            page_response = requests.get(pages_url, headers = headers)

    time.sleep(1)

    page_data = page_response.json()
    contents_url = page_data.get("value", [])

    for url in contents_url: #url contains lastModifiedDate
            content = url.get("contentUrl")
            
            #page id and last modified date of fetched page data for a given section
            id = url.get("id")
            last_modified = url.get("lastModifiedDateTime")

            if id in cached_pages: 
                cached_last_modified = cached_pages[id]["lastModifiedDate"]

                if last_modified == cached_last_modified: #means nothing has changed
                    continue

                else: 
                    print(f"Change detected, updating cache for Section: {section_name}...")

                    #update the page in the cache with the latest changes, fetch the pages content here
                    if content: 
                        content_response = requests.get(content, headers=headers) #gets the html content of the page

                        if content_response.status_code == 401: #if token expires, refresh token
                            new_token = refresh_access_token(token, token_path)
                            headers = {
                                "Authorization": f"Bearer {new_token['access_token']}"
                            }
                            content_response = requests.get(pages_url, headers = headers)
                            
                        time.sleep(1)


                        if not content_response.ok:
                            print(
                                f"[WARN] Failed to fetch content for {section_name} – "
                                f"status {content_response.status_code}"
                            )

                        if content_response.ok:
                            content_data = content_response.text
                            cached_pages[id]['content'] = content_data
                            cached_pages[id]['lastModifiedDate'] = last_modified

                            changed_pages.setdefault(section_name, []).append({
                                "id": id,
                                "content": content_data
                            })
 
            else: #if id is not in the cached_pages, that means a new page was made
                print(f"New page detected, adding to cache for Section: {section_name}...")
                if content: 
                        content_response = requests.get(content, headers=headers) #gets the html content of the page

                        if content_response.status_code == 401: #if token expires, refresh token
                            new_token = refresh_access_token(token, token_path)
                            headers = {
                                "Authorization": f"Bearer {new_token['access_token']}"
                            }
                            content_response = requests.get(pages_url, headers = headers)
                            
                        time.sleep(1)


                        if not content_response.ok:
                            print(
                                f"[WARN] Failed to fetch content for {section_name} – "
                                f"status {content_response.status_code}"
                            )

                        if content_response.ok:
                            content_data = content_response.text

                            cached_pages[id] = {
                                "id": id,
                                "content": content_data,
                                "lastModifiedDate": last_modified
                            }

                            changed_pages.setdefault(section_name, []).append({
                                "id": id,
                                "content": content_data
                            })

    all_content[section_name] = list(cached_pages.values())

#now that everything is updated in cached_pages, replace the pickle cache
with open("./caches/all_content_cache.pkl", "wb") as f:
    pickle.dump(all_content, f)
    print("Successfully updated cache!")

#=======================================================================


#==============Vector/Docstore updates ================================

results = {}


for section, pages in changed_pages.items(): 
    for page in pages: 
        page_id = page["id"]
        content = page["content"]
        all_elements = []

        elements = partition_html(text=content, metadata_include_orig_elements=True, chunking_strategy = "by_title", max_characters = 10000, combine_text_under_n_chars = 2000, new_after_n_chars=6000) #using unstructured to partition html
        all_elements.append({
            "id": page_id,
            "content": elements
        })

    results[section] = all_elements




texts = {}
tables = {}
images_b64 = {}

for section, pages in results.items(): 
    for page in pages: 
        content = page["content"]
        id = page["id"]

        for element in content: 

            if hasattr(element.metadata, "orig_elements"): 
                for sub_element in element.metadata.orig_elements: 
                        if "Table" in str(type(sub_element)):
                            tables.setdefault(section, []).append({
                                "id": id,
                                "content": element})

                if "CompositeElement" in str(type(element)): 
                            texts.setdefault(section, []).append({
                                 "id": id,
                                 "content": element
                            })

                            chunk_el = element.metadata.orig_elements
                            chunk_images = [el for el in chunk_el if 'Image' in str(type(el))]
                            #print(chunk_images)

                            for image in chunk_images:
                                images_b64.setdefault(section, []).append({
                                     "id": id,
                                     "image": image
                                })


final_images_b64 = {}

#CHATGPT GENERATED FUNCTION TO MAKE URLS FIT THE REQUEST FORMAT
def fix_graph_url(bad_url: str) -> str:
    # Step 1: Replace 'siteCollections' with 'sites'
    fixed_url = bad_url.replace("/siteCollections/", "/sites/")

    # Step 2: Extract the site ID portion (between 'sites/' and next slash)
    prefix = "sites/"
    start = fixed_url.find(prefix) + len(prefix)
    end = fixed_url.find("/", start)
    site_id = fixed_url[start:end]

    # Step 3: URL-encode the site ID because it has commas
    encoded_site_id = quote(site_id, safe='')

    # Step 4: Replace the original site ID with the encoded one
    fixed_url = fixed_url[:start] + encoded_site_id + fixed_url[end:]

    return fixed_url

for section, images in images_b64.items(): 
    for image_data in images: 
        image_obj = image_data["image"]
        image_id = image_data["id"]
        
        try: 
                
            for obj in image_obj:
                old_url = obj.metadata.to_dict().get("image_url")
                url = fix_graph_url(old_url)
                response = requests.get(url, headers=headers) 


                if response.status_code == 401: #if token expires, refresh token
                        new_token = refresh_access_token(token, token_path)
                        token = new_token
                        headers = {
                            "Authorization": f"Bearer {new_token['access_token']}"
                        }
                        response = requests.get(url, headers=headers) #retry
                        
                        if not response.ok: #if refreshing fails, give up lol
                            print(f"[FAIL] Still couldn't fetch image after refreshing token. Status: {response.status_code}")
                            continue
                
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    wait_time = int(retry_after) if retry_after else 10
                    print(f"[WARN] 429 rate limit. Waiting {wait_time}s before retrying {url}")
                    time.sleep(wait_time)
                    continue #just skip this current image, change later so it will retry image


                if response.ok:
                    b64 = base64.b64encode(response.content).decode("utf-8")
                    final_images_b64.setdefault(section_name, []).append({
                        "id": image_id,
                        "base64": b64,
                        "source": url
                    })
                    print("Success!")
                else:
                    print(f"[WARN] Failed to fetch image, status {response.status_code}")
                
                time.sleep(5)  
        
        except Exception as e: 
            print(f"[ERROR] Exception while fetching image: {e}")

#NOW SUMMARIZE THE DATA 

prompt_text = """
You are a scientific assistant summarizing experimental data from research text or tables.

Your task is to write a detailed, precise summary that includes:
- The experimental goal or objective
- All key materials, solvents, drugs, excipients, and concentrations used
- Methods or procedures (brief but clear)
- All reported numerical results (e.g. solubility values, volumes, mg/mL, amounts used, etc.)
- Any specific observations, inconsistencies, or outcomes mentioned
- Conclusions or planned next steps

Do NOT omit numerical data or details.
Respond only with the summary in paragraph form. Do NOT add comments or say "Here's the summary".

Text or table to summarize:
{element}
"""


#element will be filled with the actual table/text content

prompt = ChatPromptTemplate.from_template(prompt_text) #turns the prompt into a LangChain prompt object so it can be used with the model

summarize_chain = prompt | model | StrOutputParser()

text_summaries_all = {}
table_summaries_all = {}


for section, content in texts: 
     
     for text in content: 
          text_content = text["content"]
          text_id = text["id"]

          summaries = summarize_chain.batch(text_content, {"max_concurrency": 3})

          text_summaries_all.setdefault(section, []).append({
               "id": text_id,
               "text_summary": summaries
          })

for section, content in tables: 
     for tables in content: 
        table_content = tables["content"]
        table_id = tables["id"]

        summaries = summarize_chain.batch(table_content, {"max_concurrency": 3})

        table_summaries_all.setdefault(section, []).append({
             "id": table_id,
             "table_summary": summaries
        })




image_summaries_all = {}

#Change this template later to fit what Kenan wants (refer to Zelda report)
prompt_template = """
You are an assistant analyzing images from a scientific experiment.

Describe the image in detail. If the image contains a graph (such as a bar plot, scatter plot, or line chart), explain:
- What type of graph it is.
- What is on the x-axis and y-axis.
- What the bars, points, or lines represent.
- Any visible trends, labels, or anomalies.

If the image shows equipment (like vials, microplates, etc.), describe:
- How many items there are.
- Any labels or markings.
- The arrangement and setup.

If the image is anything else, describe it in detail. Be specific and concise.

Only give the description. Do not include extra comments or explanations.
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

chain = prompt | model | StrOutputParser()


for section_name, content in final_images_b64.items():
    all_base64 = []

    for image in content: 
        id = image["id"]

        b64 = image["base64"]

        source_url = image["source"]

        if b64 and b64.strip():
            all_base64.append({"id": id, "b64": b64})
            


    if not all_base64:
            print(f"No valid images found for {section_name}, skipping.")
            continue
    
    b64_list = [img["b64"] for img in all_base64]

    summaries = chain.batch(b64_list)

    image_summaries = [
        {
            "id": all_base64[i]["id"],
            "image_summary": summaries[i]
        }
        for i in range(len(summaries))
    ]

    image_summaries_all.setdefault(section_name, []).extend(image_summaries)
    
    print(f"Image summary complete for {section_name}")



#What i want to do is look through the summaries and documents i produced and find where the ids match in the current docstore and vectorstore and replace it
#Ids represent individual pages

#for text
for section_name, content in text_summaries_all.items(): 

    retriever = connect_db(section_name)

    for page in content: 
        id = page["id"]
        text_summary = page["text_summary"]
        stored_vector_ids = retriever.vectorstore.index_to_docstore_id

        text_id = f"text::{id}"

        matching_ids = [vector_id for id in stored_vector_ids if text_id == id]

        if text_id in stored_vector_ids: 
             retriever.vectorstore.delete(ids = [doc["metadata"]["id"] for doc in ] )
            
#=======================================================================