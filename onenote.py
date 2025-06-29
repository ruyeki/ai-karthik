import json
from pathlib import Path
import requests


#access tokens that give me authorization to use onenote api
token_path = Path.home() / ".credentials" / "onenote_graph_token.json"
token = json.loads(token_path.read_text())["access_token"]

headers = {
    "Authorization": f"Bearer {token}"
}

#CONNECT TO SHAREPOINT
search_url = "https://graph.microsoft.com/v1.0/sites?search=Research"
search_response = requests.get(search_url, headers=headers)
search_data = search_response.json()
site_id = "persistai.sharepoint.com,44d5d7d9-9894-437b-a938-37f26f5ad066,95dec0fb-afca-4a24-af80-faf5e2f9cc75"

#try to open the notebook

#GET THE NOTEBOOK
notebook_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/onenote/notebooks"
notebook_response = requests.get(notebook_url, headers=headers)
notebook_data = notebook_response.json()
#print(json.dumps(notebook_data, indent=2))

#GET THE SECTIONS
sections_url = f"https://graph.microsoft.com/v1.0/sites/persistai.sharepoint.com,44d5d7d9-9894-437b-a938-37f26f5ad066,95dec0fb-afca-4a24-af80-faf5e2f9cc75/onenote/notebooks/1-091c59f7-0e8d-4ac4-9bb6-34e621412f2f/sections"
section_response = requests.get(sections_url, headers=headers)
section_data = section_response.json()
section_id = section_data.get('value', 'id')
print(section_id)

#GET THE PAGES AND THEIR CONTENT IN EACH SECTION
