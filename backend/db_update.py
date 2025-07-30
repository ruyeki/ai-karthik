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
