import requests

def fetch_content_item(base_path, content_store="https://www.gov.uk/api/content"):
     return requests.get(content_store + base_path).json()
