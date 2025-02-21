import requests


def get_collection_id(vector_db_host: str, vector_db_port: str, collection_name: str):
    url = f"{vector_db_host}:{vector_db_port}/api/v1/collections"
    response = requests.get(url)
    if response.status_code == 200:
        collections = response.json()
        for collection in collections:
            if collection["name"] == collection_name:
                return collection["id"]
        print(
            f"Collection '{collection_name}' not found.")
    else:
        print(
            f"Error fetching collections: {response.status_code} - {response.text}")
    return None

def get_chromadb_collection_id(vector_db_host: str, vector_db_port: str, collection_name: str):
    # Check if the collection exists
    collection_id = get_collection_id(vector_db_host, vector_db_port, collection_name)
    # If the collection does not exist kill the pipeline
    if collection_id is None:
        print(f"Collection '{collection_name}' not found. Check Collection Name.")
        exit(1)
        
    print(f"Collection '{collection_name}' found with ID: {collection_id}")
    return collection_id