
import requests
from datetime import datetime


def query_chromadb(vector_db_host: str,
                   vector_db_port: str,
                   collection_id: str,
                   embedding: list,
                   top_k: int) -> dict:
    """
    Step 4: Query ChromaDB for similar data

    Args:


    Raises:
        RuntimeError: If the query to ChromaDB fails.

    Returns:
        dict: The response from ChromaDB.
    """

    payload = {
        "query_embeddings": [embedding],
        "n_results": top_k,
        "include": ["metadatas", "documents", "distances"]
    }
    
    url = f"{vector_db_host}:{vector_db_port}/api/v1/collections/{collection_id}/query"
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        # print("###QUERY_CHROMADB### Response:", response.json())
        return response.json()
    else:
        raise RuntimeError(
            f"Error querying ChromaDB: {response.status_code} - {response.text}")


def query_chromadb_with_filter(
    vector_db_host: str,
    vector_db_port: str,
    collection_id: str,
    embedding: list, 
    target_element: str,
    vector_db_top_k:int) -> dict:
    """
    Query ChromaDB recursively until top_k samples with the same element are collected.

    Args:
        embedding (list): The embedding vector of the time-series data.
        target_element (str): The element to filter for.
        top_k (int): Number of results to retrieve with the same element.

    Returns:
        dict: Filtered results with the same element.
    """

    # Note: top_k of query_chromadb is set to 300 to ensure that we get enough results
    # After receiving the results, we filter them by the target element
    # This behavior is necessary because Version 1.0 of ts2vec collection contains mixed elements
    response = query_chromadb(
        vector_db_host,
        vector_db_port,
        collection_id,
        embedding, 
        top_k=300)
    
    filtered_results = {
        "metadatas": [],
        "ids": [],
        "distances": [],
    }

    # Filter results by element and avoid duplicates
    print("Query ChromaDB with filter")

    for metadata, record_id, distance in zip(response["metadatas"][0], response["ids"][0], response["distances"][0]):
        if metadata["element"] == target_element:
            is_duplicate = any(is_overlapping(metadata, existing)
                               for existing in filtered_results["metadatas"])

            if not is_duplicate:
                filtered_results["metadatas"].append(metadata)
                filtered_results["ids"].append(record_id)
                filtered_results["distances"].append(distance)

        if len(filtered_results["metadatas"]) >= vector_db_top_k:
            break

    return filtered_results


def is_overlapping(new_metadata, existing_metadata):
    """
    새로운 데이터와 기존 데이터가 겹치는지 확인하는 함수.
    겹치는 기준: 동일한 region & element + start~end 기간이 겹치는 경우
    """
    new_start = datetime.strptime(new_metadata["start"], "%Y-%m-%d %H:%M:%S")
    new_end = datetime.strptime(new_metadata["end"], "%Y-%m-%d %H:%M:%S")

    existing_start = datetime.strptime(
        existing_metadata["start"], "%Y-%m-%d %H:%M:%S")
    existing_end = datetime.strptime(
        existing_metadata["end"], "%Y-%m-%d %H:%M:%S")

    return (
        new_metadata["region"] == existing_metadata["region"]
        and new_metadata["element"] == existing_metadata["element"]
        # 겹치는 구간 확인
        and not (new_end < existing_start or new_start > existing_end)
    )
