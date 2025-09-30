#!/usr/bin/env python3
"""
Re-embed existing ChromaDB collection from TS2Vec to T-Rep embeddings using REST API.
"""

import sys
import os
sys.path.insert(0, '/home/0_code/NIER_Pipelines')

import requests
import json
import numpy as np
from NIERModules.chroma_trep import TRepEmbedding
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_collection_id(host, port, collection_name):
    """Get collection ID by name."""
    url = f"http://{host}:{port}/api/v1/collections"
    response = requests.get(url)
    response.raise_for_status()
    
    collections = response.json()
    for coll in collections:
        if coll['name'] == collection_name:
            return coll['id']
    
    raise ValueError(f"Collection '{collection_name}' not found")


def get_collection_data(host, port, collection_id):
    """Fetch all data from collection."""
    url = f"http://{host}:{port}/api/v1/collections/{collection_id}/get"
    
    # Get all data
    payload = {
        "include": ["documents", "metadatas", "embeddings"]
    }
    
    response = requests.post(url, json=payload)
    response.raise_for_status()
    
    return response.json()


def create_collection(host, port, collection_name):
    """Create a new collection."""
    url = f"http://{host}:{port}/api/v1/collections"
    
    # Check if exists and delete
    try:
        coll_id = get_collection_id(host, port, collection_name)
        delete_url = f"http://{host}:{port}/api/v1/collections/{coll_id}"
        requests.delete(delete_url)
        logger.info(f"Deleted existing collection: {collection_name}")
    except:
        pass
    
    # Create new
    payload = {
        "name": collection_name,
        "metadata": {"description": "Time series collection with T-Rep embeddings"}
    }
    
    response = requests.post(url, json=payload)
    response.raise_for_status()
    
    return response.json()['id']


def add_to_collection(host, port, collection_id, ids, embeddings, documents, metadatas):
    """Add data to collection."""
    url = f"http://{host}:{port}/api/v1/collections/{collection_id}/add"
    
    payload = {
        "ids": ids,
        "embeddings": embeddings,
        "documents": documents,
        "metadatas": metadatas
    }
    
    response = requests.post(url, json=payload)
    response.raise_for_status()


def initialize_trep_embeddings(device="cuda"):
    """Initialize T-Rep embedding functions."""
    model_paths = {
        "SO2": "/home/0_code/NIER_Pipelines/NIERModules/chroma_trep/model_pkl/model_SO2.pt",
        "O3": "/home/0_code/NIER_Pipelines/NIERModules/chroma_trep/model_pkl/model_O3.pt",
        "CO": "/home/0_code/NIER_Pipelines/NIERModules/chroma_trep/model_pkl/model_CO.pt",
        "NO": "/home/0_code/NIER_Pipelines/NIERModules/chroma_trep/model_pkl/model_NO.pt",
        "NO2": "/home/0_code/NIER_Pipelines/NIERModules/chroma_trep/model_pkl/model_NO2.pt"
    }
    
    embedding_functions = {}
    for elem, path in model_paths.items():
        logger.info(f"Loading T-Rep model for {elem}...")
        embedding_functions[elem] = TRepEmbedding(
            weight_path=path,
            device=device,
            encoding_window='full_series',
            time_embedding='t2v_sin'
        )
    
    return embedding_functions


def reembed_collection(
    source_collection_name="time_series_collection",
    target_collection_name="time_series_collection_trep",
    host="localhost",
    port=8000,
    device="cuda",
    batch_size=50
):
    """Re-embed collection data."""
    
    # Get source collection
    logger.info(f"Fetching source collection: {source_collection_name}")
    source_id = get_collection_id(host, port, source_collection_name)
    logger.info(f"Source collection ID: {source_id}")
    
    # Fetch all data
    logger.info("Fetching all data...")
    data = get_collection_data(host, port, source_id)
    
    ids = data['ids']
    documents = data['documents']
    metadatas = data['metadatas']
    
    logger.info(f"Fetched {len(ids)} documents")
    
    if len(ids) == 0:
        logger.warning("Source collection is empty!")
        return False
    
    # Initialize T-Rep
    logger.info("Initializing T-Rep models...")
    embedding_functions = initialize_trep_embeddings(device=device)
    
    # Create target collection
    logger.info(f"Creating target collection: {target_collection_name}")
    target_id = create_collection(host, port, target_collection_name)
    logger.info(f"Target collection ID: {target_id}")
    
    # Re-embed and insert
    logger.info("Re-embedding data...")
    success_count = 0
    error_count = 0
    
    for i in tqdm(range(0, len(ids), batch_size), desc="Processing"):
        batch_ids = ids[i:i+batch_size]
        batch_docs = documents[i:i+batch_size]
        batch_metas = metadatas[i:i+batch_size]
        
        batch_embeddings = []
        valid_indices = []
        
        for j, (doc_id, doc, meta) in enumerate(zip(batch_ids, batch_docs, batch_metas)):
            try:
                element = meta.get('element')
                if element not in embedding_functions:
                    logger.warning(f"Unknown element '{element}' for {doc_id}")
                    error_count += 1
                    continue
                
                # Re-embed
                embedding_fn = embedding_functions[element]
                embedding = embedding_fn([doc])[0]
                
                # Convert numpy array to list for JSON serialization
                if hasattr(embedding, 'tolist'):
                    embedding = embedding.tolist()
                elif isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()
                
                batch_embeddings.append(embedding)
                valid_indices.append(j)
                
            except Exception as e:
                logger.error(f"Failed to re-embed {doc_id}: {e}")
                error_count += 1
        
        # Insert batch
        if batch_embeddings:
            try:
                valid_ids = [batch_ids[j] for j in valid_indices]
                valid_docs = [batch_docs[j] for j in valid_indices]
                valid_metas = [batch_metas[j] for j in valid_indices]
                
                add_to_collection(host, port, target_id, valid_ids, batch_embeddings, valid_docs, valid_metas)
                success_count += len(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Failed to insert batch: {e}")
                error_count += len(batch_embeddings)
    
    # Summary
    logger.info("="*60)
    logger.info("Re-embedding completed!")
    logger.info(f"Successfully processed: {success_count}")
    logger.info(f"Errors: {error_count}")
    logger.info(f"Total: {success_count + error_count} / {len(ids)}")
    logger.info("="*60)
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Re-embed ChromaDB collection with T-Rep")
    parser.add_argument("--source", default="time_series_collection", help="Source collection name")
    parser.add_argument("--target", default="time_series_collection_trep", help="Target collection name")
    parser.add_argument("--host", default="localhost", help="ChromaDB host")
    parser.add_argument("--port", type=int, default=8000, help="ChromaDB port")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size")
    
    args = parser.parse_args()
    
    logger.info("Starting re-embedding process...")
    logger.info(f"Source: {args.source}")
    logger.info(f"Target: {args.target}")
    logger.info(f"Device: {args.device}")
    
    try:
        success = reembed_collection(
            source_collection_name=args.source,
            target_collection_name=args.target,
            host=args.host,
            port=args.port,
            device=args.device,
            batch_size=args.batch_size
        )
        
        if success:
            logger.info("✅ Re-embedding completed successfully!")
            sys.exit(0)
        else:
            logger.error("❌ Re-embedding failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
