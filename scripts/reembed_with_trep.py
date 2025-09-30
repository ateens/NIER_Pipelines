#!/usr/bin/env python3
"""
Re-embed existing ChromaDB collection from TS2Vec to T-Rep embeddings.

This script:
1. Fetches all data from the existing TS2Vec collection
2. Re-embeds the time series data using T-Rep models
3. Creates a new collection and stores the T-Rep embeddings
"""

import sys
import os
sys.path.insert(0, '/home/0_code/NIER_Pipelines')

import chromadb
from chromadb.config import Settings
from NIERModules.chroma_trep import TRepEmbedding
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_chromadb_client(host="localhost", port=8000):
    """Connect to ChromaDB server."""
    try:
        # Try newer API (v2)
        client = chromadb.HttpClient(
            host=host,
            port=port,
            settings=Settings(allow_reset=True, anonymized_telemetry=False)
        )
    except Exception as e:
        logger.warning(f"Failed to connect with v2 API: {e}")
        logger.info("Trying direct HTTP connection...")
        # Fallback to basic connection
        import requests
        base_url = f"http://{host}:{port}"
        # Use requests directly to interact with ChromaDB
        client = chromadb.Client(Settings(
            chroma_api_impl="rest",
            chroma_server_host=host,
            chroma_server_http_port=port,
            anonymized_telemetry=False
        ))
    return client


def initialize_trep_embeddings(device="cuda"):
    """Initialize T-Rep embedding functions for each element."""
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
    batch_size=100
):
    """
    Re-embed all data from source collection to target collection.
    
    Args:
        source_collection_name: Name of the existing TS2Vec collection
        target_collection_name: Name of the new T-Rep collection
        host: ChromaDB host
        port: ChromaDB port
        device: Device for T-Rep models ('cuda' or 'cpu')
        batch_size: Batch size for processing
    """
    # Connect to ChromaDB
    logger.info("Connecting to ChromaDB...")
    client = get_chromadb_client(host, port)
    
    # Get source collection
    logger.info(f"Loading source collection: {source_collection_name}")
    try:
        source_collection = client.get_collection(name=source_collection_name)
    except Exception as e:
        logger.error(f"Failed to get source collection: {e}")
        return False
    
    # Get all data from source collection
    logger.info("Fetching all data from source collection...")
    try:
        # Get total count
        count = source_collection.count()
        logger.info(f"Total documents in source collection: {count}")
        
        if count == 0:
            logger.warning("Source collection is empty!")
            return False
        
        # Fetch all data
        results = source_collection.get(
            include=['documents', 'metadatas', 'embeddings']
        )
        
        ids = results['ids']
        documents = results['documents']
        metadatas = results['metadatas']
        
        logger.info(f"Fetched {len(ids)} documents")
        
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        return False
    
    # Initialize T-Rep embeddings
    logger.info("Initializing T-Rep embedding functions...")
    try:
        embedding_functions = initialize_trep_embeddings(device=device)
    except Exception as e:
        logger.error(f"Failed to initialize T-Rep models: {e}")
        return False
    
    # Create target collection
    logger.info(f"Creating target collection: {target_collection_name}")
    try:
        # Delete if exists
        try:
            client.delete_collection(name=target_collection_name)
            logger.info(f"Deleted existing collection: {target_collection_name}")
        except:
            pass
        
        # Create new collection (dimension will be set automatically by first embedding)
        target_collection = client.create_collection(
            name=target_collection_name,
            metadata={"description": "Time series collection with T-Rep embeddings"}
        )
        
    except Exception as e:
        logger.error(f"Failed to create target collection: {e}")
        return False
    
    # Re-embed and insert data
    logger.info("Re-embedding and inserting data...")
    
    success_count = 0
    error_count = 0
    
    # Process in batches
    for i in tqdm(range(0, len(ids), batch_size), desc="Re-embedding"):
        batch_ids = ids[i:i+batch_size]
        batch_docs = documents[i:i+batch_size]
        batch_metas = metadatas[i:i+batch_size]
        
        batch_embeddings = []
        valid_indices = []
        
        for j, (doc_id, doc, meta) in enumerate(zip(batch_ids, batch_docs, batch_metas)):
            try:
                # Get element from metadata
                element = meta.get('element')
                if element not in embedding_functions:
                    logger.warning(f"Unknown element '{element}' for document {doc_id}, skipping")
                    error_count += 1
                    continue
                
                # Re-embed with T-Rep
                embedding_fn = embedding_functions[element]
                embedding = embedding_fn([doc])[0]
                
                batch_embeddings.append(embedding)
                valid_indices.append(j)
                
            except Exception as e:
                logger.error(f"Failed to re-embed document {doc_id}: {e}")
                error_count += 1
                continue
        
        # Insert valid embeddings into target collection
        if batch_embeddings:
            try:
                valid_ids = [batch_ids[j] for j in valid_indices]
                valid_docs = [batch_docs[j] for j in valid_indices]
                valid_metas = [batch_metas[j] for j in valid_indices]
                
                target_collection.add(
                    ids=valid_ids,
                    embeddings=batch_embeddings,
                    documents=valid_docs,
                    metadatas=valid_metas
                )
                
                success_count += len(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Failed to insert batch: {e}")
                error_count += len(batch_embeddings)
    
    # Summary
    logger.info("="*60)
    logger.info("Re-embedding completed!")
    logger.info(f"Successfully processed: {success_count}")
    logger.info(f"Errors: {error_count}")
    logger.info(f"Target collection: {target_collection_name}")
    logger.info(f"Total documents in target: {target_collection.count()}")
    logger.info("="*60)
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Re-embed ChromaDB collection with T-Rep")
    parser.add_argument("--source", default="time_series_collection", help="Source collection name")
    parser.add_argument("--target", default="time_series_collection_trep", help="Target collection name")
    parser.add_argument("--host", default="localhost", help="ChromaDB host")
    parser.add_argument("--port", type=int, default=8000, help="ChromaDB port")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device for T-Rep models")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for processing")
    
    args = parser.parse_args()
    
    logger.info("Starting re-embedding process...")
    logger.info(f"Source: {args.source}")
    logger.info(f"Target: {args.target}")
    logger.info(f"Device: {args.device}")
    
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
