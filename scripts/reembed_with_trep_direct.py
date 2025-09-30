#!/usr/bin/env python3
"""
Re-embed existing ChromaDB collection from TS2Vec to T-Rep embeddings using direct file access.
"""

import sys
import os
sys.path.insert(0, '/home/0_code/NIER_Pipelines')

import chromadb
from chromadb.config import Settings
import numpy as np
from NIERModules.chroma_trep import TRepEmbedding
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
    db_path="/home/0_code/RAG/ChromaDB/ts2vec_embedding_function/NIER_DB",
    source_collection_name="time_series_collection",
    target_collection_name="time_series_collection_trep",
    device="cuda",
    batch_size=50
):
    """Re-embed collection data using direct file access."""
    
    # Connect to ChromaDB using PersistentClient
    logger.info(f"Connecting to ChromaDB at {db_path}...")
    client = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(anonymized_telemetry=False)
    )
    
    # Get source collection
    logger.info(f"Loading source collection: {source_collection_name}")
    try:
        source_collection = client.get_collection(name=source_collection_name)
    except Exception as e:
        logger.error(f"Failed to get source collection: {e}")
        return False
    
    # Get all data
    logger.info("Fetching all data from source collection...")
    try:
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
    
    # Initialize T-Rep
    logger.info("Initializing T-Rep models...")
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
        
        # Create new collection
        target_collection = client.create_collection(
            name=target_collection_name,
            metadata={"description": "Time series collection with T-Rep embeddings"}
        )
        
    except Exception as e:
        logger.error(f"Failed to create target collection: {e}")
        return False
    
    # Re-embed and insert
    logger.info("Re-embedding data...")
    success_count = 0
    error_count = 0
    skip_count = 0
    
    for i in tqdm(range(0, len(ids), batch_size), desc="Processing"):
        batch_ids = ids[i:i+batch_size]
        batch_docs = documents[i:i+batch_size]
        batch_metas = metadatas[i:i+batch_size]
        
        batch_embeddings = []
        valid_ids = []
        valid_docs = []
        valid_metas = []
        
        for doc_id, doc, meta in zip(batch_ids, batch_docs, batch_metas):
            try:
                element = meta.get('element')
                
                # Skip unsupported elements (PM10, PM25, NOX)
                if element not in embedding_functions:
                    skip_count += 1
                    continue
                
                # Re-embed
                embedding_fn = embedding_functions[element]
                embedding = embedding_fn([doc])[0]
                
                # Convert numpy array to list
                if hasattr(embedding, 'tolist'):
                    embedding = embedding.tolist()
                elif isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()
                
                batch_embeddings.append(embedding)
                valid_ids.append(doc_id)
                valid_docs.append(doc)
                valid_metas.append(meta)
                
            except Exception as e:
                logger.error(f"Failed to re-embed {doc_id}: {e}")
                error_count += 1
        
        # Insert batch
        if batch_embeddings:
            try:
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
    logger.info(f"Skipped (PM10/PM25/NOX): {skip_count}")
    logger.info(f"Errors: {error_count}")
    logger.info(f"Total documents: {len(ids)}")
    logger.info(f"Target collection count: {target_collection.count()}")
    logger.info("="*60)
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Re-embed ChromaDB collection with T-Rep")
    parser.add_argument("--db-path", default="/home/0_code/RAG/ChromaDB/ts2vec_embedding_function/NIER_DB", help="ChromaDB database path")
    parser.add_argument("--source", default="time_series_collection", help="Source collection name")
    parser.add_argument("--target", default="time_series_collection_trep", help="Target collection name")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size")
    
    args = parser.parse_args()
    
    logger.info("Starting re-embedding process (direct file access)...")
    logger.info(f"Database path: {args.db_path}")
    logger.info(f"Source: {args.source}")
    logger.info(f"Target: {args.target}")
    logger.info(f"Device: {args.device}")
    
    try:
        success = reembed_collection(
            db_path=args.db_path,
            source_collection_name=args.source,
            target_collection_name=args.target,
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
