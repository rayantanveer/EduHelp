#!/usr/bin/env python3
"""
EduHelp Embeddings Module

This module handles document embedding and vector database setup.
Responsibility: Load documents, generate embeddings, and save to FAISS index.
"""

import os
import faiss
import pickle
import logging
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("eduhelp.log")
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration constants
DATA_DIR = "../data"
INDEX_FILE = "vector_index.faiss"
DOCS_FILE = "doc_store.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"


def load_help_documents(data_dir: str) -> tuple[list, list]:
    """
    Load help documents from the data directory.
    
    Args:
        data_dir: Path to directory containing .txt files
        
    Returns:
        Tuple of (documents, filenames)
    """
    documents = []
    filenames = []
    
    try:
        for filename in os.listdir(data_dir):
            if filename.endswith(".txt"):
                file_path = os.path.join(data_dir, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    documents.append(content)
                    filenames.append(filename)
        
        logger.info(f"Loaded {len(documents)} documents from {data_dir}")
        return documents, filenames
        
    except Exception as e:
        logger.error(f"Error loading documents from {data_dir}: {str(e)}")
        raise


def initialize_embedding_model(model_name: str) -> SentenceTransformer:
    """
    Initialize the SentenceTransformer embedding model.
    
    Args:
        model_name: Name of the SentenceTransformer model to use
        
    Returns:
        Initialized SentenceTransformer model
    """
    try:
        logger.info(f"Loading SentenceTransformer model: {model_name}")
        model = SentenceTransformer(model_name)
        logger.info("SentenceTransformer model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading SentenceTransformer model: {str(e)}")
        raise


def generate_embeddings(model: SentenceTransformer, documents: list) -> list:
    """
    Generate embeddings for a list of documents.
    
    Args:
        model: SentenceTransformer model
        documents: List of document texts
        
    Returns:
        List of embeddings
    """
    try:
        logger.info(f"Generating embeddings for {len(documents)} documents")
        embeddings = model.encode(documents)
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise


def create_faiss_index(embeddings: list, dimension: int) -> faiss.Index:
    """
    Create and populate a FAISS index with embeddings.
    
    Args:
        embeddings: List of embeddings to add to index
        dimension: Dimension of the embeddings
        
    Returns:
        Populated FAISS index
    """
    try:
        logger.info(f"Creating FAISS index with dimension {dimension}")
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        logger.info(f"FAISS index created with {index.ntotal} vectors")
        return index
    except Exception as e:
        logger.error(f"Error creating FAISS index: {str(e)}")
        raise


def save_artifacts(index: faiss.Index, doc_store: list, index_file: str, docs_file: str):
    """
    Save FAISS index and document store to files.
    
    Args:
        index: FAISS index to save
        doc_store: Document store to save
        index_file: Path for FAISS index file
        docs_file: Path for document store file
    """
    try:
        # Save FAISS index
        logger.info(f"Saving FAISS index to {index_file}")
        faiss.write_index(index, index_file)
        
        # Save document store
        logger.info(f"Saving document store to {docs_file}")
        with open(docs_file, "wb") as f:
            pickle.dump(doc_store, f)
            
        logger.info("All artifacts saved successfully")
        
    except Exception as e:
        logger.error(f"Error saving artifacts: {str(e)}")
        raise


def create_document_store(filenames: list, documents: list) -> list:
    """
    Create a document store from filenames and documents.
    
    Args:
        filenames: List of filenames
        documents: List of document contents
        
    Returns:
        List of document dictionaries
    """
    return [{"filename": fn, "content": doc} for fn, doc in zip(filenames, documents)]


def main():
    """Main function to run the embedding pipeline."""
    try:
        # Load documents
        documents, filenames = load_help_documents(DATA_DIR)
        
        # Initialize model
        model = initialize_embedding_model(MODEL_NAME)
        
        # Generate embeddings
        embeddings = generate_embeddings(model, documents)
        
        # Create document store
        doc_store = create_document_store(filenames, documents)
        
        # Create FAISS index
        dimension = embeddings[0].shape[0]
        index = create_faiss_index(embeddings, dimension)
        
        # Save artifacts
        save_artifacts(index, doc_store, INDEX_FILE, DOCS_FILE)
        
        print("✅ Documents embedded and saved to FAISS index.")
        
    except Exception as e:
        logger.error(f"Embedding pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main() 