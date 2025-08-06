#!/usr/bin/env python3
"""
EduHelp Retriever Module

This module handles document retrieval using FAISS vector search.
Responsibility: Load FAISS index, perform semantic search, and return relevant documents.
"""

import os
import faiss
import pickle
import numpy as np
import logging
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing import List, Dict, Any

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

# Configuration constants
INDEX_FILE = "vector_index.faiss"
DOCS_FILE = "doc_store.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_K = 2


def load_embedding_model(model_name: str) -> SentenceTransformer:
    """
    Load the SentenceTransformer embedding model.
    
    Args:
        model_name: Name of the SentenceTransformer model
        
    Returns:
        Loaded SentenceTransformer model
    """
    try:
        logger.info(f"[Retriever] Loading SentenceTransformer model: {model_name}")
        model = SentenceTransformer(model_name)
        logger.info("[Retriever] SentenceTransformer model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"[Retriever] Error loading SentenceTransformer model: {str(e)}")
        raise


def load_faiss_index(index_file: str) -> faiss.Index:
    """
    Load the FAISS index from file.
    
    Args:
        index_file: Path to the FAISS index file
        
    Returns:
        Loaded FAISS index
    """
    try:
        logger.info(f"[Retriever] Loading FAISS index from {index_file}")
        index = faiss.read_index(index_file)
        logger.info("[Retriever] FAISS index loaded successfully")
        return index
    except Exception as e:
        logger.error(f"[Retriever] Error loading FAISS index: {str(e)}")
        raise


def load_document_store(docs_file: str) -> List[Dict[str, str]]:
    """
    Load the document store from pickle file.
    
    Args:
        docs_file: Path to the document store file
        
    Returns:
        List of document dictionaries
    """
    try:
        logger.info(f"[Retriever] Loading document store from {docs_file}")
        with open(docs_file, "rb") as f:
            doc_store = pickle.load(f)
        logger.info(f"[Retriever] Document store loaded with {len(doc_store)} documents")
        return doc_store
    except Exception as e:
        logger.error(f"[Retriever] Error loading document store: {str(e)}")
        raise


def validate_query(query: str) -> bool:
    """
    Validate that the query is not empty or whitespace-only.
    
    Args:
        query: User query string
        
    Returns:
        True if query is valid, False otherwise
    """
    return query and query.strip()


def embed_query(model: SentenceTransformer, query: str) -> np.ndarray:
    """
    Generate embedding for a user query.
    
    Args:
        model: SentenceTransformer model
        query: User query string
        
    Returns:
        Query embedding as numpy array
    """
    try:
        query_vector = model.encode([query])
        logger.info(f"[Query Embedding] Generated embedding with shape: {query_vector.shape}")
        return query_vector
    except Exception as e:
        logger.error(f"[Retriever] Error embedding query: {str(e)}")
        raise


def search_faiss_index(index: faiss.Index, query_vector: np.ndarray, k: int) -> tuple:
    """
    Search the FAISS index for similar documents.
    
    Args:
        index: FAISS index
        query_vector: Query embedding
        k: Number of documents to retrieve
        
    Returns:
        Tuple of (distances, indices)
    """
    try:
        distances, indices = index.search(query_vector, k)
        logger.info(f"[FAISS Search] Found {len(indices[0])} documents with distances: {distances[0]}")
        return distances, indices
    except Exception as e:
        logger.error(f"[Retriever] Error searching FAISS index: {str(e)}")
        raise


def retrieve_top_k_docs(query: str, k: int = DEFAULT_K) -> List[Dict[str, str]]:
    """
    Retrieve top-k documents for a given query.
    
    Args:
        query: User query string
        k: Number of documents to retrieve
        
    Returns:
        List of document dictionaries with filename and content
    """
    try:
        logger.info(f"[User Query] {query}")
        
        # Validate input
        if not validate_query(query):
            logger.warning("[Retriever] Empty query received")
            return []
        
        # Embed the query
        query_vector = embed_query(model, query)
        
        # Search FAISS
        distances, indices = search_faiss_index(index, query_vector, k)
        
        # Collect top K documents
        results = []
        for i in indices[0]:
            if i < len(doc_store):  # Ensure index is valid
                doc = doc_store[i]
                results.append({
                    "filename": doc["filename"],
                    "content": doc["content"]
                })
        
        logger.info(f"[Retrieved Docs] {[doc['filename'] for doc in results]}")
        
        if not results:
            logger.warning("[Retriever] No documents found for query")
            
        return results
        
    except Exception as e:
        logger.error(f"[Retriever] Error in retrieve_top_k_docs: {str(e)}")
        logger.error(f"[Retriever] Exception type: {type(e).__name__}")
        return []


class CustomFAISSRetriever(BaseRetriever):
    """
    Custom FAISS retriever for LangChain integration.
    
    This class implements the BaseRetriever interface to work with LangChain chains.
    """
    
    def __init__(self, search_kwargs: Dict[str, Any] = None):
        """
        Initialize the CustomFAISSRetriever.
        
        Args:
            search_kwargs: Dictionary containing search parameters (e.g., 'k')
        """
        super().__init__()
        self._search_kwargs = search_kwargs or {"k": DEFAULT_K}
    
    @property
    def search_kwargs(self) -> Dict[str, Any]:
        """Return search kwargs for LangChain compatibility."""
        return self._search_kwargs
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Get documents relevant to the query.
        
        Args:
            query: User query string
            
        Returns:
            List of LangChain Document objects
        """
        try:
            logger.info(f"[Retriever] Processing query: {query}")
            k = self.search_kwargs.get("k", DEFAULT_K)
            logger.info(f"[Retriever] Using k={k} for document retrieval")
            
            docs = retrieve_top_k_docs(query, k=k)
            
            # Convert to LangChain Document format
            langchain_docs = []
            for doc in docs:
                langchain_docs.append(Document(
                    page_content=doc["content"],
                    metadata={"source": doc["filename"]}
                ))
            
            logger.info(f"[Retriever] Returning {len(langchain_docs)} LangChain documents")
            
            if not langchain_docs:
                logger.warning("[Retriever] No documents found, returning empty list")
                
            return langchain_docs
            
        except Exception as e:
            logger.error(f"[Retriever] Error in _get_relevant_documents: {str(e)}")
            logger.error(f"[Retriever] Exception type: {type(e).__name__}")
            return []
    
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        """
        Async version of get_relevant_documents.
        
        Args:
            query: User query string
            
        Returns:
            List of LangChain Document objects
        """
        return self._get_relevant_documents(query)


def create_retriever(search_kwargs: Dict[str, Any] = None) -> CustomFAISSRetriever:
    """
    Create and return a CustomFAISSRetriever instance.
    
    Args:
        search_kwargs: Search parameters for the retriever
        
    Returns:
        Configured CustomFAISSRetriever instance
    """
    retriever = CustomFAISSRetriever(search_kwargs=search_kwargs or {"k": DEFAULT_K})
    
    # Verify the retriever has the required attributes
    assert hasattr(retriever, 'search_kwargs'), "Retriever must have search_kwargs attribute"
    assert hasattr(retriever, 'get_relevant_documents'), "Retriever must have get_relevant_documents method"
    
    return retriever


def retrieve_documents() -> CustomFAISSRetriever:
    """
    Returns a proper LangChain BaseRetriever instance.
    
    Returns:
        CustomFAISSRetriever instance configured for the application
    """
    return create_retriever()


# Initialize global components
try:
    model = load_embedding_model(MODEL_NAME)
    index = load_faiss_index(INDEX_FILE)
    doc_store = load_document_store(DOCS_FILE)
except Exception as e:
    logger.error(f"[Retriever] Failed to initialize components: {str(e)}")
    raise


def main():
    """Test function for the retriever."""
    query = input("🔎 Enter a user query: ")
    top_docs = retrieve_top_k_docs(query)
    
    for i, doc in enumerate(top_docs, 1):
        print(f"\n📄 Match #{i}: {doc['filename']}")
        print(doc['content'])


if __name__ == "__main__":
    main() 
        