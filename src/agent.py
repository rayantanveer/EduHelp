#!/usr/bin/env python3
"""
EduHelp Agent Module

This module handles the LangChain agent and QA pipeline.
Responsibility: Initialize LLM, create conversational chain, and process queries.
"""

import os
import logging
from dotenv import load_dotenv
from retriever import retrieve_documents
from memory import memory, truncate_memory_if_needed
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain

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
DEFAULT_MODEL = "gpt-3.5-turbo"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_K = 2


def check_openai_api_key() -> bool:
    """
    Check if OpenAI API key is available in environment variables.
    
    Returns:
        True if API key is available, False otherwise
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("[Agent] OpenAI API key not found in environment variables")
        return False
    return True


def initialize_llm(model: str = DEFAULT_MODEL, temperature: float = DEFAULT_TEMPERATURE) -> ChatOpenAI:
    """
    Initialize the ChatOpenAI language model.
    
    Args:
        model: OpenAI model name
        temperature: Model temperature for response generation
        
    Returns:
        Initialized ChatOpenAI instance
    """
    try:
        logger.info(f"[Agent] Initializing ChatOpenAI LLM: {model}")
        llm = ChatOpenAI(temperature=temperature, model=model)
        logger.info("[Agent] ChatOpenAI LLM initialized successfully")
        return llm
    except Exception as e:
        logger.error(f"[Agent] Error initializing ChatOpenAI LLM: {str(e)}")
        raise


def create_qa_chain(llm: ChatOpenAI, retriever, memory_obj) -> ConversationalRetrievalChain:
    """
    Create a ConversationalRetrievalChain with the given components.
    
    Args:
        llm: ChatOpenAI language model
        retriever: Document retriever
        memory_obj: Conversation memory object
        
    Returns:
        Configured ConversationalRetrievalChain
    """
    try:
        logger.info("[Agent] Initializing ConversationalRetrievalChain")
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory_obj,
            return_source_documents=True,
            output_key="answer",
        )
        logger.info("[Agent] ConversationalRetrievalChain initialized successfully")
        return qa_chain
    except Exception as e:
        logger.error(f"[Agent] Error initializing ConversationalRetrievalChain: {str(e)}")
        raise


def validate_response(result: dict) -> bool:
    """
    Validate that the LLM response is not empty or unusable.
    
    Args:
        result: Result dictionary from the QA chain
        
    Returns:
        True if response is valid, False otherwise
    """
    return result and result.get("answer") and result["answer"].strip()


def process_query(qa_chain: ConversationalRetrievalChain, query: str) -> dict:
    """
    Process a user query through the QA chain.
    
    Args:
        qa_chain: ConversationalRetrievalChain instance
        query: User query string
        
    Returns:
        Result dictionary from the QA chain
    """
    try:
        logger.info(f"[Agent] Processing query: {query}")
        result = qa_chain({"question": query})
        
        if validate_response(result):
            logger.info(f"[Agent] LLM Response received: {result['answer'][:100]}...")
            logger.info(f"[Agent] Source documents: {len(result.get('source_documents', []))}")
            logger.info("[Agent] Query processed successfully")
        else:
            logger.warning("[Agent] Empty or unusable LLM response received")
            
        return result
        
    except Exception as e:
        logger.error(f"[Agent] Error processing query: {str(e)}")
        logger.error(f"[Agent] Exception type: {type(e).__name__}")
        raise


def get_fallback_response() -> str:
    """
    Get a fallback response when the LLM fails or returns empty response.
    
    Returns:
        Fallback response string
    """
    return "I'm sorry, but I couldn't generate a helpful response. Please try rephrasing your question."


def get_error_response() -> str:
    """
    Get an error response when there's a technical issue.
    
    Returns:
        Error response string
    """
    return "I'm sorry, but I'm having technical difficulties right now. Please try again later."


def handle_query_with_fallback(qa_chain: ConversationalRetrievalChain, query: str) -> tuple[str, list]:
    """
    Handle a query with fallback responses for errors.
    
    Args:
        qa_chain: ConversationalRetrievalChain instance
        query: User query string
        
    Returns:
        Tuple of (response_text, source_documents)
    """
    try:
        result = process_query(qa_chain, query)
        
        if validate_response(result):
            return result["answer"], result.get("source_documents", [])
        else:
            return get_fallback_response(), []
            
    except Exception as e:
        logger.error(f"[Agent] Query failed with exception: {str(e)}")
        return get_error_response(), []


def initialize_agent_pipeline() -> ConversationalRetrievalChain:
    """
    Initialize the complete agent pipeline.
    
    Returns:
        Configured ConversationalRetrievalChain
    """
    # Check API key
    if not check_openai_api_key():
        raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
    
    # Initialize LLM
    llm = initialize_llm()
    
    # Get retriever
    retriever = retrieve_documents()
    
    # Create QA chain
    qa_chain = create_qa_chain(llm, retriever, memory)
    
    return qa_chain


def main():
    """Interactive CLI for testing the agent."""
    print("🤖 EduHelp Memory-Enabled Assistant")
    print("Type 'exit' to quit\n")

    try:
        # Initialize the agent pipeline
        qa_chain = initialize_agent_pipeline()
        chat_history = []

        while True:
            query = input("🧑 You: ")
            if query.lower() in ["exit", "quit"]:
                break

            # Process query with fallback handling
            response, source_docs = handle_query_with_fallback(qa_chain, query)
            
            print(f"\n🎓 EduHelp says:\n{response}")
            print("\n" + "-"*60 + "\n")

            # Add to chat history
            chat_history.append((query, response))
            
            # Truncate memory if it gets too long
            truncate_memory_if_needed(memory)

    except Exception as e:
        logger.error(f"[Agent] CLI failed: {str(e)}")
        print(f"❌ Error: {str(e)}")


# Initialize the main QA chain for the application
try:
    qa_chain = initialize_agent_pipeline()
except Exception as e:
    logger.error(f"[Agent] Failed to initialize agent pipeline: {str(e)}")
    raise


if __name__ == "__main__":
    main() 