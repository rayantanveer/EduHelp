#!/usr/bin/env python3
"""
EduHelp Memory Module

This module handles conversation memory management for the EduHelp assistant.
Responsibility: Setup and manage LangChain conversation memory.
"""

import logging
from langchain.memory import ConversationBufferMemory

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


def create_conversation_memory(
    memory_key: str = "chat_history",
    return_messages: bool = True,
    output_key: str = "answer",
    max_interactions: int = 3
) -> ConversationBufferMemory:
    """
    Create a conversation memory object for storing chat history.
    
    Args:
        memory_key: Key used to inject memory into prompt
        return_messages: Whether to return formatted messages
        output_key: Key for the output to store in memory
        max_interactions: Maximum number of interactions to store
        
    Returns:
        Configured ConversationBufferMemory instance
    """
    try:
        logger.info("[Memory] Initializing ConversationBufferMemory")
        memory = ConversationBufferMemory(
            memory_key=memory_key,
            return_messages=return_messages,
            output_key=output_key,
            k=max_interactions
        )
        logger.info("[Memory] ConversationBufferMemory initialized successfully")
        return memory
    except Exception as e:
        logger.error(f"[Memory] Error initializing ConversationBufferMemory: {str(e)}")
        raise


def truncate_memory_if_needed(memory_obj, max_messages: int = 6):
    """
    Truncate memory to keep only the last max_messages messages.
    This prevents memory from growing too large.
    
    Args:
        memory_obj: LangChain memory object
        max_messages: Maximum number of messages to keep
    """
    try:
        if hasattr(memory_obj, 'chat_memory') and memory_obj.chat_memory.messages:
            current_count = len(memory_obj.chat_memory.messages)
            if current_count > max_messages:
                # Keep only the last max_messages
                memory_obj.chat_memory.messages = memory_obj.chat_memory.messages[-max_messages:]
                logger.info(f"[Memory] Truncated memory from {current_count} to {len(memory_obj.chat_memory.messages)} messages")
    except Exception as e:
        logger.error(f"[Memory] Error truncating memory: {str(e)}")


def get_memory_stats(memory_obj) -> dict:
    """
    Get statistics about the current memory state.
    
    Args:
        memory_obj: LangChain memory object
        
    Returns:
        Dictionary with memory statistics
    """
    try:
        if hasattr(memory_obj, 'chat_memory') and memory_obj.chat_memory.messages:
            return {
                "total_messages": len(memory_obj.chat_memory.messages),
                "user_messages": len([msg for msg in memory_obj.chat_memory.messages if msg.type == "human"]),
                "ai_messages": len([msg for msg in memory_obj.chat_memory.messages if msg.type == "ai"])
            }
        return {"total_messages": 0, "user_messages": 0, "ai_messages": 0}
    except Exception as e:
        logger.error(f"[Memory] Error getting memory stats: {str(e)}")
        return {"error": str(e)}


# Create the main memory instance for the application
memory = create_conversation_memory() 