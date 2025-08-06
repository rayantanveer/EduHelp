import streamlit as st
import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from retriever import retrieve_documents
from memory import memory, truncate_memory_if_needed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),  # Print to console
        logging.FileHandler("eduhelp.log")  # Write to log file
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Set page config
st.set_page_config(page_title="EduHelp - AI Assistant", page_icon="🎓", layout="wide")

# Main container for better layout control
with st.container():
    st.markdown("# 🎓 EduHelp - AI Assistant")
    st.markdown("Your intelligent educational assistant powered by RAG & memory.")
    
    st.markdown("---")
    
    # User input section
    st.markdown("### 💬 Ask EduHelp")
    query = st.text_input("Enter your question:", placeholder="e.g. How do I upload my assignment?", key="user_query")
    
    # Button with better spacing
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        submit = st.button("🚀 Get Help", use_container_width=True)
    
    st.write("")  # Add spacing

# Check if OpenAI API key is available
if not os.getenv("OPENAI_API_KEY"):
    st.error("❌ OpenAI API key not found! Please set the OPENAI_API_KEY environment variable.")
    st.info("""
    To fix this:
    1. Create a `.env` file in the eduhelp directory
    2. Add your OpenAI API key: `OPENAI_API_KEY=your_actual_api_key_here`
    3. Restart the Streamlit app
    """)
    st.stop()

# Setup LLM + Retrieval + Memory Chain
try:
    logger.info("[UI] Initializing LLM and ConversationalRetrievalChain")
    llm = ChatOpenAI(temperature=0.2, model="gpt-3.5-turbo")
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retrieve_documents(),
        memory=memory,
        return_source_documents=True,
        output_key="answer",
    )
    logger.info("[UI] LLM and ConversationalRetrievalChain initialized successfully")
except Exception as e:
    logger.error(f"[UI] Error initializing OpenAI client: {str(e)}")
    st.error(f"❌ Error initializing OpenAI client: {str(e)}")
    st.info("Please check your OpenAI API key and ensure it's valid.")
    st.stop()

# Setup LLM + Retrieval + Memory Chain
try:
    logger.info("[UI] Initializing LLM and ConversationalRetrievalChain")
    llm = ChatOpenAI(temperature=0.2, model="gpt-3.5-turbo")
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retrieve_documents(),
        memory=memory,
        return_source_documents=True,
        output_key="answer",
    )
    logger.info("[UI] LLM and ConversationalRetrievalChain initialized successfully")
except Exception as e:
    logger.error(f"[UI] Error initializing OpenAI client: {str(e)}")
    st.error(f"❌ Error initializing OpenAI client: {str(e)}")
    st.info("Please check your OpenAI API key and ensure it's valid.")
    st.stop()

# Handle query
if submit:
    if not query or not query.strip():
        st.warning("⚠️ Please enter a question to get help!")
        logger.warning("[UI] Empty query submitted")
    else:
        logger.info(f"[UI] Query submitted: {query}")
        
        # Processing section with spinner
        with st.spinner("🤔 Thinking..."):
            try:
                # Use the LangChain memory properly
                result = qa_chain({"question": query})
                
                # Check if we got a valid response
                if not result or not result.get("answer") or not result["answer"].strip():
                    st.error("❌ That response didn't help. Try rephrasing your question.")
                    logger.warning("[UI] Empty or unusable LLM response received")
                else:
                    logger.info(f"[UI] Final answer received: {result['answer'][:100]}...")
                    logger.info(f"[UI] Source documents count: {len(result.get('source_documents', []))}")

                    st.markdown("---")
                    st.markdown("### 🎓 EduHelp's Answer")
                    st.success(result["answer"])
                    
                    # Show retrieved sources with better formatting
                    st.markdown("---")
                    st.markdown("### 📄 Retrieved Documents")
                    with st.expander("View source documents", expanded=False):
                        if result.get("source_documents"):
                            for i, doc in enumerate(result["source_documents"], 1):
                                # Document header
                                st.markdown(f"**📄 Document {i}: {doc.metadata['source']}**")
                                
                                # Show first few lines of content
                                content = doc.page_content
                                if len(content) > 200:
                                    content = content[:200] + "..."
                                
                                st.markdown(f"*{content}*")
                                st.markdown("---")
                        else:
                            st.info("No helpful documents found for this query.")
                            logger.warning("[UI] No source documents retrieved")
                    
                    # Truncate memory if it gets too long
                    truncate_memory_if_needed(memory)
                            
            except Exception as e:
                error_msg = "❌ Sorry, I'm having trouble processing your request right now. Please try again later."
                st.error(error_msg)
                logger.error(f"[UI] Error processing query: {str(e)}")
                logger.error(f"[UI] Exception type: {type(e).__name__}")

    # Conversation history section (always visible)
    st.markdown("---")
    st.markdown("### 🧠 Conversation History")
    with st.expander("View previous conversations", expanded=True):
        # Get conversation history from LangChain memory
        if hasattr(memory, 'chat_memory') and memory.chat_memory.messages:
            for i in range(0, len(memory.chat_memory.messages), 2):
                if i + 1 < len(memory.chat_memory.messages):
                    user_msg = memory.chat_memory.messages[i].content
                    bot_msg = memory.chat_memory.messages[i + 1].content
                    
                    # User message
                    st.markdown(f"**🧑 You:** {user_msg}")
                    
                    # AI response
                    st.markdown(f"**🎓 EduHelp:** {bot_msg}")
                    
                    st.markdown("---")
        else:
            st.info("No conversation history yet. Start chatting to see the history here!")

# Footer
st.markdown("---")
st.markdown("*Powered by OpenAI & LangChain*") 