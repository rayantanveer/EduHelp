# EduHelp - AI-Powered Educational Assistant

## Overview

EduHelp is a fully functional AI-powered educational assistant that helps users navigate an online learning portal. The system uses Retrieval-Augmented Generation (RAG) to provide intelligent, context-aware responses to user queries.

### Key Features
- **Semantic Document Retrieval**: FAISS vector database with SentenceTransformers
- **Conversational AI**: OpenAI GPT-3.5-turbo with conversation memory
- **Modern Web Interface**: Streamlit-based UI with real-time processing
- **Robust Error Handling**: Graceful fallbacks and user-friendly error messages
- **Memory Management**: Automatic conversation history with smart truncation

## Project Structure

```
eduhelp/
├── data/                       # Help documents (.txt files)
│   ├── 1_help_upload_assignment.txt
│   ├── 2_help_video_upload.txt
│   ├── 3_help_troubleshooting_submit.txt
│   ├── 4_help_finding_grades.txt
│   └── 5_help_deadlines.txt
├── src/                        # Core application modules
│   ├── embeddings.py           # Document embedding pipeline
│   ├── memory.py              # Conversation memory management
│   ├── retriever.py           # Document retrieval system
│   ├── agent.py               # LLM agent pipeline
│   └── ui.py                  # Streamlit web interface
├── requirements.txt            # Python dependencies
├── README.md                  # Project documentation
└── .env                       # Environment variables (API keys)
```

## Technology Stack

- **Backend**: Python 3.8+
- **AI/ML**: OpenAI GPT-3.5-turbo, SentenceTransformers
- **Vector Database**: FAISS (Facebook AI Similarity Search)
- **Web Framework**: Streamlit
- **Memory Management**: LangChain ConversationBufferMemory
- **Error Handling**: Comprehensive logging and graceful fallbacks

## Quick Start

### 1. Prerequisites
- Python 3.8 or higher
- OpenAI API key

### 2. Installation
```bash
# Clone the repository
git clone <repository-url>
cd eduhelp

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Run the Application (Automatic Setup)
The application will automatically generate the FAISS index and document store if they don't exist.

```bash
cd src
streamlit run ui.py
```

**Note**: If you want to manually generate embeddings first, you can run:
```bash
cd src
python embeddings.py
```
```bash
cd src
streamlit run ui.py
```

The application will be available at `http://localhost:8501`

## Usage

1. **Ask Questions**: Type your educational queries in the input field
2. **Get Intelligent Responses**: The AI provides context-aware answers
3. **View Sources**: Expand the "Retrieved Documents" section to see source materials
4. **Track History**: View conversation history in the expandable section

### Example Queries
- "How do I upload my assignment?"
- "What about video uploads?"
- "Where can I find my grades?"
- "When are the deadlines?"

## Architecture

### Core Components

#### **Embeddings Pipeline** (`embeddings.py`)
- Loads help documents from `data/` directory
- Generates embeddings using SentenceTransformers
- Creates and saves FAISS vector index
- Stores document metadata for retrieval

#### **Memory Management** (`memory.py`)
- Manages conversation history using LangChain
- Implements automatic memory truncation
- Provides memory statistics and utilities

#### **Document Retrieval** (`retriever.py`)
- Performs semantic search using FAISS
- Validates and processes user queries
- Returns relevant documents with metadata
- Implements LangChain-compatible retriever interface

#### **AI Agent** (`agent.py`)
- Initializes OpenAI language model
- Creates ConversationalRetrievalChain
- Handles query processing with fallbacks
- Manages error responses and validation

#### **Web Interface** (`ui.py`)
- Streamlit-based user interface
- Real-time query processing with spinners
- Displays AI responses and source documents
- Manages conversation history display

## Features

###  **Completed Features**
- **Semantic Search**: FAISS vector database with SentenceTransformers
- **Conversational AI**: OpenAI GPT-3.5-turbo integration
- **Memory System**: LangChain conversation memory with truncation
- **Modern UI**: Responsive Streamlit interface
- **Error Handling**: Comprehensive error management and fallbacks
- **Logging**: Detailed application logging
- **Documentation**: Complete code documentation
- **Modular Design**: Clean, maintainable codebase

###  **Key Capabilities**
- **Context-Aware Responses**: Uses retrieved documents for accurate answers
- **Conversation Memory**: Remembers previous interactions
- **Robust Error Handling**: Graceful degradation on failures
- **User-Friendly Interface**: Intuitive web-based UI
- **Source Transparency**: Shows retrieved documents to users

## Development

### Code Quality
- **Modular Architecture**: Single responsibility principle
- **Comprehensive Documentation**: Docstrings and inline comments
- **Error Handling**: Try-catch blocks with meaningful messages
- **Logging**: Structured logging throughout the application
- **Type Hints**: Python type annotations for better code clarity

### Testing
The application has been thoroughly tested and is production-ready with:
- Robust error handling
- Input validation
- Memory management
- UI responsiveness
- API integration stability

## Deployment

### Local Development
```bash
streamlit run src/ui.py
```

### Production Considerations
- Set up proper environment variables
- Configure logging levels
- Monitor API usage and costs
- Implement rate limiting if needed
- Set up proper error monitoring

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions, please open an issue in the repository or contact the development team. 