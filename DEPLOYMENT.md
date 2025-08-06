# EduHelp - Streamlit Cloud Deployment Guide

## 🚀 Deploying to Streamlit Cloud

### Prerequisites
- GitHub account
- Streamlit Cloud account (free tier available)
- OpenAI API key

### Step 1: Prepare Your Repository

1. **Push to GitHub**: Upload your EduHelp project to a GitHub repository
2. **Environment Variables**: Set up your OpenAI API key in Streamlit Cloud

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Configure your deployment:

#### **Repository Settings**
- **Repository**: `your-username/your-repo-name`
- **Branch**: `main` (or your default branch)
- **Main file path**: `src/ui.py`

#### **Advanced Settings**
- **Python version**: 3.9 or higher
- **Requirements file**: `requirements.txt`

### Step 3: Environment Variables

In Streamlit Cloud, add these environment variables:

```
OPENAI_API_KEY = your_openai_api_key_here
```

### Step 4: Deploy

Click "Deploy" and wait for the build to complete. The app will:
- Install dependencies from `requirements.txt`
- Generate FAISS index automatically on first run
- Be available at your Streamlit Cloud URL

## 🔧 Important Notes

### **Automatic Index Generation**
- The app automatically generates `vector_index.faiss` and `doc_store.pkl` on first run
- No manual setup required
- Files are generated in the app's temporary storage

### **File Structure for Deployment**
```
eduhelp/
├── data/                    # ✅ Include in git (for local development)
│   └── *.txt              # Help documents (original location)
├── src/                    # ✅ Include in git
│   ├── agent.py           # Core logic
│   ├── embeddings.py      # Embedding utilities
│   ├── help_docs/         # Help documents (for deployment)
│   │   └── *.txt         # Help documents
│   ├── memory.py          # Memory management
│   ├── retriever.py       # Document retrieval
│   └── ui.py              # Streamlit interface
├── requirements.txt        # ✅ Include in git
├── README.md              # ✅ Include in git
├── .gitignore             # ✅ Include in git
└── .env                   # ❌ DO NOT include (contains API keys)
```

### **Files Excluded by .gitignore**
- `src/vector_index.faiss` - Generated automatically
- `src/doc_store.pkl` - Generated automatically
- `src/eduhelp.log` - Local logs
- `.env` - API keys and secrets
- `__pycache__/` - Python cache
- `*.backup` - Backup files

## 🎯 Benefits of This Setup

### **Zero-Friction Deployment**
- No manual index generation required
- Automatic file creation on first run
- Clean repository without generated files

### **Security**
- API keys kept secure in environment variables
- No sensitive data in git repository
- Proper .gitignore configuration

### **Scalability**
- Works on Streamlit Cloud's ephemeral storage
- Regenerates index if needed
- No persistent file dependencies

## 🐛 Troubleshooting

### **Common Issues**

1. **Build Fails**: Check `requirements.txt` for correct dependencies
2. **API Key Error**: Verify environment variable is set correctly
3. **Import Errors**: Ensure all Python files are in the `src/` directory
4. **Memory Issues**: The app uses efficient embeddings and should work within limits

### **Debugging**
- Check Streamlit Cloud logs for error messages
- Verify all files are properly committed to git
- Ensure environment variables are set correctly

## 📊 Performance Considerations

- **First Run**: May take 1-2 minutes to generate embeddings
- **Subsequent Runs**: Fast startup with cached components
- **Memory Usage**: Optimized for Streamlit Cloud's free tier
- **API Calls**: Efficient caching and error handling

## 🎉 Success!

Once deployed, your EduHelp app will be available at:
`https://your-app-name-your-username.streamlit.app`

The app will automatically handle all setup and provide a seamless user experience! 