# Corrective RAG (CRAG) Implementation

This repository contains an implementation of **Corrective RAG (CRAG)**, a system that combines document retrieval with web search capabilities to provide more accurate responses to queries.

## Features

- **PDF document processing and chunking**
- **Document retrieval** using FAISS and HuggingFace embeddings
- **Automatic relevance grading** of retrieved documents
- **Web search integration** when local documents are insufficient
- **Query rewriting** for better search results
- **Streamlit-based user interface**

## Prerequisites

- Python 3.8+
- Azure OpenAI API access
- Tavily API key for web search

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/corrective-rag.git
   cd corrective-rag
   ```

2. **Create and activate a virtual environment** (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Create a `.env` file** in the root directory with your API keys:

   ```env
   AZURE_OPENAI_KEY=your_azure_openai_key
   AZURE_OPENAI_ENDPOINT=your_azure_endpoint
   TAVILY_API_KEY=your_tavily_api_key
   ```

## Usage

Run the Streamlit application:

```bash
streamlit run streamlit_main.py
```

The web interface will allow you to:

- Upload PDF documents
- Ask questions about the documents
- Get responses that combine information from your documents and web search when needed