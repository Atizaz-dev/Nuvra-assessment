# RAG-based Q&A Assistant

A command-line question-answering assistant built with LangChain that uses Retrieval-Augmented Generation (RAG) to answer questions from internal documentation.

## Features

- **Automatic Document Loading**: Automatically loads all Markdown and plain-text files from the `./docs/` directory
- **Vector Embedding**: Uses OpenAI's `text-embedding-3-small` model for document embeddings
- **Vector Store**: Uses Pinecone for efficient similarity search
- **RAG Pipeline**: Complete pipeline including ingestion, indexing, retrieval, and generation
- **CLI Interface**: Simple command-line interface for asking questions

## Architecture

The project follows a modular structure:

- **`rag_pipeline.py`**: Core RAG pipeline module containing:
  - Document loading and splitting
  - Embedding and indexing with Pinecone
  - Retrieval and generation using LangChain
- **`qa.py`**: Command-line interface script
- **`docs/`**: Directory containing documentation files (Markdown or plain-text)

## Prerequisites

- Python 3.11
- OpenAI API key
- Pinecone API key and environment

## Setup

### 1. Clone or Navigate to the Project

```bash
cd assessment-rag
```

### 2. Create Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Pinecone API Key
PINECONE_API_KEY=your_pinecone_api_key_here

# Pinecone Environment (e.g., us-east-1-aws)
PINECONE_ENVIRONMENT=us-east-1

# Pinecone Index Name (read from this env var, defaults to rag-assessment if not set)
PINECONE_INDEX_NAME=rag-assessment
```

**Important**: Replace the placeholder values with your actual API keys.

### 5. Prepare Documents

Place your documentation files (`.txt` or `.md` format) in the `./docs/` directory. The system will automatically load all files from this directory.

## Usage

### Basic Usage

Ask a question using the CLI:

```bash
python qa.py --question "How do I reset my password?"
```

Or use the short form:

```bash
python qa.py -q "How do I reset my password?"
```

### First Run

On the first run, the system will:
1. Load all documents from `./docs/`
2. Split them into chunks
3. Create embeddings
4. Create and populate the Pinecone index
5. Answer your question

This may take a few minutes depending on the number and size of documents.

### Subsequent Runs

After the initial indexing, the system will reuse the existing Pinecone index for faster responses. The index persists between runs.

### Advanced Options

**Reindex documents** (deletes existing index and recreates):

```bash
python qa.py --question "Your question" --reindex
```

**Specify custom documents directory**:

```bash
python qa.py --question "Your question" --docs-dir ./custom-docs
```

**Override Pinecone index name** (defaults to `PINECONE_INDEX_NAME` env var or `rag-assessment`):

```bash
python qa.py --question "Your question" --index-name my-custom-index
```

**Adjust number of retrieved chunks**:

```bash
python qa.py --question "Your question" --top-k 5
```

### Example Output

```
Initializing RAG pipeline...
Loaded 12 documents
Split documents into 45 chunks
Adding 45 chunks to vector store...
Document ingestion complete!

Question: How do I reset my password?

Answer: To reset your password, open Settings and choose Security. Click "Forgot Password" and enter your registered email address. Check your inbox for the reset link and follow the instructions.

Sources: reset_password.txt
```

## Project Structure

```
assessment-rag/
├── docs/                    # Documentation files directory
│   ├── intro.txt
│   ├── reset_password.txt
│   └── ...
├── rag_pipeline.py          # Core RAG pipeline module
├── qa.py                    # CLI interface script
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (create this)
└── README.md                # This file
```

## How It Works

### 1. Ingestion
- Reads all `.txt` and `.md` files from `./docs/`
- Splits documents into chunks of ~1000 characters with 200 character overlap
- Preserves source metadata for each chunk

### 2. Embed & Index
- Generates embeddings using OpenAI's `text-embedding-3-small` model
- Stores embeddings in Pinecone vector database
- Creates a cosine similarity index for efficient retrieval

### 3. Retrieve
- Given a user question, generates an embedding for the question
- Searches Pinecone for top-k most similar document chunks
- Returns relevant context passages

### 4. Generate
- Constructs a prompt combining retrieved chunks and user question
- Uses GPT-3.5-turbo to generate a concise answer
- Returns answer with source document references

## Troubleshooting

### "Missing required environment variables"
- Ensure your `.env` file exists and contains `OPENAI_API_KEY` and `PINECONE_API_KEY`
- Check that the `.env` file is in the project root directory

### "Directory ./docs does not exist"
- Create the `docs` directory: `mkdir docs`
- Add your documentation files to this directory

### "No documents found in ./docs"
- Ensure you have `.txt` or `.md` files in the `docs` directory
- Check file permissions

### Pinecone Index Errors
- Verify your Pinecone API key and environment are correct
- Ensure you have sufficient Pinecone quota
- Try using `--reindex` to recreate the index

### Slow First Run
- First run includes document ingestion and indexing, which takes time
- Subsequent runs are much faster as they reuse the existing index

## Dependencies

- `langchain`: Core framework for RAG pipeline
- `langchain-openai`: OpenAI integration for embeddings and LLM
- `langchain-pinecone`: Pinecone vector store integration
- `pinecone-client`: Pinecone Python client
- `openai`: OpenAI API client
- `python-dotenv`: Environment variable management
- `tiktoken`: Token counting for text splitting