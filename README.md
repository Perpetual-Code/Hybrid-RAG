markdown# Agentic RAG & Hybrid Retrieval Pipeline

This repository contains the source code for building an advanced Agentic RAG and Hybrid Retrieval system. To keep the repository lightweight and comply with GitHub file limits, large pre-computed search indexes and **900+ MB embedding files are excluded from version control** and must be generated locally.

---

## 🛠️ Setup Instructions

### 1. Prerequisites & Environment
This project uses `uv` for fast, reproducible dependency management. Make sure you have it installed, then synchronize the environment:

```bash
# Install dependencies from uv.lock
uv sync
```

### 2. Configure Environment Variables
The embedding pipeline requires access to external language models and datasets. Create a `.env` file in the root directory and add your credentials:

```bash
# Create your local environment file
touch .env
```

Open `.env` and add the required API keys (e.g., Hugging Face or OpenAI tokens):
```text
HUGGINGFACE_API_TOKEN=your_token_here
OPENAI_API_KEY=your_key_here
```

### 3. Generate the Embedding Indexes
Because the **`embeddings.npy`** file (~900MB) is ignored by Git, you must generate the search vector store locally before running exploration or retrieval scripts. 

Run the embedding script, which will automatically stream the dataset from Hugging Face, process the text, and build the local index directory:

```bash
uv run 3-embed.py
```

*This will populate the `02-hybrid-retrieval/indexes/` folder with `bm25` and `dense` subdirectories needed for the application.*

---

## 📂 Project Structure

*   **`01-agentic-rag/`**: Source files for the agentic orchestration layer.
*   **`02-hybrid-retrieval/`**: Core scripts combining dense (embeddings) and sparse (BM25) search techniques.
    *   `indexes/` *(Local only)*: Stores generated vector matrices (`embeddings.npy`) and corpus metadata.
    *   `3-embed.py`: Script to pull data from HF and generate vectors.
    *   `2-bm25.py`: Script to build/run the sparse search index.
    *   `4-rrf.py`: Reciprocal Rank Fusion script to merge search results.
*   **`main.py`**: Main application entry point.