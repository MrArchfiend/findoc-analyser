import os

# Groq config
GROQ_API_KEY = "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"
GROQ_MODEL = "llama-3.3-70b-versatile"

# ChromaDB
CHROMADB_DIR = 'chromadb'
CHROMA_PERSIST_DIR = 'chromadb'
CHROMA_COLLECTION_NAME = 'findoc'

# Embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384
EMBEDDING_BACKEND = "sentence-transformers"
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"

# Retrieval
TOP_K_RESULTS = 5
TOP_K_TO_LLM = 5