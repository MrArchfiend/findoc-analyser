from __future__ import annotations

import hashlib
import logging
from typing import Optional

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import (
    OllamaEmbeddingFunction,
    SentenceTransformerEmbeddingFunction,
)

from chunking import Chunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — override via constants.py in the repo root if needed.
# ---------------------------------------------------------------------------
try:
    from constants import (
        CHROMA_PERSIST_DIR,
        CHROMA_COLLECTION_NAME,
        EMBEDDING_BACKEND,       # "ollama" | "sentence-transformers"
        OLLAMA_EMBED_MODEL,      # e.g. "nomic-embed-text"
        SENTENCE_TRANSFORMER_MODEL,  # e.g. "all-MiniLM-L6-v2"
        OLLAMA_BASE_URL,         # e.g. "http://localhost:11434"
        TOP_K_RESULTS,           # default number of chunks to retrieve
    )
except ImportError:
    CHROMA_PERSIST_DIR = "./chromadb"
    CHROMA_COLLECTION_NAME = "findoc"
    EMBEDDING_BACKEND = "sentence-transformers"
    OLLAMA_EMBED_MODEL = "nomic-embed-text"
    SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
    OLLAMA_BASE_URL = "http://localhost:11434"
    TOP_K_RESULTS = 5


# ---------------------------------------------------------------------------
# Helper — build a stable document ID from chunk content so re-indexing the
# same file is idempotent (ChromaDB upsert deduplicates on ID).
# ---------------------------------------------------------------------------
def _chunk_id(chunk: Chunk, doc_name: str) -> str:
    fingerprint = f"{doc_name}::{chunk.chunk_index}::{chunk.char_start}"
    return hashlib.md5(fingerprint.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Embedding function factory — centralises backend selection.
# ---------------------------------------------------------------------------
def _make_embedding_fn(backend: str = EMBEDDING_BACKEND):
    if backend == "ollama":
        return OllamaEmbeddingFunction(
            url=f"{OLLAMA_BASE_URL}/api/embeddings",
            model_name=OLLAMA_EMBED_MODEL,
        )
    # Default: lightweight local model — no Ollama required.
    return SentenceTransformerEmbeddingFunction(
        model_name=SENTENCE_TRANSFORMER_MODEL
    )


# ---------------------------------------------------------------------------
# Core storage & retrieval class
# ---------------------------------------------------------------------------
class VectorStore:
    

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: list[Chunk], doc_name: str) -> int:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def query:
        raise NotImplementedError

    def batch_query:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Document management
    # ------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Lightweight result dataclass returned by query methods
# ---------------------------------------------------------------------------
