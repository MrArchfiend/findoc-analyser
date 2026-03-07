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
    """
    Thin wrapper around a ChromaDB collection that handles:
      - Persisting chunk embeddings to disk
      - Upserting chunks (idempotent — safe to re-index the same document)
      - Similarity search with optional metadata filters
      - Listing and deleting indexed documents
    """

    def __init__(
        self,
        persist_dir: str = CHROMA_PERSIST_DIR,
        collection_name: str = CHROMA_COLLECTION_NAME,
        embedding_backend: str = EMBEDDING_BACKEND,
    ) -> None:
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._embed_fn = _make_embedding_fn(embedding_backend)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},   # cosine similarity
        )
        logger.info(
            "VectorStore ready — collection=%r  backend=%r  persist_dir=%r",
            collection_name, embedding_backend, persist_dir,
        )
    

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: list[Chunk], doc_name: str) -> int:
        """
        Upsert a list of Chunk objects into the vector store.

        Args:
            chunks:   Output of ParagraphAwareChunker.chunk() (or any chunker).
            doc_name: Logical document name used to namespace chunk IDs
                      (typically the uploaded filename).

        Returns:
            Number of chunks upserted.
        """
        if not chunks:
            logger.warning("add_chunks called with empty list — nothing stored.")
            return 0

        ids, documents, metadatas = [], [], []

        for chunk in chunks:
            ids.append(_chunk_id(chunk, doc_name))
            documents.append(chunk.text)
            metadatas.append({
                "doc_name":     doc_name,
                "chunk_index":  chunk.chunk_index,
                "method":       chunk.method,
                "char_start":   chunk.char_start,
                "char_end":     chunk.char_end,
                "section_hint": chunk.section_hint or "",
                # Flatten any caller-supplied extras (must be str/int/float/bool).
                **{k: str(v) for k, v in chunk.extra.items()},
            })

        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info("Upserted %d chunks for document %r.", len(chunks), doc_name)
        return len(chunks)

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
