import logging

from data_preprocessing import preprocess_data
from chunking import ParagraphAwareChunker
from storing_retrieval import VectorStore
from question_generation import generate_subquestions
from response_generation import generate_response
from metadata_extraction import extract_document_metadata

logger = logging.getLogger("findoc.qna")

store = VectorStore()
chunker = ParagraphAwareChunker()
_metadata_cache: dict = {}


def index_document(uploaded_file) -> str:
    doc_name = uploaded_file.name

    if doc_name in store.list_documents():
        return f"'{doc_name}' is already indexed."

    text = preprocess_data(uploaded_file)
    chunks = chunker.chunk(text)

    if not chunks:
        raise ValueError(f"No text could be extracted from '{doc_name}'.")

    store.add_chunks(chunks, doc_name)

    try:
        metadata = extract_document_metadata(text, chunks=chunks)
    except Exception as e:
        logger.warning("Metadata extraction failed for '%s': %s — using fallback.", doc_name, e)
        metadata = {
            "company_name": None, "ticker": None, "doc_type": None,
            "fiscal_year": None, "period_end_date": None,
            "summary": "Metadata extraction failed.", "financials": {},
        }

    metadata["doc_name"] = doc_name
    metadata["chunk_count"] = len(chunks)
    _metadata_cache[doc_name] = metadata

    try:
        store.save_metadata(doc_name, metadata)
    except Exception as e:
        logger.warning("Could not persist metadata for '%s': %s", doc_name, e)

    company = metadata.get("company_name") or "unknown"
    return f"'{doc_name}' indexed — {len(chunks)} chunks, company: {company}."


def get_metadata(doc_name: str) -> dict:
    if doc_name in _metadata_cache:
        return _metadata_cache[doc_name]
    meta = store.load_metadata(doc_name)
    if meta:
        _metadata_cache[doc_name] = meta
    return meta or {}


def get_chunks(doc_name: str) -> list:
    return store.get_all_chunks(doc_name)


def ask(query: str, doc_name: str) -> tuple[str, list]:
    """
    Returns (answer, scored_chunks_sent_to_llm).
    doc_name is passed through to LLM call tracking.
    """
    subquestions = generate_subquestions(query, doc_name=doc_name)
    retrieved    = store.batch_query(subquestions, doc_name=doc_name, top_k=10)

    if not retrieved:
        return "No relevant information found in the document.", []

    return generate_response(query, retrieved, doc_name=doc_name)