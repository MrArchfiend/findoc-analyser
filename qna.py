from data_preprocessing import preprocess_data
from chunking import ParagraphAwareChunker
from storing_retrieval import VectorStore
from question_generation import generate_subquestions
from response_generation import generate_response
from metadata_extraction import extract_document_metadata

store = VectorStore()
chunker = ParagraphAwareChunker()
_metadata_cache: dict = {}


def index_document(uploaded_file) -> str:
    doc_name = uploaded_file.name

    if doc_name in store.list_documents():
        return f"'{doc_name}' is already indexed."

    text = preprocess_data(uploaded_file)
    chunks = chunker.chunk(text)
    store.add_chunks(chunks, doc_name)

    metadata = extract_document_metadata(text, chunks=chunks)
    metadata["doc_name"] = doc_name
    metadata["chunk_count"] = len(chunks)
    _metadata_cache[doc_name] = metadata
    store.save_metadata(doc_name, metadata)

    return f"'{doc_name}' indexed — {len(chunks)} chunks, company: {metadata.get('company_name') or 'unknown'}."


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
    scored_chunks is a list of dicts with text, score, section_hint, chunk_index.
    """
    subquestions = generate_subquestions(query)
    retrieved = store.batch_query(subquestions, doc_name=doc_name, top_k=10)

    if not retrieved:
        return "No relevant information found in the document.", []

    return generate_response(query, retrieved)


def compare_documents(query: str, doc_names: list[str]) -> dict[str, tuple[str, list]]:
    """Run ask() for each doc and return {doc_name: (answer, scored_chunks)}."""
    return {doc: ask(query, doc_name=doc) for doc in doc_names}