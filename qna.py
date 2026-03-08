from data_preprocessing import preprocess_data
from chunking import ParagraphAwareChunker
from storing_retrieval import VectorStore
from question_generation import generate_subquestions
from response_generation import generate_response


store = VectorStore()
chunker = ParagraphAwareChunker()


def index_document(uploaded_file) -> str:
    """Preprocess, chunk and index a uploaded PDF into ChromaDB."""
    doc_name = uploaded_file.name

    # skip if already indexed
    if doc_name in store.list_documents():
        return f"'{doc_name}' is already indexed."

    text = preprocess_data(uploaded_file)
    chunks = chunker.chunk(text)
    store.add_chunks(chunks, doc_name)

    return f"'{doc_name}' indexed successfully — {len(chunks)} chunks stored."


def ask(query: str, doc_name: str) -> str:
    """Run the full RAG pipeline for a query against an indexed document."""

    # step 1 — break query into sub-questions
    subquestions = generate_subquestions(query)

    # step 2 — retrieve relevant chunks for all sub-questions
    chunks = store.batch_query(subquestions, doc_name=doc_name)

    if not chunks:
        return "No relevant information found in the document."

    # step 3 — build context string from top chunks
    context = "\n\n".join(rc.text for rc in chunks)

    # step 4 — generate final answer
    return generate_response(query, context)
