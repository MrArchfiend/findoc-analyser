import time
import logging
from groq import Groq
from constants import GROQ_API_KEY, GROQ_MODEL, TOP_K_TO_LLM

logger = logging.getLogger("findoc.response_generation")


def _get_client() -> Groq:
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Export it as an environment variable: "
            "export GROQ_API_KEY=your_key_here"
        )
    return Groq(api_key=GROQ_API_KEY)


def generate_response(
    query: str,
    retrieved_chunks: list,
    doc_name: str | None = None,
) -> tuple[str, list]:
    """
    Select top TOP_K_TO_LLM chunks and generate an answer.
    Returns (answer_text, scored_chunks_sent_to_llm).
    """
    from monitoring import record_llm_call, record_error

    top_chunks = retrieved_chunks[:TOP_K_TO_LLM]
    # Truncate each chunk to avoid Groq context length limits
    context    = "\n\n".join(rc.text[:800] for rc in top_chunks)

    scored_chunks = [
        {
            "text":         rc.text,
            "score":        round(rc.score, 4),
            "section_hint": rc.section_hint,
            "chunk_index":  rc.chunk_index,
        }
        for rc in top_chunks
    ]

    prompt = f"""You are a financial analyst assistant helping users understand 10-K SEC filings.
Use the context below to answer the question. If the answer is not in the context, say "I couldn't find that information in the document."

Context:
{context}

Question: {query}
Answer:"""

    start = time.time()
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        latency_ms = (time.time() - start) * 1000
        usage = response.usage

        record_llm_call(
            call_type="response",
            model=GROQ_MODEL,
            latency_ms=latency_ms,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            doc_name=doc_name,
            success=True,
        )

        return response.choices[0].message.content.strip(), scored_chunks

    except Exception as e:
        latency_ms = (time.time() - start) * 1000
        record_llm_call(
            call_type="response",
            model=GROQ_MODEL,
            latency_ms=latency_ms,
            prompt_tokens=0,
            completion_tokens=0,
            doc_name=doc_name,
            success=False,
        )
        record_error("response_generation", type(e).__name__, str(e), doc_name)
        logger.warning("Response generation failed: %s", e)
        return f"Sorry, I was unable to generate a response. Error: {type(e).__name__}: {str(e)[:200]}", scored_chunks


def generate_response_with_feedback(
    query: str,
    retrieved_chunks: list,
    previous_answer: str,
    feedback_comment: str = "",
    doc_name: str | None = None,
) -> tuple[str, list]:
    """Regenerate an answer using the previous answer and user feedback as context."""
    from monitoring import record_llm_call, record_error

    top_chunks = retrieved_chunks[:TOP_K_TO_LLM]
    # Truncate each chunk to avoid Groq context length limits
    context    = "\n\n".join(rc.text[:800] for rc in top_chunks)

    scored_chunks = [
        {
            "text":         rc.text,
            "score":        round(rc.score, 4),
            "section_hint": rc.section_hint,
            "chunk_index":  rc.chunk_index,
        }
        for rc in top_chunks
    ]

    feedback_note = (
        f"\nThe user also left this comment: \"{feedback_comment}\""
        if feedback_comment else ""
    )

    prompt = f"""You are a financial analyst assistant helping users understand 10-K SEC filings.
A user asked a question and rated your previous answer as unhelpful.{feedback_note}

Your task is to provide a better, more accurate and complete answer.

Previous answer (which was not helpful):
{previous_answer}

Document context:
{context}

Question: {query}

Improved answer (be more specific, structured, and thorough than the previous answer):"""

    start = time.time()
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        latency_ms = (time.time() - start) * 1000
        usage = response.usage

        record_llm_call(
            call_type="feedback_response",
            model=GROQ_MODEL,
            latency_ms=latency_ms,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            doc_name=doc_name,
            success=True,
        )

        return response.choices[0].message.content.strip(), scored_chunks

    except Exception as e:
        latency_ms = (time.time() - start) * 1000
        record_llm_call(
            call_type="feedback_response",
            model=GROQ_MODEL,
            latency_ms=latency_ms,
            prompt_tokens=0,
            completion_tokens=0,
            doc_name=doc_name,
            success=False,
        )
        record_error("response_generation", type(e).__name__, str(e), doc_name)
        raise