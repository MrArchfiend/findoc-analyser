import json
import time
import logging
from groq import Groq
from constants import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger("findoc.question_generation")


def _get_client() -> Groq:
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Export it as an environment variable: "
            "export GROQ_API_KEY=your_key_here"
        )
    return Groq(api_key=GROQ_API_KEY)


def generate_subquestions(query: str, doc_name: str | None = None) -> list[str]:
    """
    Break a broad query into sub-questions to improve retrieval.
    Falls back to the original query if anything goes wrong.
    """
    from monitoring import record_llm_call, record_error

    prompt = f"""
You are a financial analyst assistant.
A user asked: "{query}"
Break this into 3 specific sub-questions to help retrieve relevant information from a 10-K financial document.
Return ONLY a JSON object like this: {{"questions": ["q1", "q2", "q3"]}}
"""
    start = time.time()
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        latency_ms = (time.time() - start) * 1000
        usage = response.usage

        record_llm_call(
            call_type="subquestions",
            model=GROQ_MODEL,
            latency_ms=latency_ms,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            doc_name=doc_name,
            success=True,
        )

        raw = response.choices[0].message.content
        questions = json.loads(raw).get("questions", [])
        if query not in questions:
            questions.insert(0, query)
        return questions

    except Exception as e:
        latency_ms = (time.time() - start) * 1000
        record_llm_call(
            call_type="subquestions",
            model=GROQ_MODEL,
            latency_ms=latency_ms,
            prompt_tokens=0,
            completion_tokens=0,
            doc_name=doc_name,
            success=False,
        )
        record_error("question_generation", type(e).__name__, str(e), doc_name)
        logger.warning("Sub-question generation failed: %s", e)
        return [query]