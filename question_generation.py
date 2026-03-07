import json
import requests

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_LLM_MODEL = "llama3"


def generate_subquestions(query: str) -> list[str]:
    """
    Break a broad query into sub-questions to improve retrieval.
    Falls back to the original query if anything goes wrong.
    """
    prompt = f"""
You are a financial analyst assistant.

A user asked: "{query}"

Break this into 3 specific sub-questions to help retrieve relevant information from a 10-K financial document.

Return ONLY a JSON object like this: {{"questions": ["q1", "q2", "q3"]}}
"""

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.2},
            },
            timeout=60,
        )
        response.raise_for_status()
        raw = response.json().get("response", "")
        questions = json.loads(raw).get("questions", [])

        # always include original query for full coverage
        if query not in questions:
            questions.insert(0, query)

        return questions

    except Exception as e:
        print(f"Sub-question generation failed: {e}")
        return [query]
