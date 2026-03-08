import requests

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_LLM_MODEL = "llama3"


def generate_response(query: str, context: str) -> str:
    """
    Generate an answer to the query using the retrieved context chunks.
    Falls back to an error message if the LLM call fails.
    """
    prompt = f"""
You are a financial analyst assistant helping users understand 10-K SEC filings.

Use the context below to answer the question. If the answer is not in the context, say "I couldn't find that information in the document."

Context:
{context}

Question: {query}

Answer:
"""

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3},
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()

    except Exception as e:
        print(f"Response generation failed: {e}")
        return "Sorry, I was unable to generate a response. Please make sure Ollama is running."
