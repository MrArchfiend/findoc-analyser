from groq import Groq
from constants import GROQ_API_KEY, GROQ_MODEL, TOP_K_TO_LLM

client = Groq(api_key=GROQ_API_KEY)


def generate_response(query: str, retrieved_chunks: list) -> tuple[str, list]:
    """
    Select top TOP_K_TO_LLM chunks and generate an answer.

    Args:
        query: original user question
        retrieved_chunks: list of RetrievedChunk (already sorted by score desc)

    Returns:
        (answer_text, scored_chunks_sent_to_llm)
        scored_chunks_sent_to_llm is a list of dicts with keys: text, score, section_hint
    """
    top_chunks = retrieved_chunks[:TOP_K_TO_LLM]
    context = "\n\n".join(rc.text for rc in top_chunks)

    scored_chunks = [
        {
            "text": rc.text,
            "score": round(rc.score, 4),
            "section_hint": rc.section_hint,
            "chunk_index": rc.chunk_index,
        }
        for rc in top_chunks
    ]

    prompt = f"""You are a financial analyst assistant helping users understand 10-K SEC filings.
Use the context below to answer the question. If the answer is not in the context, say "I couldn't find that information in the document."

Context:
{context}

Question: {query}
Answer:"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip(), scored_chunks

    except Exception as e:
        print(f"Response generation failed: {e}")
        return "Sorry, I was unable to generate a response. Please check your Groq API key.", scored_chunks