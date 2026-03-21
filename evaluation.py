"""
evaluation.py — RAGAS evaluation using Groq (LLaMA 3.3 70B) as judge LLM.

ragas 0.4.3 fixes:
  1. Column names: user_input, response, retrieved_contexts
  2. Set metric.llm explicitly — otherwise ragas falls back to OpenAI
  3. answer_relevancy needs embeddings — use langchain_huggingface.HuggingFaceEmbeddings
     Falls back to faithfulness-only if sentence-transformers not installed
"""

import logging
import threading
import time
import warnings

from constants import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger("findoc.evaluation")


def _build_embeddings():
    """
    Try to build local HuggingFace embeddings.
    Returns None if sentence-transformers is not installed.
    """
    # Try new langchain_huggingface first, fall back to langchain_community
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except ImportError:
            return None

    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper
        return LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        )
    except Exception as e:
        logger.warning("Could not build embeddings: %s — answer_relevancy will be skipped", e)
        return None


def _run_evaluation(
    answer_id: str,
    doc_name: str,
    query: str,
    answer: str,
    contexts: list[str],
) -> None:
    from monitoring import record_ragas_eval, record_error

    start = time.time()
    try:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set.")

        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        try:
            from ragas.metrics import faithfulness, answer_relevancy
            from ragas import evaluate
            from ragas.llms import LangchainLLMWrapper
            from langchain_groq import ChatGroq
            from datasets import Dataset
        except ImportError as e:
            raise ImportError(
                f"Missing dependency: {e}. "
                "Run: pip install ragas langchain-groq datasets"
            ) from e

        # ── Groq judge LLM ────────────────────────────────────────────────────
        judge_llm = LangchainLLMWrapper(
            ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY, temperature=0)
        )

        # ── Configure metrics ─────────────────────────────────────────────────
        # Only faithfulness — it requires no embeddings, no extra deps,
        # and is the most important RAG metric (measures hallucination).
        faithfulness.llm = judge_llm
        metrics_to_run   = [faithfulness]
        local_embeddings = None

        # ── Dataset ───────────────────────────────────────────────────────────
        dataset = Dataset.from_dict({
            "user_input":         [query],
            "response":           [answer],
            "retrieved_contexts": [contexts],
        })

        # ── Evaluate ──────────────────────────────────────────────────────────
        result = evaluate(
            dataset,
            metrics=metrics_to_run,
            llm=judge_llm,
            raise_exceptions=False,
            show_progress=False,
        )

        scores    = result.to_pandas().iloc[0].to_dict()
        latency_ms = (time.time() - start) * 1000

        def _safe(key):
            v = scores.get(key)
            if v is None:
                return None
            try:
                f = float(v)
                return None if f != f else f
            except (TypeError, ValueError):
                return None

        faith = _safe("faithfulness")
        relev = _safe("answer_relevancy")

        record_ragas_eval(
            answer_id=answer_id,
            doc_name=doc_name,
            query=query,
            faithfulness=faith,
            answer_relevancy=relev,
            context_precision=None,
            context_recall=None,
            latency_ms=latency_ms,
            success=True,
        )
        logger.info(
            "RAGAS eval complete — faithfulness=%s",
            f"{faith:.3f}" if faith is not None else "n/a",
        )

    except Exception as e:
        latency_ms = (time.time() - start) * 1000
        logger.warning("RAGAS evaluation failed for answer_id=%s: %s", answer_id, e)
        record_ragas_eval(
            answer_id=answer_id,
            doc_name=doc_name,
            query=query,
            faithfulness=None,
            answer_relevancy=None,
            context_precision=None,
            context_recall=None,
            latency_ms=latency_ms,
            success=False,
            error=str(e)[:300],
        )
        record_error(
            source="ragas",
            error_type=type(e).__name__,
            message=str(e),
            doc_name=doc_name,
        )


def evaluate_async(
    answer_id: str,
    doc_name: str,
    query: str,
    answer: str,
    contexts: list[str],
) -> None:
    """Fire-and-forget RAGAS evaluation in a daemon thread."""
    t = threading.Thread(
        target=_run_evaluation,
        args=(answer_id, doc_name, query, answer, contexts),
        daemon=True,
        name=f"ragas-{answer_id[:8]}",
    )
    t.start()