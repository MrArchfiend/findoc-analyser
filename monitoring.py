"""
monitoring.py — Structured logging, request metrics, LLM call tracking.

Metrics are stored in-memory with a rolling 24-hour window.
In production, swap _metrics_store for a time-series DB (Prometheus, InfluxDB).
"""

import json
import logging
import time
import threading
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Optional


# ── JSON log formatter ────────────────────────────────────────────────────────

class JSONFormatter(logging.Formatter):
    """Emit every log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        obj = {
            "ts":      datetime.now(timezone.utc).isoformat(),
            "level":   record.levelname,
            "logger":  record.name,
            "msg":     record.getMessage(),
        }
        if record.exc_info:
            obj["exc"] = self.formatException(record.exc_info)
        # Attach any extra fields passed via extra={...}
        for key, val in record.__dict__.items():
            if key not in ("msg", "args", "levelname", "levelno", "name",
                           "pathname", "filename", "module", "exc_info",
                           "exc_text", "stack_info", "lineno", "funcName",
                           "created", "msecs", "relativeCreated", "thread",
                           "threadName", "processName", "process", "message",
                           "taskName"):
                obj[key] = val
        return json.dumps(obj, default=str)


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with JSON output. Call once at startup."""
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))


# ── In-memory rolling metrics store ──────────────────────────────────────────

_WINDOW_SECONDS = 86_400   # 24-hour rolling window
_lock = threading.Lock()

# Each entry: {"ts": float, ...fields}
_request_events:  deque = deque()
_llm_events:      deque = deque()
_ragas_events:    deque = deque()
_error_events:    deque = deque()


def _prune(dq: deque) -> None:
    """Remove entries older than the rolling window. Must be called under _lock."""
    cutoff = time.time() - _WINDOW_SECONDS
    while dq and dq[0]["ts"] < cutoff:
        dq.popleft()


# ── Request tracking ──────────────────────────────────────────────────────────

def record_request(
    method: str,
    path: str,
    status_code: int,
    latency_ms: float,
    doc_name: Optional[str] = None,
) -> None:
    with _lock:
        _request_events.append({
            "ts":          time.time(),
            "method":      method,
            "path":        path,
            "status_code": status_code,
            "latency_ms":  round(latency_ms, 2),
            "doc_name":    doc_name,
        })
        _prune(_request_events)


# ── LLM call tracking ─────────────────────────────────────────────────────────

def record_llm_call(
    call_type: str,          # "subquestions" | "response" | "feedback_response" | "identity"
    model: str,
    latency_ms: float,
    prompt_tokens: int,
    completion_tokens: int,
    doc_name: Optional[str] = None,
    success: bool = True,
) -> None:
    with _lock:
        _llm_events.append({
            "ts":               time.time(),
            "call_type":        call_type,
            "model":            model,
            "latency_ms":       round(latency_ms, 2),
            "prompt_tokens":    prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens":     prompt_tokens + completion_tokens,
            "doc_name":         doc_name,
            "success":          success,
        })
        _prune(_llm_events)


# ── RAGAS evaluation tracking ─────────────────────────────────────────────────

def record_ragas_eval(
    answer_id: str,
    doc_name: str,
    query: str,
    faithfulness: Optional[float],
    answer_relevancy: Optional[float],
    context_precision: Optional[float],
    context_recall: Optional[float],
    latency_ms: float,
    success: bool = True,
    error: Optional[str] = None,
) -> None:
    with _lock:
        _ragas_events.append({
            "ts":               time.time(),
            "answer_id":        answer_id,
            "doc_name":         doc_name,
            "query":            query[:200],
            "faithfulness":     faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall":   context_recall,
            "latency_ms":       round(latency_ms, 2),
            "success":          success,
            "error":            error,
        })
        _prune(_ragas_events)


# ── Error tracking ────────────────────────────────────────────────────────────

def record_error(
    source: str,
    error_type: str,
    message: str,
    doc_name: Optional[str] = None,
) -> None:
    with _lock:
        _error_events.append({
            "ts":         time.time(),
            "source":     source,
            "error_type": error_type,
            "message":    message[:500],
            "doc_name":   doc_name,
        })
        _prune(_error_events)


# ── Aggregated metrics summary ────────────────────────────────────────────────

def get_metrics_summary() -> dict:
    """Return a full aggregated metrics snapshot for the past 24 hours."""
    with _lock:
        now = time.time()
        for dq in (_request_events, _llm_events, _ragas_events, _error_events):
            _prune(dq)

        reqs  = list(_request_events)
        llms  = list(_llm_events)
        ragas = list(_ragas_events)
        errs  = list(_error_events)

    # ── Request metrics ──
    total_requests = len(reqs)
    status_counts: dict[int, int] = defaultdict(int)
    latencies_by_path: dict[str, list] = defaultdict(list)
    for r in reqs:
        status_counts[r["status_code"]] += 1
        latencies_by_path[r["path"]].append(r["latency_ms"])

    endpoint_stats = {}
    for path, lats in latencies_by_path.items():
        lats_s = sorted(lats)
        n = len(lats_s)
        endpoint_stats[path] = {
            "count":   n,
            "avg_ms":  round(sum(lats_s) / n, 1),
            "p50_ms":  round(lats_s[n // 2], 1),
            "p95_ms":  round(lats_s[min(int(n * 0.95), n - 1)], 1),
            "max_ms":  round(max(lats_s), 1),
        }

    error_rate = round(
        sum(1 for r in reqs if r["status_code"] >= 500) / max(total_requests, 1) * 100, 2
    )

    # ── LLM metrics ──
    total_llm_calls  = len(llms)
    total_tokens     = sum(e["total_tokens"] for e in llms)
    llm_failures     = sum(1 for e in llms if not e["success"])
    llm_by_type: dict[str, dict] = defaultdict(lambda: {"count": 0, "tokens": 0, "latency_ms": []})
    for e in llms:
        ct = e["call_type"]
        llm_by_type[ct]["count"]    += 1
        llm_by_type[ct]["tokens"]   += e["total_tokens"]
        llm_by_type[ct]["latency_ms"].append(e["latency_ms"])

    llm_type_stats = {}
    for ct, data in llm_by_type.items():
        lats = sorted(data["latency_ms"])
        n = len(lats)
        llm_type_stats[ct] = {
            "count":    data["count"],
            "tokens":   data["tokens"],
            "avg_ms":   round(sum(lats) / n, 1) if n else 0,
            "p95_ms":   round(lats[min(int(n * 0.95), n - 1)], 1) if n else 0,
        }

    # ── RAGAS metrics ──
    successful_evals = [e for e in ragas if e["success"]]

    def _avg(field: str) -> Optional[float]:
        vals = [e[field] for e in successful_evals if e[field] is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    ragas_by_doc: dict[str, list] = defaultdict(list)
    for e in successful_evals:
        ragas_by_doc[e["doc_name"]].append(e)

    doc_quality = {}
    for doc, evals in ragas_by_doc.items():
        def _doc_avg(field):
            vals = [e[field] for e in evals if e[field] is not None]
            return round(sum(vals) / len(vals), 4) if vals else None
        doc_quality[doc] = {
            "eval_count":        len(evals),
            "faithfulness":      _doc_avg("faithfulness"),
            "answer_relevancy":  _doc_avg("answer_relevancy"),
            "context_precision": _doc_avg("context_precision"),
            "context_recall":    _doc_avg("context_recall"),
        }

    # ── Error breakdown ──
    error_by_source: dict[str, int] = defaultdict(int)
    for e in errs:
        error_by_source[e["source"]] += 1

    return {
        "window_hours": 24,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "requests": {
            "total":        total_requests,
            "error_rate_pct": error_rate,
            "status_codes": dict(status_counts),
            "by_endpoint":  endpoint_stats,
        },
        "llm": {
            "total_calls":   total_llm_calls,
            "total_tokens":  total_tokens,
            "failure_count": llm_failures,
            "by_type":       llm_type_stats,
        },
        "ragas": {
            "total_evaluations":  len(ragas),
            "successful":         len(successful_evals),
            "failed":             len(ragas) - len(successful_evals),
            "avg_faithfulness":        _avg("faithfulness"),
            "avg_answer_relevancy":    _avg("answer_relevancy"),
            "avg_context_precision":   _avg("context_precision"),
            "avg_context_recall":      _avg("context_recall"),
            "by_document":             doc_quality,
        },
        "errors": {
            "total":      len(errs),
            "by_source":  dict(error_by_source),
            "recent":     errs[-10:],
        },
    }