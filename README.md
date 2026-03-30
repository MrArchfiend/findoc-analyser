# FinDoc Analyser: Financial Document Analysis System using Ai

A production-grade **Retrieval-Augmented Generation (RAG)** pipeline for analysing SEC 10-K and 10-Q financial documents. Upload any annual or quarterly filing as a PDF or DOCX, and ask natural-language questions to get accurate, source-grounded answers — with full observability, RAGAS evaluation, and a FastAPI backend alongside the Streamlit UI.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the App](#running-the-app)
- [Running the API](#running-the-api)
- [API Reference](#api-reference)
- [Module Reference](#module-reference)
- [Key Design Decisions](#key-design-decisions)
- [Monitoring & Evaluation](#monitoring--evaluation)
- [Security](#security)
- [Roadmap](#roadmap)
- [Contributors](#contributors)

---

## Overview

FinDoc Analyser v2 is a full-stack AI application that lets analysts, investors, and developers interrogate SEC filings in plain English. The system:

- Accepts **PDF** and **DOCX** uploads (including scanned/image PDFs via OCR fallback)
- Extracts and cleans text, then splits it into semantically coherent chunks
- Embeds chunks using a local sentence-transformer model and stores them in **ChromaDB**
- Decomposes broad user queries into focused sub-questions for better retrieval
- Generates answers with **Groq's LLaMA 3.3 70B** via the Groq API
- Extracts structured financial metadata (revenue, margins, EPS, ratios, etc.) through a two-phase LLM + regex pipeline
- Evaluates every answer asynchronously using **RAGAS** (faithfulness metric)
- Exposes a **FastAPI** REST API with authentication, rate limiting, CORS, and security headers
- Provides a **Streamlit** UI with a chat interface, document library, chunk viewer, and metadata dashboard

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          User Interfaces                            │
│                                                                     │
│    Streamlit (app.py)                  FastAPI REST (main.py)       │
│    ├── Chat tab                        ├── POST /ingest             │
│    ├── Documents tab (metadata+KPIs)   ├── POST /ask                │
│    └── Chunk Viewer tab                ├── GET  /documents          │
│                                        ├── POST /feedback           │
│                                        └── GET  /metrics            │
└─────────────────────┬───────────────────────────┬───────────────────┘
                      │                           │
                      ▼                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        Pipeline Orchestrator                        │
│                             qna.py                                  │
│   index_document()  ──►  preprocess → chunk → embed → store        │
│   ask()             ──►  subquestions → retrieve → generate         │
└──────┬──────────────────────────────────────────────────────────────┘
       │
       ├──► data_preprocessing.py   (PDF extraction, OCR, text cleaning)
       ├──► chunking.py             (paragraph-aware chunker + section labels)
       ├──► storing_retrieval.py    (ChromaDB vector store wrapper)
       ├──► question_generation.py  (sub-question decomposition via Groq)
       ├──► response_generation.py  (answer synthesis via Groq)
       ├──► metadata_extraction.py  (LLM identity + regex financials)
       ├──► evaluation.py           (async RAGAS faithfulness scoring)
       └──► monitoring.py           (structured JSON logging + rolling metrics)
```

### RAG Data Flow

```
1. User uploads PDF/DOCX
         │
         ▼
   data_preprocessing.py
   ├── pypdf layout extraction (pass 1)
   ├── OCR fallback for scanned pages (pass 2, Tesseract)
   └── Text cleaning (spacing, merged words, boilerplate removal)
         │
         ▼
   chunking.py  (ParagraphAwareChunker)
   ├── Split on double-newlines into paragraphs
   ├── Merge small paragraphs to fill max_chars window (1500 chars)
   ├── Sentence-split oversized paragraphs with overlap
   └── Label chunks with 10-K section (Risk Factors, MD&A, etc.)
         │
         ▼
   storing_retrieval.py  (VectorStore → ChromaDB)
   ├── Embed with all-MiniLM-L6-v2 (sentence-transformers)
   └── Upsert with stable content-hash IDs (idempotent)
         │
         ▼
   metadata_extraction.py
   ├── Phase 1: LLM call on first 3500 chars → company name, ticker,
   │            doc type, fiscal year, period end date, summary
   └── Phase 2: Regex scan of all chunks → revenue, EPS, margins,
               ratios, cash flow, EBITDA (formula-derived)

2. User asks a question
         │
         ▼
   question_generation.py
   └── Groq LLaMA 3.3 → 3 focused sub-questions + original query
         │
         ▼
   storing_retrieval.py  (batch_query)
   ├── Query ChromaDB for each sub-question (cosine similarity)
   ├── Deduplicate by chunk_index across sub-questions
   └── Return top-K chunks sorted by score
         │
         ▼
   response_generation.py
   └── Groq LLaMA 3.3 → final answer from retrieved context
         │
         ▼
   evaluation.py  (async, daemon thread)
   └── RAGAS faithfulness score via Groq judge LLM
```

---

## Project Structure

```
findoc-analyser/
│
├── app.py                  # Streamlit UI (chat, document library, chunk viewer)
├── main.py                 # FastAPI REST API (all endpoints, auth, rate limiting)
│
├── qna.py                  # Pipeline orchestrator — index_document(), ask()
│
├── data_preprocessing.py   # PDF extraction (layout mode + OCR), DOCX extraction,
│                           # text cleaning (spacing, merged words, boilerplate)
│
├── chunking.py             # ParagraphAwareChunker + Chunk dataclass +
│                           # 10-K section detection registry
│
├── storing_retrieval.py    # VectorStore class wrapping ChromaDB:
│                           # add_chunks(), query(), batch_query(),
│                           # list_documents(), delete_document(),
│                           # save/load metadata (JSON files)
│
├── question_generation.py  # Groq call → 3 sub-questions per user query
│
├── response_generation.py  # Groq call → answer from retrieved context;
│                           # second call for feedback-driven retry
│
├── metadata_extraction.py  # Two-phase extraction:
│                           # Phase 1 — LLM on front matter (identity fields)
│                           # Phase 2 — regex scan of chunks (financial KPIs)
│
├── evaluation.py           # Async RAGAS faithfulness evaluation
│                           # (Groq as judge LLM, daemon thread per answer)
│
├── monitoring.py           # JSON structured logging, in-memory rolling metrics
│                           # (requests, LLM calls, RAGAS scores, errors)
│
├── constants.py            # All configuration constants (API keys, model names,
│                           # ChromaDB paths, embedding settings, retrieval K)
│
├── requirements.txt        # All Python dependencies with version pins
├── README.md               # This file
│
├── chromadb/               # ChromaDB persistent storage (auto-created)
│   └── metadata/           # Per-document JSON metadata files (auto-created)
│
└── data/                   # Optional: place 10-K PDFs here for batch indexing
```

---

## Installation

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | ≥ 3.10 | f-strings with `str \| None` type syntax required |
| Tesseract OCR | any | Only needed for scanned/image PDFs |
| Groq API key | — | Free tier available at console.groq.com |

#### Install Tesseract (for OCR fallback)

```bash
# Ubuntu / Debian
sudo apt install tesseract-ocr

# macOS
brew install tesseract

# Windows
# Download installer from https://github.com/UB-Mannheim/tesseract/wiki
```

### Python Dependencies

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate.bat       # Windows

# Install all dependencies
pip install -r requirements.txt
```

The `requirements.txt` installs:

| Package | Purpose |
|---|---|
| `groq` | LLM inference (question generation, response, metadata) |
| `pypdf` | PDF text extraction (layout mode) |
| `chromadb` | Vector store (embedding storage + similarity search) |
| `sentence-transformers` | Local embedding model (all-MiniLM-L6-v2) |
| `nltk` | Sentence tokenisation for oversized paragraph splitting |
| `python-docx` | DOCX text extraction |
| `pdf2image` | PDF → image conversion for OCR fallback |
| `pytesseract` | Tesseract OCR Python wrapper |
| `fastapi` | REST API framework |
| `uvicorn[standard]` | ASGI server for FastAPI |
| `python-multipart` | File upload support for FastAPI |
| `slowapi` | Rate limiting middleware |
| `ragas` | RAG evaluation (faithfulness, answer relevancy) |
| `langchain-groq` | Groq LangChain integration (RAGAS judge LLM) |
| `langchain-community` | HuggingFace embeddings for RAGAS |
| `datasets` | `Dataset.from_dict()` required by RAGAS |
| `streamlit` | Web UI |

---

## Configuration

All constants live in `constants.py`. Override them there or set environment variables for deployment.

```python
# constants.py (defaults shown)

# ── Groq ──────────────────────────────────────────────────────────────────
GROQ_API_KEY   = "your_groq_api_key_here"   # or set GROQ_API_KEY env var
GROQ_MODEL     = "llama-3.3-70b-versatile"

# ── ChromaDB ──────────────────────────────────────────────────────────────
CHROMADB_DIR         = "chromadb"
CHROMA_PERSIST_DIR   = "chromadb"
CHROMA_COLLECTION_NAME = "findoc"

# ── Embeddings ────────────────────────────────────────────────────────────
EMBEDDING_MODEL           = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS      = 384
EMBEDDING_BACKEND         = "sentence-transformers"  # or "ollama"
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"

# ── Ollama (optional alternative embedding backend) ───────────────────────
OLLAMA_EMBED_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL    = "http://localhost:11434"

# ── Retrieval ─────────────────────────────────────────────────────────────
TOP_K_RESULTS = 5    # chunks retrieved per sub-question from ChromaDB
TOP_K_TO_LLM  = 5    # top chunks passed to the LLM after re-ranking
```

### FastAPI environment variables

The FastAPI server reads these at startup:

| Variable | Default | Description |
|---|---|---|
| `FINDOC_API_KEY` | auto-generated | Bearer token for API authentication. Set this in production — the ephemeral key is logged at startup and lost on restart. |
| `FINDOC_CORS_ORIGINS` | `http://localhost:8000` | Comma-separated list of allowed CORS origins |
| `FINDOC_MAX_FILE_MB` | `50` | Maximum upload file size in megabytes |
| `LOG_LEVEL` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

---

## Running the App

### Streamlit UI (recommended for interactive use)

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

**Workflow:**
1. In the **sidebar**, click "Browse files" and upload a 10-K or 10-Q PDF/DOCX
2. Click **Index Document** — this extracts text, chunks it, embeds it, and extracts metadata. Large documents take 10–30 seconds.
3. The document appears in the **Indexed documents** list. Click it to make it active.
4. Switch to the **Chat** tab and type your question.
5. The answer appears with an expandable **Chunks used** panel showing which passages were retrieved, their section label, and their cosine similarity score.
6. Browse the **Documents** tab to see extracted financial KPIs (revenue, margins, EPS, etc.) and download metadata as JSON or CSV.
7. Use the **Chunk Viewer** tab to inspect every chunk, filter by 10-K section, and see exact character offsets.

---

## Running the API

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Interactive API docs are available at:
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

On first startup, if `FINDOC_API_KEY` is not set, an ephemeral key is printed to stdout:

```
[FinDoc] FINDOC_API_KEY not set — generated ephemeral key: <token>
```

Use this token as a Bearer header on all requests:

```
Authorization: Bearer <token>
```

---

## API Reference

All endpoints require `Authorization: Bearer <api_key>` unless noted otherwise.

### `GET /health`
Health check. No authentication required.

**Response:**
```json
{"status": "ok"}
```

---

### `POST /ingest`
Upload and index a PDF or DOCX document.

**Request:** `multipart/form-data` with field `file`

**Constraints:**
- Extensions: `.pdf`, `.docx`
- Max size: 50 MB (configurable via `FINDOC_MAX_FILE_MB`)
- MIME types: `application/pdf`, `application/vnd.openxmlformats-...`

**Rate limit:** 10/minute per IP

**Response:**
```json
{
  "doc_name": "apple_10k_2024.pdf",
  "message": "'apple_10k_2024.pdf' indexed — 312 chunks, company: Apple Inc..",
  "chunk_count": 312,
  "metadata": {
    "company_name": "Apple Inc.",
    "ticker": "AAPL",
    "doc_type": "10-K",
    "fiscal_year": "2024",
    "period_end_date": "2024-09-28",
    "summary": "Apple Inc. is a global technology company...",
    "financials": {
      "revenue": "$391.04B",
      "net_income": "$93.74B",
      "gross_margin": "45.6%",
      ...
    }
  }
}
```

---

### `POST /ask`
Ask a natural-language question about an indexed document.

**Rate limit:** 20/minute per IP

**Request body:**
```json
{
  "query": "What are the main risk factors for the company?",
  "doc_name": "apple_10k_2024.pdf"
}
```

**Response:**
```json
{
  "query": "What are the main risk factors for the company?",
  "doc_name": "apple_10k_2024.pdf",
  "answer": "The company identifies several key risk factors including...",
  "chunks_used": [
    {
      "text": "The company faces intense competition...",
      "score": 0.8912,
      "section_hint": "Risk Factors",
      "chunk_index": 47
    }
  ],
  "answer_id": "3f8a1c2e..."
}
```

The `answer_id` is used to submit feedback. A RAGAS faithfulness evaluation is triggered asynchronously and does not delay this response.

---

### `GET /documents`
List all indexed documents with their metadata.

**Response:**
```json
{
  "documents": ["apple_10k_2024.pdf", "msft_10k_2024.pdf"],
  "metadata": {
    "apple_10k_2024.pdf": { "company_name": "Apple Inc.", ... },
    "msft_10k_2024.pdf": { "company_name": "Microsoft Corporation", ... }
  }
}
```

---

### `GET /documents/{doc_name}`
Get full metadata for a specific document.

---

### `DELETE /documents/{doc_name}`
Delete all chunks and metadata for a document.

**Response:**
```json
{"message": "'apple_10k_2024.pdf' deleted — 312 chunks removed."}
```

---

### `GET /documents/{doc_name}/metadata/download/json`
Download extracted metadata as a JSON file.

---

### `GET /documents/{doc_name}/metadata/download/csv`
Download extracted metadata as a flat CSV file (`field,value` rows, with `financials.*` prefix for financial fields).

---

### `GET /documents/{doc_name}/chunks`
Retrieve all chunks for a document.

**Query parameters:**
- `section` (optional): Filter by 10-K section label (e.g. `Risk Factors`, `MD&A`)

**Response:**
```json
{
  "doc_name": "apple_10k_2024.pdf",
  "total": 312,
  "chunks": [
    {
      "chunk_index": 0,
      "section_hint": "Risk Factors",
      "char_start": 0,
      "char_end": 1487,
      "text": "..."
    }
  ]
}
```

---

### `POST /feedback`
Submit a thumbs-up or thumbs-down vote on an answer. A downvote triggers automatic answer regeneration using the original context plus the feedback comment.

**Rate limit:** 30/minute per IP

**Request body:**
```json
{
  "answer_id": "3f8a1c2e...",
  "vote": "down",
  "query": "What are the main risk factors?",
  "doc_name": "apple_10k_2024.pdf",
  "previous_answer": "The company faces...",
  "comment": "Answer was too vague, please be more specific about supply chain risks."
}
```

`vote` must be `"up"` or `"down"`.

**Response (downvote):**
```json
{
  "answer_id": "new_id...",
  "improved_answer": "The company's 10-K identifies several specific supply chain risks...",
  "chunks_used": [...],
  "message": "Here's an improved answer based on your feedback."
}
```

---

### `GET /feedback`
Retrieve the full in-memory feedback log (all votes and comments).

---

### `GET /metrics`
Full 24-hour rolling metrics snapshot covering requests, LLM calls, RAGAS scores, and errors.

**Response structure:**
```json
{
  "window_hours": 24,
  "generated_at": "2024-10-01T12:00:00Z",
  "requests": {
    "total": 142,
    "error_rate_pct": 0.7,
    "status_codes": {"200": 140, "422": 2},
    "by_endpoint": {
      "/ask": {"count": 55, "avg_ms": 3210, "p50_ms": 2980, "p95_ms": 5400, "max_ms": 8100}
    }
  },
  "llm": {
    "total_calls": 165,
    "total_tokens": 284000,
    "failure_count": 0,
    "by_type": {
      "subquestions": {"count": 55, "tokens": 28000, "avg_ms": 420, "p95_ms": 680},
      "response": {"count": 55, "tokens": 210000, "avg_ms": 2800, "p95_ms": 5100},
      "identity": {"count": 3, "tokens": 4200, "avg_ms": 380, "p95_ms": 500}
    }
  },
  "ragas": {
    "total_evaluations": 55,
    "successful": 53,
    "avg_faithfulness": 0.8741,
    "avg_answer_relevancy": null,
    "by_document": {
      "apple_10k_2024.pdf": {"eval_count": 30, "faithfulness": 0.892}
    }
  },
  "errors": {
    "total": 2,
    "by_source": {"ragas": 2},
    "recent": [...]
  }
}
```

---

### `GET /metrics/quality`
RAGAS quality scores (faithfulness, answer relevancy, context precision, context recall) aggregated overall and per document.

---

### `GET /metrics/llm`
LLM call statistics and token usage broken down by call type.

---

## Module Reference

### `data_preprocessing.py`

**`preprocess_data(uploaded_file) → str`**

Entry point for document ingestion. Detects file type, extracts text, cleans it, and returns a single string.

**PDF extraction** uses a two-pass strategy:
- **Pass 1:** `pypdf` layout-mode extraction on every page. Layout mode uses character x-positions to reconstruct word spacing, fixing the run-together-words problem common in PDF extraction.
- **Pass 2:** For any page that yields no text (scanned/image pages), the page is converted to a 300 DPI image and run through Tesseract OCR. `pdf2image` and `pytesseract` are imported lazily so they don't add startup cost when OCR isn't needed.

**Text cleaning** performs:
- Fixes spaced-out single characters from column layouts (e.g. `R e v e n u e` → `Revenue`) — only triggers on 3+ consecutive single-letter tokens to avoid touching normal text
- Fixes camelCase-merged words where the space was lost during extraction
- Fixes merged bracket tokens (`95014(Address` → `95014 (Address`)
- Removes standalone page numbers (lines containing only digits)
- Strips common 10-K boilerplate that repeats on every page (SEC header, Form 10-K label, etc.)
- Collapses multiple spaces/tabs to a single space
- Collapses 3+ newlines to 2 (preserves paragraph breaks, removes excessive blank lines)

---

### `chunking.py`

**`ParagraphAwareChunker(max_chars=1500, overlap_sentences=2)`**

Splits document text into semantically coherent chunks. Uses a three-path strategy:

- **PATH A/B (merge):** Paragraphs shorter than `max_chars` are buffered and merged with neighbours. When adding a paragraph would overflow the buffer, the last paragraph is carried forward as overlap before starting a new segment.
- **PATH C (sentence-split):** Paragraphs longer than `max_chars` are passed to NLTK's sentence tokeniser. Sentences are accumulated into sub-chunks with `overlap_sentences` sentences of overlap between consecutive sub-chunks.

**`_detect_section(text)`** scans the first 300 characters of each chunk against a registry of `(regex, section_name)` pairs to classify it into a 10-K section:

| Detected Section | Example Trigger |
|---|---|
| Risk Factors | "risk factors" |
| MD&A | "management's discussion" |
| Financial Statements | "financial statements" |
| Properties | "properties" |
| Legal Proceedings | "legal proceedings" |
| Market Risk | "quantitative and qualitative" |
| Controls & Procedures | "controls and procedures" |
| Executive Compensation | "executive compensation" |
| Security Ownership | "security ownership" |

**`Chunk` dataclass fields:**
- `text` — the raw chunk text
- `chunk_index` — sequential position in the document
- `method` — always `"paragraph-aware"` for this chunker
- `char_start`, `char_end` — character offsets in the cleaned document
- `section_hint` — detected 10-K section label, or `None`
- `extra` — caller-supplied extra metadata (dict)

---

### `storing_retrieval.py`

**`VectorStore`** wraps a ChromaDB persistent collection with cosine similarity search.

**Key methods:**

| Method | Description |
|---|---|
| `add_chunks(chunks, doc_name)` | Upsert chunks with stable MD5 IDs derived from `doc_name:chunk_index:char_start`. Safe to call multiple times — ChromaDB upsert deduplicates. |
| `query(query_text, top_k, doc_name, section_hint)` | Single-query retrieval. Converts cosine distance to similarity score (`1.0 - distance`). |
| `batch_query(queries, top_k, doc_name, deduplicate)` | Multi-query retrieval for sub-questions. Merges and deduplicates by chunk identity, then re-sorts by score. |
| `get_all_chunks(doc_name)` | Returns all chunks for a document sorted by `chunk_index`. Used by the Chunk Viewer tab. |
| `list_documents()` | Returns sorted list of unique `doc_name` values in the collection. |
| `delete_document(doc_name)` | Deletes all chunks with matching `doc_name` metadata. |
| `save_metadata(doc_name, metadata)` | Writes a JSON file to `chromadb/metadata/<doc_name>.json`. |
| `load_metadata(doc_name)` | Reads the JSON metadata file. Returns `None` if not found. |
| `list_metadata()` | Returns a dict of all stored metadata files, keyed by `doc_name`. |

**Embedding backend** is selected via `EMBEDDING_BACKEND`:
- `"sentence-transformers"` (default): Uses `all-MiniLM-L6-v2` locally. No external service required. 384-dimensional embeddings.
- `"ollama"`: Uses Ollama's embedding API (requires Ollama running at `OLLAMA_BASE_URL`). Useful for GPU-accelerated embedding.

**`RetrievedChunk` dataclass** — returned by all query methods:
- `text`, `score`, `doc_name`, `chunk_index`, `section_hint`, `char_start`, `char_end`

---

### `question_generation.py`

**`generate_subquestions(query, doc_name) → list[str]`**

Sends the user's query to Groq and requests 3 focused sub-questions as a JSON object. The original query is always prepended to the returned list so retrieval always includes results for the literal user question. Falls back to `[query]` on any error (network failure, JSON parse error, API limit).

**Prompt pattern:**
```
You are a financial analyst assistant.
A user asked: "<query>"
Break this into 3 specific sub-questions to help retrieve relevant information from a 10-K financial document.
Return ONLY a JSON object like this: {"questions": ["q1", "q2", "q3"]}
```

---

### `response_generation.py`

**`generate_response(query, retrieved_chunks, doc_name) → (str, list)`**

Selects the top `TOP_K_TO_LLM` chunks, truncates each to 800 characters to stay within Groq's context limits, and generates a grounded answer. Returns the answer text and a list of scored chunk dicts for display.

**`generate_response_with_feedback(query, retrieved_chunks, previous_answer, feedback_comment, doc_name) → (str, list)`**

Used by the `/feedback` endpoint on downvotes. Includes the previous (rejected) answer and the user's comment in the prompt to guide improvement. Uses `temperature=0.2` (slightly lower than the default `0.3`) for more deterministic improvement.

---

### `metadata_extraction.py`

Two-phase extraction with no hallucination risk on financial numbers:

**Phase 1 — LLM identity extraction:**
Sends the first 3500 characters of the document to Groq with `response_format={"type": "json_object"}` and extracts:
- `company_name`, `ticker`, `doc_type`, `fiscal_year`, `period_end_date`, `summary`

**Phase 2 — Regex financial extraction:**
Walks every chunk and applies a registry of `(field_name, [regex_patterns])` pairs. Patterns are tried in order; the first positive match per field across all chunks wins. Numeric values are scaled by a document-level multiplier inferred from phrases like "in millions" or "in billions".

Extracted raw values → formula-derived metrics:

| Metric | Formula |
|---|---|
| Gross Margin | `gross_profit / revenue × 100` |
| Operating Margin | `operating_income / revenue × 100` |
| Net Margin | `net_income / revenue × 100` |
| Debt-to-Equity | `total_liabilities / shareholders_equity` |
| Current Ratio | `current_assets / current_liabilities` |
| Return on Equity | `net_income / shareholders_equity × 100` |
| Return on Assets | `net_income / total_assets × 100` |
| Free Cash Flow | `operating_cash_flow − capital_expenditures` |
| EBITDA | Extracted directly, or `operating_income + depreciation`, or `net_income + tax + interest + depreciation` |

---

### `evaluation.py`

**`evaluate_async(answer_id, doc_name, query, answer, contexts)`**

Spawns a daemon thread to run RAGAS faithfulness evaluation after each `/ask` response. The main request returns immediately — evaluation results are recorded in `monitoring.py` and retrievable via `/metrics/quality`.

Uses Groq's LLaMA 3.3 70B as the judge LLM via `LangchainLLMWrapper`. Only the **faithfulness** metric is enabled by default (measures whether the answer is grounded in the retrieved context, i.e. detects hallucination). `answer_relevancy` requires local embeddings and is skipped if `sentence-transformers` is unavailable.

**RAGAS dataset schema used:**
```
user_input         → the query
response           → the generated answer
retrieved_contexts → list of chunk texts sent to the LLM
```

---

### `monitoring.py`

Structured JSON logging and in-memory rolling metrics (24-hour window). In production, swap `_metrics_store` for Prometheus / InfluxDB.

**Four event streams** (all in-memory `deque`s with automatic pruning):

| Stream | Recorded by | Key fields |
|---|---|---|
| `_request_events` | FastAPI middleware | method, path, status_code, latency_ms, doc_name |
| `_llm_events` | question_generation, response_generation, metadata_extraction | call_type, model, latency_ms, prompt_tokens, completion_tokens, success |
| `_ragas_events` | evaluation.py | answer_id, faithfulness, answer_relevancy, latency_ms, success |
| `_error_events` | all modules | source, error_type, message, doc_name |

**`get_metrics_summary()`** returns a full aggregated snapshot including p50/p95/max latency per endpoint, token usage per LLM call type, average RAGAS scores overall and per document, and recent errors.

**`setup_logging(level)`** configures the root logger to emit every log record as a single-line JSON object with timestamp, level, logger name, message, and any extra fields passed via `extra={...}`.

---

## Key Design Decisions

**Paragraph-aware chunking over fixed-size sliding window**
Fixed character windows often split mid-sentence or mid-table, producing incoherent chunks that hurt retrieval precision. Paragraph-aware chunking respects the document's natural structure, so every chunk is a coherent passage a human could read.

**Section detection for retrieval precision**
10-K documents have highly distinct sections with different vocabularies. Labelling chunks by section allows targeted queries (e.g. "what are the risk factors?" naturally retrieves from the Risk Factors section rather than from the Financial Statements section).

**Sub-question decomposition**
Broad queries like "how is the company performing financially?" are too diffuse for single-shot vector search. Decomposing them into 3 focused sub-questions (e.g. "What was revenue growth year over year?", "What were the operating margins?", "What guidance did management give?") pulls relevant passages from multiple sections and significantly improves answer quality.

**Idempotent indexing**
Chunk IDs are deterministic MD5 hashes of `doc_name:chunk_index:char_start`. Re-uploading the same document calls ChromaDB's `upsert` rather than `add`, so no duplicates are created.

**No LLM for financial numbers**
Financial KPIs are extracted by regex, not by LLM. This eliminates hallucinated numbers — a critical reliability requirement for financial analysis. The LLM is only used for unstructured identity fields (company name, ticker, summary) where exact extraction is less important than fluency.

**Async RAGAS evaluation**
RAGAS evaluation takes 5–15 seconds per answer. Running it in a daemon thread means the user receives their answer immediately, while quality scores accumulate in the background and are visible via `/metrics/quality`.

**Two UI surfaces**
The Streamlit app is optimised for interactive exploration by analysts. The FastAPI backend enables integration into other systems, CI pipelines, or bulk processing workflows.

---

## Monitoring & Evaluation

### Viewing metrics

```bash
# Via API (replace <token> with your API key)
curl -H "Authorization: Bearer <token>" http://localhost:8000/metrics | python -m json.tool

# Quality scores by document
curl -H "Authorization: Bearer <token>" http://localhost:8000/metrics/quality | python -m json.tool

# LLM token usage
curl -H "Authorization: Bearer <token>" http://localhost:8000/metrics/llm | python -m json.tool
```

### RAGAS Faithfulness Score

Faithfulness measures whether every claim in the generated answer is supported by the retrieved context. A score of 1.0 means every claim is grounded; lower scores indicate potential hallucination. Target: **> 0.85** for production use.

### Log format

All logs are emitted as JSON to stdout:
```json
{
  "ts": "2024-10-01T12:00:00.123Z",
  "level": "INFO",
  "logger": "findoc.response_generation",
  "msg": "LLM call complete",
  "latency_ms": 2841.3,
  "doc_name": "apple_10k_2024.pdf"
}
```

Pipe to `jq` for filtering:
```bash
# Show only errors
uvicorn main:app 2>&1 | jq 'select(.level == "ERROR")'

# Monitor latency
uvicorn main:app 2>&1 | jq 'select(.logger == "findoc.response_generation") | .latency_ms'
```

---

## Security

The FastAPI backend implements multiple security layers:

| Layer | Implementation |
|---|---|
| **Authentication** | Bearer token (`Authorization: Bearer <key>`) with constant-time comparison (`secrets.compare_digest`) to prevent timing attacks |
| **Rate limiting** | `slowapi` — 60 req/min default, 20/min for `/ask`, 10/min for `/ingest` |
| **CORS** | Restricted to configured origins via `FINDOC_CORS_ORIGINS` |
| **Security headers** | `X-Content-Type-Options`, `X-Frame-Options: DENY`, `X-XSS-Protection`, `Referrer-Policy`, `Content-Security-Policy`, `Permissions-Policy` |
| **Input validation** | Pydantic models with `min_length`, `max_length`, HTML-strip validators on all user input |
| **File validation** | Extension allowlist (`.pdf`, `.docx`), MIME type allowlist, max file size check, empty file check |
| **Path traversal prevention** | `doc_name` parameters validated against `^[\w\s\-\.]+$` regex; `..`, `/`, `\` rejected with HTTP 400 |
| **XSS prevention** | All user-controlled strings reflected in error messages are HTML-escaped via `html.escape()` |

**Production checklist:**
- Set `FINDOC_API_KEY` to a strong random secret before deployment
- Set `FINDOC_CORS_ORIGINS` to your actual frontend domain(s)
- Set `docs_url=None` in `FastAPI(...)` to hide the Swagger UI from public access
- Replace the in-memory feedback log and metrics store with a persistent database
- Run behind a TLS-terminating reverse proxy (nginx, Caddy, AWS ALB)

---

## Roadmap

### Version 1 (completed)
- Upload any 10-K / SEC PDF or DOCX
- Ask natural-language questions with RAG-grounded answers
- Paragraph-aware chunking with 10-K section detection
- Sub-question decomposition for improved retrieval

### Version 2 (current)
- Structured financial KPI extraction (revenue, margins, EPS, ratios, cash flow)
- Two-phase metadata extraction (LLM identity + regex financials)
- FastAPI REST backend with auth, rate limiting, and security hardening
- Async RAGAS faithfulness evaluation per answer
- Structured JSON logging and 24-hour rolling metrics
- Feedback loop with automatic answer regeneration on downvote
- Chunk Viewer in UI with section filtering
- Metadata export (JSON + CSV)

---

## Contributors

- **Aditya Singh** — Architecture Design and Model Development
- **Dheeraj Yadav** — Monitoring, Security and UI/UX
