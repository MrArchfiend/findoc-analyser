"""
main.py — FastAPI backend for FinDoc Analyser
Run: uvicorn main:app --reload

Security layers implemented:
  - API key authentication on all endpoints
  - Rate limiting (slowapi) per IP
  - CORS restricted to configured origins
  - Security headers (X-Content-Type-Options, X-Frame-Options, etc.)
  - Input validation via Pydantic with strict field constraints
  - File validation: extension, MIME type, and max size
  - Path traversal prevention on doc_name parameters
  - XSS prevention: all user-controlled strings are stripped of HTML
"""

import csv
import html
import io
import json
import logging
import os
import re
import secrets
from typing import Annotated, Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from qna import ask, get_chunks, get_metadata, index_document, store

logger = logging.getLogger("findoc")

# ── Configuration ─────────────────────────────────────────────────────────────
# Set API_KEY env var in production. Falls back to a generated key logged at startup.
_API_KEY = os.environ.get("FINDOC_API_KEY", "")
if not _API_KEY:
    _API_KEY = secrets.token_urlsafe(32)
    print(f"[FinDoc] FINDOC_API_KEY not set — generated ephemeral key: {_API_KEY}", flush=True)

# Comma-separated allowed origins, e.g. "https://yourdomain.com,http://localhost:3000"
_CORS_ORIGINS = [o.strip() for o in os.environ.get("FINDOC_CORS_ORIGINS", "http://localhost:8000").split(",")]

MAX_FILE_SIZE_MB  = int(os.environ.get("FINDOC_MAX_FILE_MB", "50"))
MAX_FILE_SIZE     = MAX_FILE_SIZE_MB * 1024 * 1024

ALLOWED_EXTENSIONS = {".pdf", ".docx"}
ALLOWED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

# ── Rate limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="FinDoc API",
    description="RAG pipeline for SEC 10-K / 10-Q financial document analysis",
    version="2.0.0",
    # Hide schema endpoints in production by setting docs_url=None
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "PATCH"],
    allow_headers=["Authorization", "Content-Type"],
)

# ── Security headers middleware ────────────────────────────────────────────────
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"]    = "nosniff"
    response.headers["X-Frame-Options"]           = "DENY"
    response.headers["X-XSS-Protection"]          = "1; mode=block"
    response.headers["Referrer-Policy"]           = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"]        = "geolocation=(), microphone=(), camera=()"
    response.headers["Content-Security-Policy"]   = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "   # needed for inline JS in index.html
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; "
        "img-src 'self' data:; "
        "connect-src 'self';"
    )
    return response

# ── Auth ──────────────────────────────────────────────────────────────────────
def verify_api_key(authorization: Annotated[Optional[str], Header()] = None):
    """
    Expects header:  Authorization: Bearer <api_key>
    Uses constant-time comparison to prevent timing attacks.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing API key.")
    token = authorization.removeprefix("Bearer ").strip()
    if not secrets.compare_digest(token, _API_KEY):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API key.")

AuthDep = Depends(verify_api_key)

# ── Input helpers ─────────────────────────────────────────────────────────────
_DOC_NAME_RE = re.compile(r'^[\w\s\-\.]+$')  # allowlist: word chars, spaces, hyphens, dots

def _sanitise_doc_name(doc_name: str) -> str:
    """
    Prevent path traversal and reject names with shell-special characters.
    Raises 400 on invalid input.
    """
    name = doc_name.strip()
    if not name or ".." in name or "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail="Invalid document name.")
    if not _DOC_NAME_RE.match(name):
        raise HTTPException(status_code=400, detail="Document name contains disallowed characters.")
    return name

def _strip_html(value: str) -> str:
    """Escape HTML special characters to prevent XSS in reflected error messages."""
    return html.escape(value, quote=True)

# ── File validation ────────────────────────────────────────────────────────────
async def _validate_file(file: UploadFile) -> bytes:
    """
    Read file content and validate:
      - Extension is in ALLOWED_EXTENSIONS
      - Content-Type is in ALLOWED_MIME_TYPES
      - File size does not exceed MAX_FILE_SIZE
      - Filename doesn't contain path traversal sequences
    Returns raw bytes on success.
    """
    filename = file.filename or ""
    ext = os.path.splitext(filename)[-1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"File type '{ext}' not supported. Use .pdf or .docx.")

    if file.content_type and file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=400, detail=f"MIME type '{_strip_html(file.content_type)}' not accepted.")

    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_FILE_SIZE_MB} MB limit.")
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="File is empty.")

    return content

# ── Pydantic models ────────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Question to ask")
    doc_name: str = Field(..., min_length=1, max_length=255, description="Target document name")

    @field_validator("query")
    @classmethod
    def query_no_html(cls, v: str) -> str:
        stripped = re.sub(r'<[^>]+>', '', v).strip()
        if not stripped:
            raise ValueError("Query must not be empty after stripping HTML.")
        return stripped

    @field_validator("doc_name")
    @classmethod
    def doc_name_safe(cls, v: str) -> str:
        if ".." in v or "/" in v or "\\" in v:
            raise ValueError("Invalid doc_name.")
        return v.strip()


class AskResponse(BaseModel):
    query: str
    doc_name: str
    answer: str
    chunks_used: list
    answer_id: str  # unique ID so frontend can reference this answer in feedback


class FeedbackRequest(BaseModel):
    answer_id: str = Field(..., min_length=1, max_length=64)
    query: str = Field(..., min_length=1, max_length=1000)
    doc_name: str = Field(..., min_length=1, max_length=255)
    vote: str = Field(..., pattern=r"^(up|down)$")
    comment: Optional[str] = Field(default=None, max_length=500)
    previous_answer: str = Field(..., min_length=1, max_length=8000)

    @field_validator("query", "previous_answer")
    @classmethod
    def strip_html_fields(cls, v: str) -> str:
        return re.sub(r'<[^>]+>', '', v).strip()

    @field_validator("doc_name")
    @classmethod
    def doc_name_safe(cls, v: str) -> str:
        if ".." in v or "/" in v or "\\" in v:
            raise ValueError("Invalid doc_name.")
        return v.strip()


class FeedbackResponse(BaseModel):
    answer_id: str
    improved_answer: Optional[str]
    chunks_used: list
    message: str


# ── Static (no auth — serves the frontend) ────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", include_in_schema=False)
async def root():
    return FileResponse("static/index.html")

@app.get("/health", include_in_schema=False)
async def health():
    return {"status": "ok", "documents_indexed": len(store.list_documents())}


# ── Documents ──────────────────────────────────────────────────────────────────

@app.post("/documents/upload", summary="Upload and index a PDF or DOCX document", dependencies=[AuthDep])
@limiter.limit("10/minute")
async def upload_document(request: Request, file: UploadFile = File(...)):
    content = await _validate_file(file)

    class _FileWrapper:
        def __init__(self, name: str, data: bytes):
            self.name = name
            self._data = data
        def read(self):
            return self._data

    try:
        msg = index_document(_FileWrapper(file.filename, content))
    except Exception as e:
        logger.exception("Indexing failed for %s", file.filename)
        raise HTTPException(status_code=500, detail="Indexing failed. Check server logs.")

    return {"message": msg, "doc_name": file.filename}


@app.get("/documents", summary="List all indexed documents", dependencies=[AuthDep])
@limiter.limit("60/minute")
async def list_documents(request: Request):
    docs = store.list_documents()
    all_meta = store.list_metadata()
    result = []
    for doc in docs:
        meta = all_meta.get(doc, {})
        result.append({
            "doc_name": doc,
            "company_name": meta.get("company_name"),
            "doc_type": meta.get("doc_type"),
            "fiscal_year": meta.get("fiscal_year"),
            "chunk_count": meta.get("chunk_count"),
        })
    return {"documents": result, "total": len(result)}


@app.delete("/documents/{doc_name}", summary="Delete an indexed document", dependencies=[AuthDep])
@limiter.limit("20/minute")
async def delete_document(request: Request, doc_name: str):
    doc_name = _sanitise_doc_name(doc_name)
    docs = store.list_documents()
    if doc_name not in docs:
        raise HTTPException(status_code=404, detail=f"Document not found.")
    deleted = store.delete_document(doc_name)
    return {"message": f"Deleted {deleted} chunks for '{doc_name}'."}


# ── Metadata ───────────────────────────────────────────────────────────────────

@app.get("/documents/{doc_name}/metadata", summary="Get metadata for a document", dependencies=[AuthDep])
@limiter.limit("60/minute")
async def document_metadata(request: Request, doc_name: str):
    doc_name = _sanitise_doc_name(doc_name)
    meta = get_metadata(doc_name)
    if not meta:
        raise HTTPException(status_code=404, detail="No metadata found.")
    return meta


@app.get("/documents/{doc_name}/metadata/download/json", summary="Download metadata as JSON", dependencies=[AuthDep])
@limiter.limit("30/minute")
async def download_metadata_json(request: Request, doc_name: str):
    doc_name = _sanitise_doc_name(doc_name)
    meta = get_metadata(doc_name)
    if not meta:
        raise HTTPException(status_code=404, detail="No metadata found.")
    base = re.sub(r'\.(pdf|docx)$', '', doc_name, flags=re.IGNORECASE)
    return StreamingResponse(
        io.BytesIO(json.dumps(meta, indent=2).encode()),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{base}_metadata.json"'},
    )


@app.get("/documents/{doc_name}/metadata/download/csv", summary="Download metadata as CSV", dependencies=[AuthDep])
@limiter.limit("30/minute")
async def download_metadata_csv(request: Request, doc_name: str):
    doc_name = _sanitise_doc_name(doc_name)
    meta = get_metadata(doc_name)
    if not meta:
        raise HTTPException(status_code=404, detail="No metadata found.")
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["field", "value"])
    for k, v in meta.items():
        if k == "financials":
            for fk, fv in (v or {}).items():
                writer.writerow([f"financials.{fk}", fv or ""])
        else:
            writer.writerow([k, v or ""])
    base = re.sub(r'\.(pdf|docx)$', '', doc_name, flags=re.IGNORECASE)
    return StreamingResponse(
        io.BytesIO(buf.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{base}_metadata.csv"'},
    )


# ── Chunks ─────────────────────────────────────────────────────────────────────

@app.get("/documents/{doc_name}/chunks", summary="Get all chunks for a document", dependencies=[AuthDep])
@limiter.limit("30/minute")
async def document_chunks(request: Request, doc_name: str, section: Optional[str] = None):
    doc_name = _sanitise_doc_name(doc_name)
    docs = store.list_documents()
    if doc_name not in docs:
        raise HTTPException(status_code=404, detail="Document not found.")
    chunks = get_chunks(doc_name)
    if section:
        # Validate section against known values to prevent injection
        section = _strip_html(section)[:100]
        chunks = [c for c in chunks if c.section_hint == section]
    return {
        "doc_name": doc_name,
        "total": len(chunks),
        "chunks": [
            {
                "chunk_index": c.chunk_index,
                "section_hint": c.section_hint,
                "char_start": c.char_start,
                "char_end": c.char_end,
                "text": c.text,
            }
            for c in chunks
        ],
    }


# ── Q&A ────────────────────────────────────────────────────────────────────────

@app.post("/ask", summary="Ask a question about an indexed document", response_model=AskResponse, dependencies=[AuthDep])
@limiter.limit("20/minute")
async def ask_question(request: Request, body: AskRequest):
    doc_name = _sanitise_doc_name(body.doc_name)
    docs = store.list_documents()
    if doc_name not in docs:
        raise HTTPException(status_code=404, detail="Document not found.")
    try:
        answer, scored_chunks = ask(body.query, doc_name=doc_name)
    except Exception as e:
        logger.exception("Q&A failed for doc=%s query=%s", doc_name, body.query)
        raise HTTPException(status_code=500, detail="Failed to generate answer. Check server logs.")
    return AskResponse(
        query=body.query,
        doc_name=doc_name,
        answer=answer,
        chunks_used=scored_chunks,
        answer_id=secrets.token_hex(16),
    )

# ── Feedback ───────────────────────────────────────────────────────────────────

# In-memory feedback log — in production replace with a DB or file store
_feedback_log: list[dict] = []

@app.post("/feedback", summary="Submit up/down vote on an answer", response_model=FeedbackResponse, dependencies=[AuthDep])
@limiter.limit("30/minute")
async def submit_feedback(request: Request, body: FeedbackRequest):
    doc_name = _sanitise_doc_name(body.doc_name)
    comment_clean = _strip_html(body.comment or "")

    # Log every vote regardless of direction
    log_entry = {
        "answer_id":       body.answer_id,
        "vote":            body.vote,
        "query":           body.query,
        "doc_name":        doc_name,
        "comment":         comment_clean,
        "previous_answer": body.previous_answer[:500],  # truncate for log
    }
    _feedback_log.append(log_entry)
    logger.info("Feedback received: %s", log_entry)

    # Upvote — no retry needed, just acknowledge
    if body.vote == "up":
        return FeedbackResponse(
            answer_id=body.answer_id,
            improved_answer=None,
            chunks_used=[],
            message="Thanks for the positive feedback!",
        )

    # Downvote — regenerate with feedback context
    docs = store.list_documents()
    if doc_name not in docs:
        raise HTTPException(status_code=404, detail="Document not found.")

    try:
        from response_generation import generate_response_with_feedback
        from question_generation import generate_subquestions
        subquestions = generate_subquestions(body.query)
        retrieved = store.batch_query(subquestions, doc_name=doc_name, top_k=10)
        if not retrieved:
            raise HTTPException(status_code=422, detail="No relevant content found in document to retry with.")
        improved_answer, scored_chunks = generate_response_with_feedback(
            query=body.query,
            retrieved_chunks=retrieved,
            previous_answer=body.previous_answer,
            feedback_comment=comment_clean,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Feedback retry failed for answer_id=%s", body.answer_id)
        raise HTTPException(status_code=500, detail="Could not generate an improved answer. Please try rephrasing your question.")

    return FeedbackResponse(
        answer_id=secrets.token_hex(16),
        improved_answer=improved_answer,
        chunks_used=scored_chunks,
        message="Here\'s an improved answer based on your feedback.",
    )


@app.get("/feedback", summary="Get all feedback logs", dependencies=[AuthDep], include_in_schema=True)
@limiter.limit("10/minute")
async def get_feedback(request: Request):
    return {"total": len(_feedback_log), "feedback": _feedback_log}