import json
import re
from groq import Groq
from constants import GROQ_API_KEY, GROQ_MODEL

client = Groq(api_key=GROQ_API_KEY)

_EXTRACTION_PROMPT = """
You are a financial data extraction expert. You are given text from a 10-K or SEC filing.
Extract structured metadata from whatever information is present.

Return ONLY a valid JSON object with this exact schema (use null for any field you cannot find):
{{
  "company_name": "string",
  "ticker": "string or null",
  "doc_type": "10-K or 10-Q or 8-K or other",
  "fiscal_year": "YYYY or null",
  "period_end_date": "YYYY-MM-DD or null",
  "summary": "2-3 sentence plain English summary of the company and filing",
  "financials": {{
    "revenue": "string with units e.g. $12.3B or null",
    "net_income": "string with units or null",
    "eps_basic": "string or null",
    "eps_diluted": "string or null",
    "total_assets": "string with units or null",
    "total_liabilities": "string with units or null",
    "shareholders_equity": "string with units or null",
    "operating_cash_flow": "string with units or null",
    "gross_margin": "percentage string or null",
    "operating_margin": "percentage string or null",
    "net_margin": "percentage string or null",
    "debt_to_equity": "ratio string or null",
    "current_ratio": "ratio string or null",
    "return_on_equity": "percentage string or null",
    "return_on_assets": "percentage string or null",
    "revenue_growth_yoy": "percentage string or null",
    "ebitda": "string with units or null",
    "free_cash_flow": "string with units or null"
  }}
}}

Document text:
{text}
"""

# Keywords that signal the start of financial data in the document body.
_FINANCIAL_KEYWORDS = [
    "net sales", "total net sales", "revenue", "net revenue",
    "gross margin", "operating income", "net income",
    "earnings per share", "diluted earnings",
    "total assets", "total liabilities", "shareholders' equity",
    "cash flow", "free cash flow",
]


def _find_financial_sample(text: str, window: int = 3000) -> str:
    """
    Scan the document for the first occurrence of a financial keyword
    and return a window of text around it. Falls back to empty string
    if none found (caller will rely on the front-matter sample only).
    """
    lower = text.lower()
    earliest = len(text)
    for kw in _FINANCIAL_KEYWORDS:
        idx = lower.find(kw)
        if 0 < idx < earliest:
            earliest = idx

    if earliest == len(text):
        return ""

    start = max(0, earliest - 200)
    return text[start: start + window]


def extract_document_metadata(text: str) -> dict:
    """
    Extract structured financial metadata from document text.
    Uses two text samples:
      - front matter (first 3000 chars) for company name, ticker, dates
      - financial section snippet for key metrics
    Returns dict.
    """
    front = text[:3000]
    fin_sample = _find_financial_sample(text)
    combined = front
    if fin_sample and fin_sample not in front:
        combined = front + "\n\n--- Financial section ---\n\n" + fin_sample

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": _EXTRACTION_PROMPT.format(text=combined)}],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Metadata extraction failed: {e}")
        return {
            "company_name": None,
            "ticker": None,
            "doc_type": None,
            "fiscal_year": None,
            "period_end_date": None,
            "summary": "Could not extract summary.",
            "financials": {},
        }