import json
from groq import Groq
from constants import GROQ_API_KEY, GROQ_MODEL

client = Groq(api_key=GROQ_API_KEY)

_EXTRACTION_PROMPT = """
You are a financial data extraction expert. Given the first ~4000 characters of a 10-K or SEC filing, extract structured metadata.

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


def extract_document_metadata(text: str) -> dict:
    """Extract structured financial metadata from document text. Returns dict."""
    sample = text[:4000]
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": _EXTRACTION_PROMPT.format(text=sample)}],
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
