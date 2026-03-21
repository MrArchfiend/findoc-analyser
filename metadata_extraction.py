"""
metadata_extraction.py
Two-phase extraction:
  Phase 1 — LLM call on front matter only for identity fields
             (company name, ticker, doc type, dates, summary).
  Phase 2 — Regex scanner walks every chunk to pull raw numeric values,
             then computes derived metrics with explicit formulas.
             No LLM involved in financials — numbers come from the document.
"""

import json
import re
from typing import Optional
from groq import Groq
from constants import GROQ_API_KEY, GROQ_MODEL

client = Groq(api_key=GROQ_API_KEY)


# ── Phase 1: identity extraction ─────────────────────────────────────────────

_IDENTITY_PROMPT = """
You are a financial document parser. Given the first page of a 10-K or SEC filing,
extract only the identity fields listed below.

Return ONLY a valid JSON object with this exact schema (use null for missing fields):
{{
  "company_name": "string",
  "ticker": "string or null",
  "doc_type": "10-K or 10-Q or 8-K or other",
  "fiscal_year": "YYYY or null",
  "period_end_date": "YYYY-MM-DD or null",
  "summary": "2-3 sentence plain English summary of the company and what this filing covers"
}}

Document text:
{text}
"""


def _extract_identity(front_text: str) -> dict:
    """LLM call on front matter only — fast, cheap, focused."""
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": _IDENTITY_PROMPT.format(text=front_text[:3500])}],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Identity extraction failed: {e}")
        return {
            "company_name": None,
            "ticker": None,
            "doc_type": None,
            "fiscal_year": None,
            "period_end_date": None,
            "summary": "Could not extract summary.",
        }


# ── Phase 2: numeric extraction from chunks ───────────────────────────────────

# Each entry: (field_name, list_of_regex_patterns)
# Patterns are tried in order; first match wins.
# All patterns expect a capturing group 1 for the raw numeric string.
_FIELD_PATTERNS: list[tuple[str, list[str]]] = [
    ("revenue_raw", [
        r"(?:total\s+)?net\s+(?:sales|revenue)[^\d$]{0,30}\$?\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion|M|B)?",
        r"(?:total\s+)?revenue[^\d$]{0,30}\$?\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion|M|B)?",
    ]),
    ("gross_profit_raw", [
        r"gross\s+(?:profit|margin)[^\d$]{0,30}\$?\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion|M|B)?",
    ]),
    ("operating_income_raw", [
        r"(?:total\s+)?operating\s+(?:income|profit)[^\d$]{0,30}\$?\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion|M|B)?",
    ]),
    ("net_income_raw", [
        r"net\s+(?:income|earnings|profit)[^\d$]{0,30}\$?\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion|M|B)?",
    ]),
    ("ebitda_raw", [
        r"ebitda[^\d$]{0,30}\$?\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion|M|B)?",
    ]),
    ("eps_basic_raw", [
        r"(?:basic\s+)?(?:net\s+)?earnings\s+per\s+(?:common\s+)?share[^\d$]{0,40}\$?\s*([\d,]+(?:\.\d+)?)",
        r"eps[,\s\-]+basic[^\d$]{0,20}\$?\s*([\d,]+(?:\.\d+)?)",
    ]),
    ("eps_diluted_raw", [
        r"diluted[^\d$]{0,30}earnings\s+per\s+(?:common\s+)?share[^\d$]{0,30}\$?\s*([\d,]+(?:\.\d+)?)",
        r"eps[,\s\-]+diluted[^\d$]{0,20}\$?\s*([\d,]+(?:\.\d+)?)",
    ]),
    ("total_assets_raw", [
        r"total\s+assets[^\d$]{0,30}\$?\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion|M|B)?",
    ]),
    ("total_liabilities_raw", [
        r"total\s+(?:liabilities|debt)[^\d$]{0,30}\$?\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion|M|B)?",
    ]),
    ("shareholders_equity_raw", [
        r"(?:total\s+)?(?:shareholders?|stockholders?)['\s]+equity[^\d$]{0,30}\$?\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion|M|B)?",
    ]),
    ("current_assets_raw", [
        r"total\s+current\s+assets[^\d$]{0,30}\$?\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion|M|B)?",
    ]),
    ("current_liabilities_raw", [
        r"total\s+current\s+liabilities[^\d$]{0,30}\$?\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion|M|B)?",
    ]),
    ("operating_cash_flow_raw", [
        r"(?:net\s+)?cash\s+(?:provided\s+by|from)\s+operating[^\d$]{0,30}\$?\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion|M|B)?",
    ]),
    ("capex_raw", [
        r"capital\s+expenditures?[^\d$]{0,30}\$?\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion|M|B)?",
        r"purchases?\s+of\s+(?:property|PP&E)[^\d$]{0,30}\$?\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion|M|B)?",
    ]),
    ("interest_expense_raw", [
        r"interest\s+expense[^\d$]{0,30}\$?\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion|M|B)?",
    ]),
    ("tax_expense_raw", [
        r"(?:provision\s+for\s+)?income\s+tax(?:es)?[^\d$]{0,30}\$?\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion|M|B)?",
    ]),
    ("depreciation_raw", [
        r"depreciation(?:\s+and\s+amortization)?[^\d$]{0,30}\$?\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion|M|B)?",
    ]),
]

# Scale multipliers inferred from surrounding text
_SCALE_PATTERNS = [
    (re.compile(r'\bin\s+billions\b', re.I), 1_000),
    (re.compile(r'\bin\s+millions\b', re.I), 1),
    (re.compile(r'\bin\s+thousands\b', re.I), 0.001),
]


def _infer_scale(text: str) -> float:
    """Return multiplier to convert extracted values to $millions. Default 1."""
    for pattern, mult in _SCALE_PATTERNS:
        if pattern.search(text):
            return mult
    return 1.0


def _parse_number(raw: str) -> Optional[float]:
    """Strip commas and convert to float. Returns None on failure."""
    try:
        return float(raw.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def _scan_chunks(chunks) -> dict[str, float]:
    """
    Walk every chunk and apply _FIELD_PATTERNS.
    Returns a dict of field_name -> best numeric value found (in $M).
    Takes the first confident match per field across all chunks.
    """
    raw_values: dict[str, float] = {}
    scale = 1.0

    # First pass: infer scale from any chunk
    for chunk in chunks:
        s = _infer_scale(chunk.text)
        if s != 1.0:
            scale = s
            break

    # Second pass: extract values
    for chunk in chunks:
        text = chunk.text
        for field, patterns in _FIELD_PATTERNS:
            if field in raw_values:
                continue  # already found
            for pat in patterns:
                m = re.search(pat, text, re.I)
                if m:
                    val = _parse_number(m.group(1))
                    if val is not None and val > 0:
                        # Apply inline unit if present in surrounding context
                        context = text[max(0, m.start() - 5):m.end() + 10].lower()
                        if "billion" in context or context.strip().endswith("b"):
                            val *= 1_000
                        raw_values[field] = val * scale
                        break

    return raw_values


def _fmt_millions(v: Optional[float]) -> Optional[str]:
    """Format a $M value into a human-readable string."""
    if v is None:
        return None
    if v >= 1_000:
        return f"${v / 1_000:.2f}B"
    return f"${v:.0f}M"


def _fmt_pct(v: Optional[float]) -> Optional[str]:
    if v is None:
        return None
    return f"{v:.1f}%"


def _fmt_ratio(v: Optional[float]) -> Optional[str]:
    if v is None:
        return None
    return f"{v:.2f}x"


def _compute_financials(raw: dict[str, float]) -> dict:
    """
    Compute all financial metrics.
    Raw extracted values -> formula-derived metrics wherever possible.
    Falls back to None if required inputs are missing.
    """
    def get(k): return raw.get(k)

    rev    = get("revenue_raw")
    gp     = get("gross_profit_raw")
    oi     = get("operating_income_raw")
    ni     = get("net_income_raw")
    ebitda = get("ebitda_raw")
    ta     = get("total_assets_raw")
    tl     = get("total_liabilities_raw")
    se     = get("shareholders_equity_raw")
    ca     = get("current_assets_raw")
    cl     = get("current_liabilities_raw")
    ocf    = get("operating_cash_flow_raw")
    capex  = get("capex_raw")
    dep    = get("depreciation_raw")
    ie     = get("interest_expense_raw")
    tax    = get("tax_expense_raw")

    # Formula-derived metrics
    gross_margin     = (gp / rev * 100)  if (gp  and rev) else None
    operating_margin = (oi / rev * 100)  if (oi  and rev) else None
    net_margin       = (ni / rev * 100)  if (ni  and rev) else None
    debt_to_equity   = (tl / se)         if (tl  and se)  else None
    current_ratio    = (ca / cl)         if (ca  and cl)  else None
    roe              = (ni / se * 100)   if (ni  and se)  else None
    roa              = (ni / ta * 100)   if (ni  and ta)  else None
    free_cash_flow   = (ocf - capex)     if (ocf and capex) else None

    # EBITDA: use extracted value, or build from components
    if not ebitda and oi and dep:
        ebitda = oi + dep
    if not ebitda and ni and tax and ie and dep:
        ebitda = ni + tax + ie + dep

    return {
        "revenue":             _fmt_millions(rev),
        "net_income":          _fmt_millions(ni),
        "gross_profit":        _fmt_millions(gp),
        "operating_income":    _fmt_millions(oi),
        "ebitda":              _fmt_millions(ebitda),
        "eps_basic":           f"${raw['eps_basic_raw']:.2f}"   if get("eps_basic_raw")   else None,
        "eps_diluted":         f"${raw['eps_diluted_raw']:.2f}" if get("eps_diluted_raw") else None,
        "total_assets":        _fmt_millions(ta),
        "total_liabilities":   _fmt_millions(tl),
        "shareholders_equity": _fmt_millions(se),
        "current_assets":      _fmt_millions(ca),
        "current_liabilities": _fmt_millions(cl),
        "operating_cash_flow": _fmt_millions(ocf),
        "capital_expenditures": _fmt_millions(capex),
        "free_cash_flow":      _fmt_millions(free_cash_flow),
        "gross_margin":        _fmt_pct(gross_margin),
        "operating_margin":    _fmt_pct(operating_margin),
        "net_margin":          _fmt_pct(net_margin),
        "debt_to_equity":      _fmt_ratio(debt_to_equity),
        "current_ratio":       _fmt_ratio(current_ratio),
        "return_on_equity":    _fmt_pct(roe),
        "return_on_assets":    _fmt_pct(roa),
    }


# ── Public API ────────────────────────────────────────────────────────────────

def extract_document_metadata(text: str, chunks=None) -> dict:
    """
    Full metadata extraction.
      - Identity fields: LLM on front matter only.
      - Financials: regex scan of all chunks + formula computation.
        Falls back to scanning full text if no chunks provided.
    """
    identity = _extract_identity(text)

    if chunks:
        raw = _scan_chunks(chunks)
    else:
        class _PseudoChunk:
            def __init__(self, t): self.text = t
        raw = _scan_chunks([_PseudoChunk(text)])

    financials = _compute_financials(raw)

    return {
        **identity,
        "financials": financials,
    }