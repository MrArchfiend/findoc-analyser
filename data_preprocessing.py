import re
from pypdf import PdfReader

def preprocess_data(uploaded_file):
    """
    Main entry point. Detects file type, extracts text,
    cleans it, and returns a single clean string.
    """
    if uploaded_file.name.endswith(".pdf"):
        text = _extract_pdf(uploaded_file)
    elif uploaded_file.name.endswith(".docx"):
        text = _extract_docx(uploaded_file)
    else:
        raise ValueError("Unsupported file type! Supported: .pdf, .docx")

    text = _clean_text(text)
    return text


def _extract_pdf(uploaded_file):
    """
    Tries normal text extraction first on every page using layout mode,
    which preserves word spacing from positional PDF data.
    If a page yields no text (scanned/image page),
    falls back to OCR for that specific page only.
    """
    # Read file bytes once — needed for both pypdf and pdf2image
    file_bytes = uploaded_file.read()

    # PdfReader must be initialised from bytes, not the already-read file object
    import io
    pdf_reader = PdfReader(io.BytesIO(file_bytes))
    total_pages = len(pdf_reader.pages)

    # Track which page numbers need OCR
    scanned_page_indices = []
    pages_text = {}

    # --- Pass 1: attempt layout-mode text extraction on all pages ---
    # extraction_mode="layout" uses character x-positions to reconstruct
    # word spacing, fixing the run-together-words problem from pypdf's
    # default plain extraction mode.
    for i, page in enumerate(pdf_reader.pages):
        try:
            extracted = page.extract_text(extraction_mode="layout")
        except TypeError:
            # Fallback for older pypdf versions that don't support the kwarg
            extracted = page.extract_text()

        if extracted and extracted.strip():
            pages_text[i] = extracted
        else:
            scanned_page_indices.append(i)

    # --- Pass 2: OCR only the pages that need it ---
    if scanned_page_indices:
        pages_text = _ocr_pages(file_bytes, scanned_page_indices, pages_text)

    # Reassemble in correct page order
    full_text = "\n".join(pages_text[i] for i in range(total_pages) if i in pages_text)
    return full_text


def _extract_docx(uploaded_file):
    """
    Extract text from a .docx file using python-docx.
    Preserves paragraph structure for the chunker.
    """
    try:
        from docx import Document
    except ImportError:
        raise ImportError(
            "python-docx is required for .docx files. "
            "Run: pip install python-docx"
        )
    doc = Document(uploaded_file)
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _ocr_pages(file_bytes, page_indices, pages_text):
    """
    Converts specific pages to images and runs Tesseract OCR on them.
    Only called if scanned pages are detected — avoids importing
    heavy libraries (pdf2image, pytesseract) unless actually needed.
    """
    try:
        import pytesseract
        from pdf2image import convert_from_bytes
    except ImportError:
        print(
            "WARNING: pytesseract or pdf2image not installed. "
            "Skipping OCR for scanned pages. "
            "Run: pip install pytesseract pdf2image"
        )
        return pages_text

    # Convert only the scanned pages to images (1-indexed for pdf2image)
    # first_page and last_page args don't support non-contiguous pages,
    # so we convert all and select by index
    print(f"Running OCR on {len(page_indices)} scanned page(s): {page_indices}")
    images = convert_from_bytes(file_bytes, dpi=300)  # 300 DPI = good OCR accuracy

    for i in page_indices:
        if i < len(images):
            ocr_text = pytesseract.image_to_string(images[i], lang='eng')
            if ocr_text.strip():
                pages_text[i] = ocr_text
            else:
                print(f"WARNING: OCR returned no text for page {i + 1} — page may be blank or image-only.")

    return pages_text


def _clean_text(text):
    """
    Cleans raw extracted/OCR text to remove noise that would
    hurt chunking quality and retrieval accuracy.
    """

    # Fix spaced-out single characters from PDF column layouts
    # e.g. "R e v e n u e" → "Revenue", "A d d r e s s" → "Address"
    # Requires 3+ consecutive single-letter tokens so normal words
    # like "One Apple Park Way" are never touched.
    text = re.sub(r'\b[A-Za-z](?: [A-Za-z]){2,}\b', lambda m: m.group(0).replace(' ', ''), text)

    # Fix camelCase-merged words where a space before a capital was lost,
    # e.g. "OneApple" → "One Apple", "ZipCode" → "Zip Code",
    # "principalexecutiveoffices" is all-lower so handled separately below.
    # Only splits at lowercase→uppercase boundaries to avoid breaking acronyms.
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    # Fix merged bracket/paren tokens, e.g. "95014(Address" → "95014 (Address"
    text = re.sub(r'(\w)([\(\[])', r'\1 \2', text)
    text = re.sub(r'([\)\]])(\w)', r'\1 \2', text)

    # Remove standalone page numbers (a line containing only digits)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

    # Remove common 10-K boilerplate that repeats on every page
    # These strings add noise to chunks without adding meaning
    boilerplate = [
        r'Table of Contents',
        r'UNITED STATES SECURITIES AND EXCHANGE COMMISSION',
        r'Form 10-K',
        r'Annual Report Pursuant to Section \d+',
    ]
    for pattern in boilerplate:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # Collapse multiple spaces/tabs into a single space
    text = re.sub(r'[ \t]+', ' ', text)

    # Collapse 3 or more newlines into 2 (preserve paragraph breaks, remove excess gaps)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()