import re
from pypdf import PdfReader

def preprocess_data(uploaded_file):
    """
    Main entry point. Detects file type, extracts text,
    cleans it, and returns a single clean string.
    """
    if uploaded_file.name.endswith(".pdf"):
        text = _extract_pdf(uploaded_file)
    else:
        raise ValueError("Unsupported file type!")

    text = _clean_text(text)
    return text


def _extract_pdf(uploaded_file):
    """
    Tries normal text extraction first on every page.
    If a page yields no text (scanned/image page),
    falls back to OCR for that specific page only.
    """
    # Read file bytes once — needed for both pypdf and pdf2image
    file_bytes = uploaded_file.read()

    pdf_reader = PdfReader(uploaded_file)
    total_pages = len(pdf_reader.pages)

    # Track which page numbers need OCR
    scanned_page_indices = []
    pages_text = {}

    # --- Pass 1: attempt normal text extraction on all pages ---
    for i, page in enumerate(pdf_reader.pages):
        extracted = page.extract_text()

        if extracted and extracted.strip():
            # Page has real text — store it
            pages_text[i] = extracted
        else:
            # Page is blank or image-based — flag for OCR
            scanned_page_indices.append(i)

    # --- Pass 2: OCR only the pages that need it ---
    if scanned_page_indices:
        pages_text = _ocr_pages(file_bytes, scanned_page_indices, pages_text)

    # Reassemble in correct page order
    full_text = "\n".join(pages_text[i] for i in range(total_pages) if i in pages_text)
    return full_text


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

    # Fix spaced-out characters from PDF column layouts
    # e.g. "R e v e n u e" → "Revenue"
    # Matches a letter, a single space, then another letter
    text = re.sub(r'(?<=[a-zA-Z])\s(?=[a-zA-Z])', '', text)

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
