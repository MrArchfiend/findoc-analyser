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
  
# function _ocr_pages    
    """
    define a function _ocr_pages that converts specific pages to images and runs Tesseract OCR on them.
    Only called if scanned pages are detected — avoids importing
    heavy libraries (pdf2image, pytesseract) unless actually needed.
    """

# function _clean_text 
    """
    define a function _clean_text that cleans raw extracted/OCR text to remove noise that would
    hurt chunking quality and retrieval accuracy.
    """
