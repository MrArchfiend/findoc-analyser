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

# function _extract_pdf
    """
    define a function _extract_pdf that 
    tries normal text extraction first on every page.
    If a page yields no text (scanned/image page),
    falls back to OCR for that specific page only.
    """
  
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
