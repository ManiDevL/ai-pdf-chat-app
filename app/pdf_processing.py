from pypdf import PdfReader


def extract_text_from_pdf(pdf_file) -> str:
    """
    Extract text from an uploaded PDF file.

    Streamlit provides the uploaded file as a file-like object.
    PdfReader can read this object directly.
    """
    reader = PdfReader(pdf_file)
    extracted_text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            extracted_text += page_text + "\n"

    return extracted_text.strip()
