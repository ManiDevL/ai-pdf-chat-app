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


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """
    Split the extracted PDF text into smaller overlapping text chunks.

    This function is intentionally placed in pdf_processing.py because it prepares
    the raw PDF text before the RAG pipeline retrieves relevant sections.

    Team task:
    Your teammate can improve this function later.
    Possible improvements:
    - sentence-based chunking
    - paragraph-based chunking
    - page-based chunking
    - hierarchical chunking
    """
    if not text:
        return []

    chunks = []
    start = 0

    # Current simple prototype logic:
    # Example with chunk_size=1000 and chunk_overlap=200:
    # Chunk 1 = characters 0-1000
    # Chunk 2 = characters 800-1800
    # Chunk 3 = characters 1600-2600
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - chunk_overlap

    return chunks
