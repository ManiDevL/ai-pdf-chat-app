
import html
import re
from io import BytesIO
from pathlib import Path

import streamlit as st


try:
    from pdf_processing import extract_text_from_pdf
except ImportError:
    extract_text_from_pdf = None

try:
    from pdf_processing import PDFProcessor
except ImportError:
    PDFProcessor = None

try:
    from rag_pipeline import answer_question_with_rag
except ImportError:
    answer_question_with_rag = None

try:
    from rag_pipeline import pipeline
except ImportError:
    pipeline = None

try:
    from rag_pipeline import GWDGRagPipeline
except ImportError:
    GWDGRagPipeline = None

try:
    from json_export import create_json_export, convert_json_export_to_string
except ImportError:
    create_json_export = None
    convert_json_export_to_string = None


# Sets the browser tab title, app icon, and page layout for the Streamlit UI.
st.set_page_config(
    page_title="AI PDF Chat App",
    page_icon="💬",
    layout="wide",
)


# Defines the visual design of the app, including the dark background, cards, sidebar, and chat styling.
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #111827 45%, #064e3b 100%);
    }

    .block-container {
        max-width: 1100px;
        padding-top: 4.5rem;
        padding-bottom: 6rem;
    }

    .main-header {
        padding: 1.2rem 1.4rem;
        border-radius: 1.2rem;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 18px 45px rgba(0, 0, 0, 0.25);
        margin-top: 0.5rem;
        margin-bottom: 1.2rem;
    }

    .main-header h1 {
        margin: 0 0 0.35rem 0;
        color: #f8fafc;
        font-size: 2.6rem;
        line-height: 1.1;
    }

    .main-header p {
        margin: 0;
        color: #cbd5e1;
        font-size: 1rem;
    }

    .chat-info-box {
        padding: 0.9rem 1.1rem;
        border-radius: 1rem;
        background: rgba(255, 255, 255, 0.10);
        border: 1px solid rgba(255, 255, 255, 0.14);
        margin-bottom: 1rem;
        color: #f8fafc;
    }

    .status-card {
        padding: 0.9rem 1rem;
        border-radius: 1rem;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.30);
        margin: 0.7rem 0;
    }

    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.55rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        background: rgba(16, 185, 129, 0.22);
        color: #a7f3d0;
        border: 1px solid rgba(167, 243, 208, 0.35);
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
    }

    .chat-row {
        display: flex;
        align-items: flex-end;
        width: 100%;
        margin: 0.45rem 0;
        gap: 0.55rem;
    }

    .chat-row-user {
        justify-content: flex-end;
    }

    .chat-row-assistant {
        justify-content: flex-start;
    }

    .chat-bubble {
        max-width: 72%;
        padding: 0.65rem 0.85rem;
        border-radius: 1rem;
        line-height: 1.4;
        font-size: 0.92rem;
        box-shadow: 0 8px 22px rgba(0, 0, 0, 0.20);
        overflow-wrap: break-word;
        white-space: normal;
    }

    .chat-bubble-user {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: #ecfdf5;
        border-bottom-right-radius: 0.25rem;
        border: 1px solid rgba(167, 243, 208, 0.35);
    }

    .chat-bubble-assistant {
        background: rgba(255, 255, 255, 0.10);
        color: #f8fafc;
        border-bottom-left-radius: 0.25rem;
        border: 1px solid rgba(255, 255, 255, 0.14);
    }

    .chat-avatar {
        width: 2rem;
        height: 2rem;
        min-width: 2rem;
        border-radius: 999px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        background: rgba(255, 255, 255, 0.11);
        border: 1px solid rgba(255, 255, 255, 0.16);
        box-shadow: 0 8px 18px rgba(0, 0, 0, 0.18);
    }

    .source-pill {
        min-width: fit-content;
        padding: 0.32rem 0.55rem;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 700;
        color: #a7f3d0;
        background: rgba(16, 185, 129, 0.14);
        border: 1px solid rgba(167, 243, 208, 0.28);
        box-shadow: 0 8px 18px rgba(0, 0, 0, 0.14);
    }

    section[data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.92);
        border-right: 1px solid rgba(255, 255, 255, 0.10);
    }

    div[data-testid="stFileUploader"] {
        padding: 0.7rem;
        border-radius: 1rem;
        background: rgba(255, 255, 255, 0.06);
        border: 1px dashed rgba(167, 243, 208, 0.45);
    }

    div[data-testid="stChatInput"] {
        border-radius: 1rem;
    }

    /* Stable bubble styling for Streamlit's built-in chat component.
       We only style Streamlit elements here; we do not store custom HTML in chat history. */
    div[data-testid="stChatMessage"] {
        width: fit-content;
        max-width: 76%;
        background: rgba(255, 255, 255, 0.10);
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 1rem;
        padding: 0.75rem 0.95rem;
        margin: 0.7rem 0;
        box-shadow: 0 8px 22px rgba(0, 0, 0, 0.18);
    }

    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        margin-left: auto;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        border: 1px solid rgba(167, 243, 208, 0.35);
    }

    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        margin-right: auto;
        background: rgba(255, 255, 255, 0.10);
        border: 1px solid rgba(255, 255, 255, 0.14);
    }

    div[data-testid="stChatMessage"] p {
        color: #f8fafc;
        font-size: 0.95rem;
        line-height: 1.45;
    }

    div[data-testid="stCaptionContainer"] p {
        color: #a7f3d0;
        font-size: 0.78rem;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Initializes Streamlit session variables so chat history, uploaded PDF data, and extracted page texts persist across reruns.
if "messages" not in st.session_state:
    st.session_state.messages = []

if "extracted_text" not in st.session_state:
    st.session_state.extracted_text = ""

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

if "json_export" not in st.session_state:
    st.session_state.json_export = None


if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

if "page_texts" not in st.session_state:
    st.session_state.page_texts = {}





# Cleans assistant and user messages before rendering, removing old HTML fragments and unsafe markup.
def clean_message_content(content: str) -> str:
    content = html.unescape(str(content)).strip()

    # Handle text that contains escaped HTML from earlier render attempts.
    content = content.replace("```html", "").replace("```", "")
    content = content.replace('\\"', '"').replace("\\'", "'")

    # If an older message accidentally contains a rendered chat bubble as raw HTML,
    # extract only the text inside that bubble and drop the wrapper.
    if "chat-bubble" in content:
        bubble_match = re.search(
            r"<div[^>]*chat-bubble[^>]*>(.*?)</div>",
            content,
            flags=re.DOTALL,
        )
        if bubble_match:
            content = bubble_match.group(1)

    # Final hard fallback: remove any remaining chat-bubble wrapper fragments.
    content = re.sub(r".*?<div[^>]*chat-bubble[^>]*>", "", content, flags=re.DOTALL)
    content = re.sub(r"</div>.*", "", content, flags=re.DOTALL)

    # Remove remaining UI-only HTML fragments if they still exist.
    content = re.sub(r"<div[^>]*chat-row[^>]*>", "", content, flags=re.DOTALL)
    content = re.sub(r"<div[^>]*chat-bubble[^>]*>", "", content, flags=re.DOTALL)
    content = re.sub(r"<div[^>]*chat-avatar[^>]*>.*?</div>", "", content, flags=re.DOTALL)
    content = re.sub(r"<div[^>]*source-pill[^>]*>.*?</div>", "", content, flags=re.DOTALL)
    content = re.sub(r"</div>", "", content)
    content = re.sub(r"<br\s*/?>", "\n", content)
    content = re.sub(r"<[^>]+>", "", content)

    return content.strip()


#
# Reads page numbers from RAG source citations so the app can display source references.
def extract_page_numbers_from_answer(content: str) -> list[int]:
    """
    Extract page numbers from source citations in an answer.

    Example:
    (Source: report.pdf, Pages 2, 4, 8) -> [2, 4, 8]
    """
    content = clean_message_content(content)
    source_matches = re.findall(r"\(Source:[^)]*\)", content)

    page_numbers = []
    for source_match in source_matches:
        page_sections = re.findall(r"Pages?\s+([0-9,\sand]+)", source_match, flags=re.IGNORECASE)
        for page_section in page_sections:
            page_numbers.extend(re.findall(r"\d+", page_section))

    unique_pages = []
    for page_number in page_numbers:
        page_number_int = int(page_number)
        if page_number_int not in unique_pages:
            unique_pages.append(page_number_int)

    return unique_pages


#
# Separates the actual assistant answer from the source page labels shown above the answer.
def split_answer_and_sources(content: str) -> tuple[str, str]:
    """
    Remove source citations from the assistant answer and return them separately.

    Example input:
    Answer text (Source: Apple_Report.pdf, Page 2, 4, 8).

    Example output:
    - Answer text
    - Pages: 2, 4, 8
    """
    content = clean_message_content(content)
    page_numbers = extract_page_numbers_from_answer(content)
    cleaned_content = re.sub(r"\s*\(Source:[^)]*\)", "", content).strip()

    if not page_numbers:
        return cleaned_content, ""

    if len(page_numbers) == 1:
        return cleaned_content, f"Page {page_numbers[0]}"

    return cleaned_content, f"Pages {', '.join(str(page_number) for page_number in page_numbers)}"


#
# Renders each chat message and adds expandable source-page previews for assistant answers.
def render_chat_bubble(role: str, content: str) -> None:
    """
    Render chat messages with Streamlit's stable built-in chat component.

    This avoids storing or rendering custom HTML as message content.
    """
    source_label = ""
    content = clean_message_content(content)
    source_pages = []

    if role == "assistant":
        source_pages = extract_page_numbers_from_answer(content)
        content, source_label = split_answer_and_sources(content)
        avatar = "🌱"
    else:
        avatar = "👤"

    with st.chat_message(role, avatar=avatar):
        if source_label:
            st.caption(f"📄 {source_label}")
        st.write(content)

        if role == "assistant" and source_pages and st.session_state.page_texts:
            with st.expander("Show source pages"):
                for page_number in source_pages:
                    page_text = st.session_state.page_texts.get(page_number, "")
                    if page_text:
                        with st.expander(f"Page {page_number}"):
                            st.write(page_text)
                    else:
                        st.caption(f"Page {page_number}: No extracted text available.")

#
# Extracts text page by page from the uploaded PDF for the source-page preview feature.
def extract_page_texts_from_pdf(uploaded_file) -> dict[int, str]:
    """
    Extract readable text per PDF page for source preview expanders.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            return {}

    try:
        uploaded_file.seek(0)
        reader = PdfReader(BytesIO(uploaded_file.getvalue()))

        page_texts = {}
        for page_index, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            page_texts[page_index] = page_text.strip()

        uploaded_file.seek(0)
        return page_texts
    except Exception:
        uploaded_file.seek(0)
        return {}


#
# Saves the uploaded PDF file locally because the backend PDF processor expects files in the pdfs folder.
def save_uploaded_pdf(uploaded_file) -> Path:
    """
    Save the Streamlit upload into the local pdfs folder.

    The new PDFProcessor from the updated backend expects real files in ./pdfs,
    so the uploaded file has to be written to disk first.
    """
    pdf_folder = Path("pdfs")
    pdf_folder.mkdir(exist_ok=True)

    pdf_path = pdf_folder / uploaded_file.name
    pdf_path.write_bytes(uploaded_file.getbuffer())
    return pdf_path


#
# Processes a new PDF upload by extracting page text and preparing the RAG index.
def process_uploaded_pdf(uploaded_file) -> str:
    """
    Process the uploaded PDF with whichever backend version is currently available.

    Compatibility paths:
    1. Old backend: extract_text_from_pdf(uploaded_file)
    2. New backend: save PDF to ./pdfs and call PDFProcessor().process_all_pdfs()
    """
    st.session_state.page_texts = extract_page_texts_from_pdf(uploaded_file)
    if extract_text_from_pdf is not None:
        return extract_text_from_pdf(uploaded_file)

    save_uploaded_pdf(uploaded_file)

    if PDFProcessor is not None:
        processor = PDFProcessor()
        if hasattr(processor, "process_all_pdfs"):
            processor.process_all_pdfs()

    return "PDF was processed and indexed in the vector database."


#
# Sends the user question to the available RAG pipeline and returns the generated answer.
def get_rag_answer(question: str) -> str:
    """
    Generate an answer with whichever RAG interface is currently available.

    Compatibility paths:
    1. Old backend: answer_question_with_rag(question, document_text)
    2. New backend: pipeline.ask(question)
    3. New backend alternative: GWDGRagPipeline().ask(question)
    """
    if answer_question_with_rag is not None:
        return answer_question_with_rag(question, st.session_state.extracted_text)

    active_pipeline = pipeline

    if active_pipeline is None and GWDGRagPipeline is not None:
        active_pipeline = GWDGRagPipeline()

    if active_pipeline is not None:
        if hasattr(active_pipeline, "ask"):
            return active_pipeline.ask(question)
        if hasattr(active_pipeline, "answer_question_with_rag"):
            return active_pipeline.answer_question_with_rag(question)

    return "Error: No compatible RAG pipeline function was found. Please check rag_pipeline.py."


# Displays the main app header shown above the chat area.
st.markdown(
    """
    <div class='main-header'>
        <h1>🌱 AI Sustainability PDF Chat</h1>
        <p>A RAG-powered chat interface for analyzing sustainability reports.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


#
# Builds the sidebar control panel for PDF upload, processing status, chat reset, and JSON export.
with st.sidebar:
    st.header("📄 Control Panel")
    st.caption("Upload a report and chat with its content.")

    uploaded_file = st.file_uploader(
        "Upload a sustainability report as PDF",
        type=["pdf"],
    )

    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.uploaded_file_name:
            with st.spinner("Processing PDF and preparing RAG index..."):
                st.session_state.extracted_text = process_uploaded_pdf(uploaded_file)

            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.pdf_processed = True
            st.session_state.messages = []
            st.session_state.json_export = None

        st.success("PDF uploaded successfully.")
        st.markdown(
            f"""
            <div class='status-card'>
                <span class='status-badge'>PDF READY</span>
                <span class='status-badge'>RAG ACTIVE</span>
                <span class='status-badge'>GEMMA 4</span>
                <br><br>
                <strong>File:</strong> {uploaded_file.name}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.session_state.extracted_text:
            with st.expander("PDF processing preview"):
                st.text_area(
                    "Preview / status",
                    st.session_state.extracted_text[:3000],
                    height=250,
                )
        else:
            st.warning("No text or index status could be created from this PDF.")

    if st.button("Clear chat"):
        st.session_state.messages = []

    st.divider()
    st.subheader("📦 JSON Export")

    if create_json_export is None or convert_json_export_to_string is None:
        st.caption("JSON export is currently disabled because json_export.py is not compatible with the latest RAG pipeline.")
    elif st.session_state.extracted_text:
        if st.button("Generate sustainability JSON"):
            with st.spinner("Extracting structured sustainability data..."):
                st.session_state.json_export = create_json_export(
                    st.session_state.extracted_text
                )

        if st.session_state.json_export is not None:
            json_export_string = convert_json_export_to_string(
                st.session_state.json_export
            )

            st.download_button(
                label="Download JSON",
                data=json_export_string,
                file_name="sustainability_export.json",
                mime="application/json",
            )

            with st.expander("Preview JSON"):
                st.json(st.session_state.json_export)
    else:
        st.caption("Upload a PDF first to enable JSON export.")

    st.divider()
    st.caption("System status")
    st.markdown("<span class='status-badge'>Streamlit UI</span>", unsafe_allow_html=True)
    st.markdown("<span class='status-badge'>PDF Processing</span>", unsafe_allow_html=True)
    st.markdown("<span class='status-badge'>RAG Pipeline</span>", unsafe_allow_html=True)


#
# Shows a status box that tells the user whether a PDF has already been loaded.
if not st.session_state.extracted_text:
    st.markdown(
        """
        <div class='chat-info-box'>
        <strong>No PDF uploaded yet.</strong><br>
        Upload a sustainability report in the sidebar to start the chat.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class='chat-info-box'>
        <strong>Document is ready.</strong><br>
        Ask a question below. The app retrieves relevant sections and sends them to SAIA/Gemma.
        </div>
        """,
        unsafe_allow_html=True,
    )


#
# Re-renders the stored chat history after every Streamlit rerun.
for message in st.session_state.messages:
    render_chat_bubble(message["role"], message["content"])


#
# Creates the chat input field where the user asks questions about the uploaded PDF.
user_question = st.chat_input("Ask a question about the PDF...")

#
# Handles a new user question: store it, query the RAG pipeline, render the answer, and save it to chat history.
if user_question:
    if not st.session_state.extracted_text:
        st.warning("Please upload a PDF first.")
    else:
        st.session_state.messages.append(
            {"role": "user", "content": user_question}
        )

        render_chat_bubble("user", user_question)

        with st.spinner("Searching relevant PDF sections and generating answer with SAIA/Gemma..."):
            raw_answer = get_rag_answer(user_question)

        answer = clean_message_content(raw_answer)

        render_chat_bubble("assistant", answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
