
import html
import re
from io import BytesIO
from pathlib import Path

import streamlit as st


from json_export import create_json_export, convert_json_export_to_string
from pdf_processing import PDF_DIR, PDFProcessor
from rag_pipeline import get_pipeline, is_summary_question


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

if "processing_status" not in st.session_state:
    st.session_state.processing_status = ""

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

if "json_export" not in st.session_state:
    st.session_state.json_export = None


if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

if "page_texts" not in st.session_state:
    st.session_state.page_texts = {}

if "active_source" not in st.session_state:
    st.session_state.active_source = None

if "just_deleted_source" not in st.session_state:
    st.session_state.just_deleted_source = None


# Die RAG-Pipeline (LLM-Client, Vektor-DB, Embedding- und Reranker-Modelle) wird
# einmal pro Prozess geladen und über alle Streamlit-Reruns hinweg wiederverwendet.
@st.cache_resource(show_spinner="Loading models and vector database...")
def load_pipeline():
    return get_pipeline()


try:
    pipeline = load_pipeline()
    pipeline_error = None
except Exception as e:
    pipeline = None
    pipeline_error = str(e)





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
    source_matches = re.findall(r"\((?:Source|Quelle):[^)]*\)", content)

    page_numbers = []
    for source_match in source_matches:
        page_sections = re.findall(r"(?:Pages?|Seiten?):?\s+([0-9,\sand]+)", source_match, flags=re.IGNORECASE)
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
    cleaned_content = re.sub(r"\s*\((?:Source|Quelle):[^)]*\)", "", content).strip()

    if not page_numbers:
        return cleaned_content, ""

    if len(page_numbers) == 1:
        return cleaned_content, f"Page {page_numbers[0]}"

    return cleaned_content, f"Pages {', '.join(str(page_number) for page_number in page_numbers)}"


#
# Renders each chat message and adds expandable source previews for assistant answers.
def render_chat_bubble(role: str, content: str, sources: list | None = None) -> None:
    """
    Render chat messages with Streamlit's stable built-in chat component.

    Assistant messages carry structured source metadata from the RAG pipeline
    (document, page, relevance, optional figure image). Older messages without
    metadata fall back to parsing the citation text.
    """
    content = clean_message_content(content)

    if role != "assistant":
        with st.chat_message(role, avatar="👤"):
            st.write(content)
        return

    # Seitenangaben bevorzugt aus den echten Retrieval-Metadaten statt aus dem Antworttext
    meta_pages = []
    for entry in sources or []:
        page = entry.get("page")
        if page and page not in meta_pages:
            meta_pages.append(page)
    meta_pages.sort()

    display_content, regex_label = split_answer_and_sources(content)
    if meta_pages:
        if len(meta_pages) == 1:
            source_label = f"Page {meta_pages[0]}"
        else:
            source_label = "Pages " + ", ".join(str(page) for page in meta_pages)
    else:
        source_label = regex_label

    with st.chat_message("assistant", avatar="🌱"):
        if source_label:
            st.caption(f"📄 {source_label}")
        st.write(display_content)

        if sources:
            with st.expander("Show sources"):
                for entry in sources:
                    if entry.get("type") == "summary":
                        st.caption(f"📋 Document summary: {entry.get('source')}")
                    else:
                        page = entry.get("page")
                        page_end = entry.get("page_end") or page
                        page_label = f"{page}-{page_end}" if page_end != page else f"{page}"
                        icon = "🖼️" if entry.get("type") == "image" else "📄"
                        score = entry.get("score")
                        score_label = (
                            f" · relevance {score:.0%}" if isinstance(score, (int, float)) else ""
                        )
                        st.caption(f"{icon} {entry.get('source')} – Page {page_label}{score_label}")

                    image_path = entry.get("image_path")
                    if image_path and Path(image_path).exists():
                        st.image(image_path)

                    entry_text = entry.get("text", "")
                    if entry_text:
                        st.write(entry_text[:600] + ("…" if len(entry_text) > 600 else ""))
                    st.divider()
        else:
            # Fallback für ältere Nachrichten ohne gespeicherte Quellen-Metadaten
            source_pages = extract_page_numbers_from_answer(content)
            if source_pages and st.session_state.page_texts:
                with st.expander("Show source pages"):
                    for page_number in source_pages:
                        page_text = st.session_state.page_texts.get(page_number, "")
                        st.caption(f"Page {page_number}")
                        if page_text:
                            st.write(page_text[:1500])
                        else:
                            st.caption("No extracted text available.")

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
    PDF_DIR.mkdir(exist_ok=True)

    pdf_path = PDF_DIR / uploaded_file.name
    pdf_path.write_bytes(uploaded_file.getbuffer())
    return pdf_path


#
# Processes a new PDF upload by extracting page text and preparing the RAG index.
def process_uploaded_pdf(uploaded_file) -> str:
    """
    Process ONLY the newly uploaded PDF: chunk and index its text.

    This is deliberately the ONLY blocking step. Chunking + embedding is fast
    (CPU-only, no LLM calls) and is enough to make the document immediately
    chattable. Two more expensive steps are deferred:
    - The document summary is built lazily, only when a summary question is
      actually asked (see rag_pipeline.answer).
    - Figure descriptions are generated in a background thread (see
      pipeline.ensure_document_images), since each image needs its own
      vision-model call and a chart-heavy report can have dozens of them.
    Older PDFs stay untouched in the index; they are not re-chunked on every upload.
    """
    st.session_state.page_texts = extract_page_texts_from_pdf(uploaded_file)
    save_uploaded_pdf(uploaded_file)

    stats = PDFProcessor().process_pdf(uploaded_file.name)
    pipeline.ensure_document_images(uploaded_file.name)

    return (
        f"'{uploaded_file.name}' was processed: {stats['chunks']} text chunks from "
        f"{stats['pages']} pages indexed and ready to chat. Figures are being described "
        "in the background and will become searchable within a few minutes."
    )


#
# Lists the documents that are currently indexed in the vector database.
def get_indexed_sources() -> list[str]:
    if pipeline is None:
        return []
    return pipeline.list_sources()


#
# Sends the user question to the RAG pipeline and returns answer + sources.
def get_rag_result(question: str) -> dict:
    """
    Returns {"answer": str | generator, "sources": list[dict]}. The question is
    scoped to the document selected in the sidebar (st.session_state.active_source),
    and the recent chat history is passed along for follow-up questions.
    """
    # Die letzten Chat-Nachrichten (ohne die gerade gestellte Frage) als Kontext
    history = [
        {"role": message["role"], "content": message["content"]}
        for message in st.session_state.messages[:-1]
    ]
    return pipeline.answer(
        question,
        source=st.session_state.active_source,
        history=history,
        stream=True,
    )


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
# Without a working pipeline (e.g. missing .env values) the app cannot do anything;
# show the real error instead of failing silently later.
if pipeline is None:
    st.error(
        f"The RAG pipeline could not be initialized: {pipeline_error}\n\n"
        "Check the .env file (SAIA_API_KEY, SAIA_BASE_URL, SAIA_MODEL) and restart the app."
    )
    st.stop()


#
# The list of documents currently in the vector database drives upload skipping,
# the document selector, and the chat gate.
indexed_sources = get_indexed_sources()

#
# Builds the sidebar control panel for PDF upload, document management, chat reset, and JSON export.
with st.sidebar:
    st.header("📄 Control Panel")
    st.caption("Upload a report and chat with its content.")

    uploaded_file = st.file_uploader(
        "Upload a sustainability report as PDF",
        type=["pdf"],
    )

    if uploaded_file is None:
        # Once the deleted file is removed from the uploader, it may be re-uploaded again.
        st.session_state.just_deleted_source = None

    if uploaded_file is not None:
        if uploaded_file.name == st.session_state.just_deleted_source:
            st.info(
                "This document was just deleted. Remove the file from the upload "
                "field first if you want to re-index it."
            )
        elif uploaded_file.name not in indexed_sources:
            with st.spinner("Processing PDF: chunking and indexing text..."):
                st.session_state.processing_status = process_uploaded_pdf(uploaded_file)

            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.pdf_processed = True
            st.session_state.messages = []
            st.session_state.json_export = None
            indexed_sources = get_indexed_sources()
        elif uploaded_file.name != st.session_state.uploaded_file_name:
            # Already indexed (e.g. from an earlier session): only load the page preview.
            # If figure captioning never finished (e.g. interrupted upload), resume it
            # in the background; the summary stays lazy either way.
            st.session_state.page_texts = extract_page_texts_from_pdf(uploaded_file)
            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.processing_status = f"'{uploaded_file.name}' is already indexed."
            pipeline.ensure_document_images(uploaded_file.name)

        st.success("PDF uploaded successfully.")
        st.markdown(
            f"""
            <div class='status-card'>
                <span class='status-badge'>PDF READY</span>
                <span class='status-badge'>RAG ACTIVE</span>
                <span class='status-badge'>VISION</span>
                <br><br>
                <strong>File:</strong> {uploaded_file.name}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.session_state.processing_status:
            with st.expander("PDF processing preview"):
                st.text_area(
                    "Preview / status",
                    st.session_state.processing_status[:3000],
                    height=250,
                )
        else:
            st.warning("No text or index status could be created from this PDF.")

    #
    # Document management: choose which document questions refer to, delete old ones.
    if indexed_sources:
        st.divider()
        st.subheader("🎯 Documents")

        scope_options = ["All documents"] + indexed_sources
        selected_scope = st.selectbox(
            "Answer questions using:",
            scope_options,
            key="active_source_select",
        )
        st.session_state.active_source = (
            None if selected_scope == "All documents" else selected_scope
        )

        st.caption("Indexed documents:")
        for source_name in indexed_sources:
            name_column, delete_column = st.columns([5, 1])
            label = source_name
            if pipeline.is_indexing_images(source_name):
                label += " · 🖼️ describing figures…"
            name_column.caption(label)
            if delete_column.button("🗑️", key=f"delete_{source_name}", help=f"Delete {source_name}"):
                PDFProcessor().delete_pdf(source_name)
                st.session_state.just_deleted_source = source_name
                if st.session_state.uploaded_file_name == source_name:
                    st.session_state.uploaded_file_name = None
                    st.session_state.processing_status = ""
                    st.session_state.page_texts = {}
                st.rerun()
    else:
        st.session_state.active_source = None

    if st.button("Clear chat"):
        st.session_state.messages = []

    st.divider()
    st.subheader("📦 JSON Export")

    if not indexed_sources:
        st.caption("Upload a PDF first to enable JSON export.")
    elif st.session_state.active_source is None:
        st.caption("Select a specific document above to enable JSON export.")
    else:
        if st.button("Generate sustainability JSON"):
            with st.spinner("Extracting structured sustainability data..."):
                st.session_state.json_export = {
                    "document": st.session_state.active_source,
                    "data": create_json_export(pipeline, st.session_state.active_source),
                }

        export = st.session_state.json_export
        if export is not None and export.get("document") == st.session_state.active_source:
            st.download_button(
                label="Download JSON",
                data=convert_json_export_to_string(export["data"]),
                file_name=f"{Path(export['document']).stem}_sustainability.json",
                mime="application/json",
            )

            with st.expander("Preview JSON"):
                st.json(export["data"])

    st.divider()
    st.caption("System status")
    st.markdown("<span class='status-badge'>Streamlit UI</span>", unsafe_allow_html=True)
    st.markdown("<span class='status-badge'>PDF Processing</span>", unsafe_allow_html=True)
    st.markdown("<span class='status-badge'>RAG Pipeline</span>", unsafe_allow_html=True)


#
# Shows a status box that tells the user whether documents are ready for the chat.
# The chat works as soon as the vector database contains at least one document,
# even after an app restart without a fresh upload.
documents_ready = bool(indexed_sources) or bool(st.session_state.processing_status)

if not documents_ready:
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
    active_scope = st.session_state.active_source or "all uploaded documents"
    st.markdown(
        f"""
        <div class='chat-info-box'>
        <strong>Document is ready.</strong><br>
        Ask a question below. Answers are based on: <strong>{html.escape(active_scope)}</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )


#
# Re-renders the stored chat history after every Streamlit rerun.
for message in st.session_state.messages:
    render_chat_bubble(message["role"], message["content"], message.get("sources"))


#
# Creates the chat input field where the user asks questions about the uploaded PDF.
user_question = st.chat_input("Ask a question about the PDF...")

#
# Handles a new user question: store it, query the RAG pipeline, render the answer, and save it to chat history.
if user_question:
    if not documents_ready:
        st.warning("Please upload a PDF first.")
    else:
        st.session_state.messages.append(
            {"role": "user", "content": user_question}
        )

        render_chat_bubble("user", user_question)

        if is_summary_question(user_question) and not pipeline.has_summary(st.session_state.active_source):
            spinner_text = "Generating document summary (first time for this document, can take a bit)..."
        else:
            spinner_text = "Searching and re-ranking relevant PDF sections..."

        with st.spinner(spinner_text):
            result = get_rag_result(user_question)

        raw_answer = result["answer"]

        # Antwort streamen (Generator) oder direkt anzeigen (fertiger String)
        with st.chat_message("assistant", avatar="🌱"):
            if isinstance(raw_answer, str):
                answer_text = raw_answer
                st.write(answer_text)
            else:
                answer_text = st.write_stream(raw_answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": clean_message_content(str(answer_text)),
                "sources": result.get("sources", []),
            }
        )

        # Neu rendern, damit die Antwort mit Quellen-Pills und Expander erscheint
        st.rerun()
