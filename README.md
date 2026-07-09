# AI PDF Chat App - RAG Rangers ;)

A RAG-powered web app for analyzing corporate **sustainability reports**: upload a PDF, ask questions about it in natural language, and get precise, fact-based answers with page-level citations — plus structured JSON exports of key sustainability data.

Built with Streamlit, ChromaDB, and the GWDG SAIA API (Gemma), with all embeddings computed locally.

## Features

- **Chat with your reports** — ask detailed questions in German or English; answers are grounded strictly in the uploaded PDFs and cite document and page inline (e.g. *(Source: report.pdf, Page 4)*), with expandable source previews including relevance scores.
- **Document summaries** — a one-click summary button and automatic detection of global questions ("What is this report about?"), answered from a map-reduce summary generated in the background after upload.
- **Figures made searchable** — charts and infographics are extracted from the PDF and described by a vision model, so questions can also be answered from figures, not just text.
- **Follow-up questions** — recent chat history is used to rewrite questions like "And in 2021?" into self-contained search queries.
- **Multi-document support** — manage several indexed reports, scope questions to one document or all, and delete documents cleanly.
- **JSON export** — extract key sustainability indicators of a selected report into a structured JSON file, and export the chat history (questions, answers, sources, response times).

## How it works

```
PDF upload
   └─ pdf_processing.py   PyMuPDF text extraction → noise filtering → overlapping chunks
                          → local embeddings (intfloat/multilingual-e5-base) → ChromaDB
   └─ background threads  figure captioning (vision model) + map-reduce document summary
Question
   └─ rag_pipeline.py     query rewriting → vector search (top 20) → cross-encoder
                          re-ranking (mmarco-mMiniLMv2) → answer via SAIA LLM, streamed
```

- **Embeddings and re-ranking run locally on CPU** — only answer generation, query rewriting, summaries, and figure descriptions call the SAIA API (`gemma-4-31b-it` by default).
- The vector index (`chroma_db/`) persists across restarts; indexed documents stay chattable without re-uploading.
- If the embedding model is ever changed in the code, the index detects the mismatch on startup and rebuilds itself automatically from the stored texts — locally, without API calls.

## Setup

1. **Install dependencies** (Python 3.12+):

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure the SAIA API** in a `.env` file at the project root:

   ```env
   SAIA_API_KEY=your-key-here
   SAIA_BASE_URL=https://chat-ai.academiccloud.de/v1
   SAIA_MODEL=gemma-4-31b-it
   ```

   `check_models.py` lists the models currently available on the server.

3. **Run the app:**

   ```bash
   streamlit run app/main.py
   ```

   The first run downloads the embedding and re-ranking models from Hugging Face (~1.5 GB, one-time).

## Project structure

```
app/
  main.py             Streamlit UI: upload, chat, document management, exports
  pdf_processing.py   PDF parsing, chunking, image extraction, ChromaDB indexing
  rag_pipeline.py     retrieval, re-ranking, summaries, vision captions, LLM calls
  json_export.py      structured sustainability JSON + chat history export
check_models.py       lists models available on the SAIA endpoint
pdfs/, extracted_images/, chroma_db/   local data (created at runtime)
frontend (Plan B)/    experimental React frontend (not required for the app)
```

## Notes

- The SAIA quota is per API key. If it is temporarily exhausted, the app shows a rate-limit notice instead of failing — wait a moment and ask again; documents and index are unaffected.
- Uploaded PDFs, extracted images, and the vector database are stored locally. Report content is only sent to the SAIA API where needed for generation: retrieved text excerpts as answer context, section texts for summaries, and figure images for captioning.
