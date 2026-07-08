import base64
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from math import exp
from pathlib import Path

import chromadb
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

from pdf_processing import CHROMA_DIR, DB_WRITE_LOCK, IMAGE_DIR, get_collection

# Lade Umgebungsvariablen
load_dotenv(find_dotenv(), override=True)


# Schlagwörter, die auf eine globale Frage ("Fasse den Report zusammen") hindeuten.
# Solche Fragen lassen sich nicht über Ähnlichkeitssuche beantworten, sondern
# brauchen die beim Ingest erzeugte Dokument-Zusammenfassung.
SUMMARY_KEYWORDS = (
    "summar", "zusammenfass", "zusammenfassung", "überblick", "ueberblick",
    "overview", "worum geht", "um was geht", "what is the report about",
    "what is this document about", "main points", "key points", "key findings",
    "main findings", "main topics", "hauptpunkte", "kernaussagen", "kernpunkte",
    "wichtigsten punkte", "wichtigsten themen", "tl;dr", "tldr", "in a nutshell",
)

# Wie viele Zeichen Dokumenttext pro Map-Aufruf zusammengefasst werden
SUMMARY_BATCH_CHARS = 15000
# Parallele LLM-Aufrufe beim Map-Schritt
SUMMARY_MAX_WORKERS = 4

# Retrieval: erst breit suchen, dann mit dem Cross-Encoder präzise nachsortieren
RETRIEVAL_CANDIDATES = 20   # Kandidaten aus der Vektorsuche vor dem Re-Ranking
RERANK_TOP_K = 6            # maximal so viele Chunks gehen ans LLM
RERANK_MIN_SCORE = 0.15     # Relevanz-Schwelle (Sigmoid-Score) gegen irrelevante Treffer
RERANK_MIN_CHUNKS = 3       # so viele Chunks bleiben trotz Schwelle mindestens erhalten
RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# Chat-Historie: wie viele vergangene Nachrichten in Rewriting und Antwort einfließen
HISTORY_MAX_MESSAGES = 6

# Ohne expliziten Timeout wartet das OpenAI-SDK bis zu 10 Minuten auf eine einzelne
# hängende Anfrage. Bei mehreren sequenziellen Calls (Summary-Batches, Bildunterschriften)
# reicht ein einziger hängender Server-Request, um alles lahmzulegen. Fail-fast statt dessen.
LLM_TIMEOUT_SECONDS = 60.0
LLM_MAX_RETRIES = 1

# Vision: Bilder werden beim Ingest von einem multimodalen Modell beschrieben
DEFAULT_VISION_MODEL = "gemma-4-31b-it"
MAX_IMAGES_PER_DOC = 60     # Obergrenze an Vision-Beschreibungen pro Dokument
VISION_MAX_WORKERS = 4


def is_summary_question(question: str) -> bool:
    question_lower = question.lower()
    if any(keyword in question_lower for keyword in SUMMARY_KEYWORDS):
        return True
    # Deutsches trennbares Verb: "Fasse den Bericht zusammen", "Fassen Sie ... zusammen"
    return re.search(r"\bfass\w*\b.*\bzusammen", question_lower) is not None


# Welche Dokumente werden gerade im Hintergrund mit Bildbeschreibungen indexiert.
# Rein informativ für die UI (Sidebar-Hinweis "Bilder werden noch beschrieben...").
_image_indexing_in_progress: set[str] = set()

_reranker = None


def get_reranker():
    """Lädt den Cross-Encoder nur einmal pro Prozess (teurer Modell-Load)."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


# Erkennt Modellantworten, die keine echte Zusammenfassung sind (Rückfragen,
# Verweigerungen), damit solche Antworten nicht dauerhaft als "fertige Summary"
# gespeichert werden und künftige Summary-Fragen für immer blockieren.
_REFUSAL_PATTERN = re.compile(
    r"^(please provide|i cannot|i can't|i'?m unable|i am unable|sorry|as an ai|"
    r"i need (the|a) (report|document|text))",
    re.IGNORECASE,
)


def _is_valid_summary(text: str) -> bool:
    text = (text or "").strip()
    return len(text) >= 40 and not _REFUSAL_PATTERN.match(text)


def _as_probability(score: float) -> float:
    """Normalisiert Reranker-Ausgaben auf [0, 1] (Logits werden durch Sigmoid geschickt)."""
    score = float(score)
    if 0.0 <= score <= 1.0:
        return score
    return 1.0 / (1.0 + exp(-score))


class GWDGRagPipeline:
    def __init__(self, persist_directory=None):
        api_key = os.getenv("SAIA_API_KEY")
        base_url = os.getenv("SAIA_BASE_URL")
        self.model_name = os.getenv("SAIA_MODEL")
        self.vision_model = os.getenv("SAIA_VISION_MODEL", DEFAULT_VISION_MODEL)

        if not api_key or not base_url or not self.model_name:
            raise ValueError("Fehlende SAIA Umgebungsvariablen in der .env Datei!")

        # GWDG LLM Client. Expliziter Timeout + begrenzte Retries, damit ein einzelner
        # hängender Request nicht die ganze Pipeline (Summary, Bildbeschreibungen, Antworten)
        # für bis zu 10 Minuten blockiert (SDK-Default ohne eigenen Timeout).
        self.llm_client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=LLM_TIMEOUT_SECONDS,
            max_retries=LLM_MAX_RETRIES,
        )

        # Verbinde mit derselben Chroma-Datenbank, die von pdf_processing.py gefüllt wird
        self.chroma_client = chromadb.PersistentClient(
            path=str(persist_directory) if persist_directory else str(CHROMA_DIR)
        )
        self.collection = get_collection(self.chroma_client)

    # ==========================================
    # Retrieval
    # ==========================================

    def list_sources(self) -> list[str]:
        """Listet die Dateinamen aller aktuell indexierten Dokumente."""
        data = self.collection.get(include=["metadatas"])
        return sorted({meta["source"] for meta in data["metadatas"] if meta.get("source")})

    @staticmethod
    def _where_filter(source: str | None, chunk_types: list[str]) -> dict:
        """Baut den Chroma-Metadaten-Filter für Chunk-Typen und optional ein Dokument."""
        type_filter = (
            {"type": chunk_types[0]} if len(chunk_types) == 1 else {"type": {"$in": chunk_types}}
        )
        if source:
            return {"$and": [type_filter, {"source": source}]}
        return type_filter

    def retrieve_relevant_chunks(
        self, question: str, max_chunks: int = RERANK_TOP_K, source: str | None = None
    ) -> list[dict]:
        """Zweistufiges Retrieval: breite Vektorsuche, dann Cross-Encoder-Re-Ranking.

        Durchsucht Text-Chunks UND Bildbeschreibungen. Mit `source` wird die Suche
        auf ein einzelnes Dokument eingeschränkt. Rückgabe: Liste aus Dicts mit
        "text", "meta" und "score" (Relevanz 0-1), absteigend sortiert.
        """
        total = self.collection.count()
        if total == 0:
            return []

        results = self.collection.query(
            query_texts=[question],
            n_results=min(RETRIEVAL_CANDIDATES, total),
            where=self._where_filter(source, ["text", "image"]),
        )

        if not results["documents"] or not results["documents"][0]:
            return []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        scores = get_reranker().predict([(question, doc) for doc in documents])
        ranked = sorted(
            (
                {"text": doc, "meta": meta, "score": _as_probability(score)}
                for doc, meta, score in zip(documents, metadatas, scores)
            ),
            key=lambda item: item["score"],
            reverse=True,
        )

        selected = ranked[:max_chunks]
        confident = [item for item in selected if item["score"] >= RERANK_MIN_SCORE]
        if len(confident) < RERANK_MIN_CHUNKS:
            confident = selected[:RERANK_MIN_CHUNKS]
        return confident

    @staticmethod
    def _format_chunks_for_llm(chunks: list[dict]) -> list[str]:
        """Rahmt Chunks mit Quellen-Metadaten für den LLM-Kontext ein.

        Die Labels sind bewusst englisch ("Source"/"Page"), damit das Modell sie 1:1
        in das geforderte Zitierformat "(Source: ..., Page ...)" übernimmt.
        """
        formatted = []
        for item in chunks:
            meta = item["meta"]
            page = meta.get("page", "unknown")
            page_end = meta.get("page_end", page)
            page_label = f"{page}-{page_end}" if page_end != page else f"{page}"
            source_name = meta.get("source", "unknown")
            kind = " | figure description" if meta.get("type") == "image" else ""

            formatted.append(
                f"--- SECTION START (Source: {source_name}, Page: {page_label}{kind}) ---\n"
                f"{item['text']}\n--- SECTION END ---"
            )
        return formatted

    @staticmethod
    def _sources_from_chunks(chunks: list[dict]) -> list[dict]:
        """Baut die strukturierte Quellenliste für die UI aus den Chunk-Metadaten."""
        sources = []
        for item in chunks:
            meta = item["meta"]
            sources.append(
                {
                    "source": meta.get("source"),
                    "page": meta.get("page"),
                    "page_end": meta.get("page_end", meta.get("page")),
                    "type": meta.get("type", "text"),
                    "image_path": meta.get("image_path"),
                    "text": item["text"],
                    "score": round(item["score"], 3),
                }
            )
        return sources

    # ==========================================
    # Bildbeschreibungen (Vision-Modell)
    # ==========================================

    def _caption_image(self, image_path: Path, source: str) -> str | None:
        """Lässt das Vision-Modell eine Abbildung beschreiben.

        Liefert None für Deko-Bilder ohne Informationsgehalt (Modell antwortet SKIP)
        oder bei Fehlern, damit kein Müll in der Vektordatenbank landet.
        """
        mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
        encoded = base64.b64encode(image_path.read_bytes()).decode()

        prompt = (
            f"This image comes from the corporate/sustainability report '{source}'. "
            "If it is a chart, diagram, table, or infographic: describe it precisely in 2-5 "
            "sentences (chart type, axes, categories, all legible values and labels, and the key "
            "statement), in the language of the text in the image (default: English). "
            "If it is a photo, illustration, logo, or decorative graphic without data content, "
            "reply with exactly one word: SKIP. Never explain why you cannot describe something."
        )

        try:
            response = self.llm_client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                        ],
                    }
                ],
                temperature=0.0,
                max_tokens=400,
            )
            caption = (response.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"Vision-Fehler bei {image_path.name}: {e}")
            return None

        if not caption or caption.upper().startswith("SKIP"):
            return None
        # Verweigerungen und Halluzinationen (z. B. erfundene Dateipfade) aussortieren,
        # damit kein Müll in der Vektordatenbank landet
        if re.match(r"^(i cannot|i can't|i'?m unable|i am unable|sorry)", caption, re.IGNORECASE):
            return None
        if re.search(r"[A-Z]:\\|/Users/", caption) or len(caption) < 60:
            return None
        return caption

    def index_document_images(self, source: str) -> dict:
        """Beschreibt die extrahierten Bilder eines Dokuments und indexiert die
        Beschreibungen als durchsuchbare Chunks (type="image").

        Läuft typischerweise in einem Hintergrund-Thread (siehe ensure_document_images),
        da jedes Bild einen eigenen Vision-Modell-Aufruf braucht und das bei
        bildreichen Reports mehrere Minuten dauern kann. Größere Bilder zuerst
        (Charts sind meist groß), gedeckelt durch MAX_IMAGES_PER_DOC.
        """
        _image_indexing_in_progress.add(source)
        try:
            image_paths = sorted(
                IMAGE_DIR.glob(f"{source}_p*"),
                key=lambda path: path.stat().st_size,
                reverse=True,
            )[:MAX_IMAGES_PER_DOC]

            stats = {"captioned": 0, "skipped": 0}
            if not image_paths:
                return stats

            print(f"Beschreibe {len(image_paths)} Abbildungen aus '{source}' mit {self.vision_model} ...")
            with ThreadPoolExecutor(max_workers=VISION_MAX_WORKERS) as executor:
                captions = list(
                    executor.map(lambda path: self._caption_image(path, source), image_paths)
                )

            documents, metadatas, ids = [], [], []
            for image_path, caption in zip(image_paths, captions):
                if caption is None:
                    stats["skipped"] += 1
                    continue
                page_match = re.search(r"_p(\d+)_img_\d+\.\w+$", image_path.name)
                page = int(page_match.group(1)) if page_match else 0

                documents.append(caption)
                metadatas.append(
                    {
                        "source": source,
                        "type": "image",
                        "page": page,
                        "page_end": page,
                        "image_path": str(image_path),
                    }
                )
                ids.append(image_path.name)
                stats["captioned"] += 1

            if documents:
                with DB_WRITE_LOCK:
                    self.collection.upsert(documents=documents, metadatas=metadatas, ids=ids)

            print(
                f"'{source}': {stats['captioned']} Bildbeschreibungen indexiert, "
                f"{stats['skipped']} Bilder übersprungen."
            )
            return stats
        finally:
            _image_indexing_in_progress.discard(source)

    def is_indexing_images(self, source: str) -> bool:
        """Ob für dieses Dokument gerade im Hintergrund Bilder beschrieben werden."""
        return source in _image_indexing_in_progress

    def ensure_document_images(self, source: str) -> bool:
        """Stößt Bild-Captioning im Hintergrund an, falls noch nicht geschehen.

        Nicht blockierend: läuft in einem Daemon-Thread, damit ein bildreiches
        Dokument den Chat nicht erst nach mehreren Minuten Vision-Aufrufen
        freigibt. Gibt True zurück, wenn ein neuer Hintergrund-Lauf gestartet wurde.
        """
        if self.is_indexing_images(source):
            return False
        existing = self.collection.get(where=self._where_filter(source, ["image"]), include=[])
        if existing["ids"]:
            return False

        thread = threading.Thread(target=self.index_document_images, args=(source,), daemon=True)
        thread.start()
        return True

    # ==========================================
    # Dokument-Zusammenfassungen (für globale Fragen)
    # ==========================================

    def _summarize_text(self, text: str, instruction: str) -> str:
        """Einzelner LLM-Aufruf für den Map- bzw. Reduce-Schritt."""
        response = self.llm_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert analyst for corporate and sustainability reports. "
                        "Write factual, dense summaries without filler. "
                        "Always write the summary in the same language as the source text."
                    ),
                },
                {"role": "user", "content": f"{instruction}\n\n{text}"},
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content

    def _safe_summarize_text(self, text: str, instruction: str) -> str:
        """Wie _summarize_text, aber fehlertolerant für den parallelen Map-Schritt.

        Ein einzelner Timeout/Fehler bei einem von mehreren Abschnitten soll nicht
        die gesamte Dokument-Zusammenfassung zum Scheitern bringen.
        """
        try:
            result = self._summarize_text(text, instruction)
        except Exception as e:
            print(f"Zusammenfassung eines Abschnitts fehlgeschlagen, wird übersprungen: {e}")
            return ""

        if not _is_valid_summary(result):
            print(f"Zusammenfassung eines Abschnitts wirkte ungültig, wird übersprungen: {result[:80]!r}")
            return ""
        return result

    def build_document_summary(self, source: str) -> str | None:
        """Erzeugt per Map-Reduce eine Gesamtzusammenfassung eines Dokuments
        und speichert sie als Summary-Chunk in der Collection.

        Wird einmal beim Ingest aufgerufen; globale Fragen wie "Summarize the report"
        werden später direkt aus dieser Zusammenfassung beantwortet.
        """
        data = self.collection.get(
            where=self._where_filter(source, ["text"]),
            include=["documents"],
        )
        if not data["documents"]:
            return None

        # Chunks in Dokumentreihenfolge bringen (IDs enden auf _c<index>)
        def chunk_index(chunk_id: str) -> int:
            try:
                return int(chunk_id.rsplit("_c", 1)[1])
            except (IndexError, ValueError):
                return 0

        ordered = [doc for _, doc in sorted(zip(data["ids"], data["documents"]), key=lambda pair: chunk_index(pair[0]))]

        # Map-Schritt: Dokument in große Abschnitte teilen und parallel zusammenfassen
        batches: list[str] = []
        current_batch: list[str] = []
        current_len = 0
        for chunk in ordered:
            if current_len + len(chunk) > SUMMARY_BATCH_CHARS and current_batch:
                batches.append("\n\n".join(current_batch))
                current_batch = []
                current_len = 0
            current_batch.append(chunk)
            current_len += len(chunk)
        if current_batch:
            batches.append("\n\n".join(current_batch))

        map_instruction = (
            "Summarize the following report section in 5-8 sentences. "
            "Keep concrete figures, targets, and key statements."
        )
        print(f"Erzeuge Zusammenfassung für '{source}' ({len(batches)} Abschnitte) ...")

        if len(batches) == 1:
            section_summaries = [self._safe_summarize_text(batches[0], map_instruction)]
        else:
            with ThreadPoolExecutor(max_workers=SUMMARY_MAX_WORKERS) as executor:
                section_summaries = list(
                    executor.map(lambda batch: self._safe_summarize_text(batch, map_instruction), batches)
                )
        section_summaries = [summary for summary in section_summaries if summary]

        # Reduce-Schritt: Abschnitts-Zusammenfassungen zu einer Gesamtzusammenfassung verdichten
        if len(section_summaries) == 1:
            summary = section_summaries[0]
        else:
            reduce_instruction = (
                "The following are section summaries of one report, in order. "
                "Combine them into one coherent overall summary of the report "
                "(around 300-400 words). Keep the most important figures, targets, "
                "and key statements."
            )
            summary = self._summarize_text("\n\n".join(section_summaries), reduce_instruction)

        if not _is_valid_summary(summary):
            print(
                f"Zusammenfassung für '{source}' wirkte ungültig und wird NICHT gespeichert "
                f"(nächste Summary-Frage versucht es erneut): {summary[:80]!r}"
            )
            return None

        with DB_WRITE_LOCK:
            self.collection.upsert(
                documents=[summary],
                metadatas=[{"source": source, "type": "summary", "page": 0, "page_end": 0}],
                ids=[f"{source}_summary"],
            )
        print(f"Zusammenfassung für '{source}' gespeichert.")
        return summary

    def _get_stored_summaries(self, source: str | None = None) -> list[tuple[str, str]]:
        """Liefert gespeicherte (Dokumentname, Zusammenfassung)-Paare."""
        data = self.collection.get(
            where=self._where_filter(source, ["summary"]),
            include=["documents", "metadatas"],
        )
        return [
            (meta.get("source", "Unbekannt"), doc)
            for doc, meta in zip(data["documents"], data["metadatas"])
        ]

    def has_summary(self, source: str | None) -> bool:
        """Ob für das/die betroffene(n) Dokument(e) bereits eine Zusammenfassung existiert.

        Für die UI, um den Spinner-Text vor einer Summary-Frage passend zu wählen.
        """
        sources_to_check = [source] if source else self.list_sources()
        return bool(sources_to_check) and all(self._get_stored_summaries(s) for s in sources_to_check)

    # ==========================================
    # Chat-Historie / Query-Rewriting
    # ==========================================

    @staticmethod
    def _history_messages(history: list[dict]) -> list[dict]:
        """Reduziert die Chat-Historie auf role/content und die letzten Nachrichten."""
        trimmed = []
        for message in history[-HISTORY_MAX_MESSAGES:]:
            role = message.get("role")
            content = str(message.get("content", "")).strip()
            if role in ("user", "assistant") and content:
                trimmed.append({"role": role, "content": content})
        return trimmed

    def _rewrite_question(self, question: str, history: list[dict]) -> str:
        """Macht Follow-up-Fragen ("Und 2021?") zu eigenständigen Suchanfragen.

        Nur die Retrieval-Suche nutzt die umgeschriebene Frage; die Antwort wird
        weiterhin auf die Originalfrage gegeben. Bei Fehlern: Originalfrage.
        """
        messages = self._history_messages(history)
        if not messages:
            return question

        transcript = "\n".join(
            f"{message['role']}: {message['content'][:500]}" for message in messages
        )

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You rewrite follow-up questions into self-contained search queries. "
                            "Given a conversation and the latest question, rewrite that question so it "
                            "is fully understandable without the conversation (resolve pronouns, add the "
                            "topic). Keep the original language. Return ONLY the rewritten question."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Conversation:\n{transcript}\n\nLatest question: {question}",
                    },
                ],
                temperature=0.0,
                max_tokens=150,
            )
            rewritten = (response.choices[0].message.content or "").strip()
            return rewritten or question
        except Exception as e:
            print(f"Query-Rewriting fehlgeschlagen, nutze Originalfrage: {e}")
            return question

    # ==========================================
    # Antwort-Generierung
    # ==========================================

    def generate_answer(
        self,
        question: str,
        context_chunks: list[str],
        summary_mode: bool = False,
        history: list[dict] | None = None,
        stream: bool = False,
    ):
        """Sendet Frage, Chat-Historie und Kontext-Chunks an das Modell.

        Mit stream=True wird ein Generator über Text-Deltas zurückgegeben
        (für st.write_stream), sonst der fertige Antwort-String.
        """
        context_text = "\n\n".join(context_chunks)

        if summary_mode:
            citation_rule = (
                "5. CITATIONS: The context consists of whole-document summaries. "
                "Cite the document name in parentheses, e.g. (Source: [Document Name]). "
                "Do not invent page numbers."
            )
        else:
            citation_rule = (
                "5. IN-TEXT CITATIONS: Each context section is provided with a document name and page number. "
                "You MUST cite your sources directly in the text. Place the corresponding source immediately "
                "at the end of the respective sentence in parentheses, using exactly this format: "
                "(Source: [Document Name], Page [X]). NEVER just list the sources at the end of your response!"
            )

        system_prompt = (
            "You are a highly qualified academic assistant and subject matter expert for analyzing corporate and sustainability reports. "
            "Your task is to answer user questions precisely, objectively, and factually.\n\n"
            "STRICTLY adhere to the following rules:\n"
            "1. FACT-BASED ONLY: Answer the question SOLELY based on the provided PDF context below. Do not use any external knowledge.\n"
            "2. ANTI-HALLUCINATION: If the provided context does not contain sufficient information to answer the question, do not guess. Reply exactly with: 'The provided document does not contain information on this topic.' (or the equivalent in the user's language).\n"
            "3. STYLE & TONE: Formulate your answers in a clean, professional, and objective academic tone. Use complete sentences.\n"
            "4. MULTILINGUAL (CRITICAL): You MUST answer in the EXACT SAME LANGUAGE as the user's question. If the user asks in English, write your entire response in English. If the user asks in German, write your entire response in German. Match the language perfectly.\n"
            f"{citation_rule}"
        )

        user_prompt = f"Context from the document:\n{context_text}\n\nUser question: {question}"

        messages = (
            [{"role": "system", "content": system_prompt}]
            + self._history_messages(history or [])
            + [{"role": "user", "content": user_prompt}]
        )

        response = self.llm_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.0,
            stream=stream,
        )

        if not stream:
            return response.choices[0].message.content

        def token_generator():
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        return token_generator()

    def answer(
        self,
        question: str,
        source: str | None = None,
        history: list[dict] | None = None,
        stream: bool = False,
    ) -> dict:
        """Hauptfunktion für die App: beantwortet eine Frage mit Quellen-Metadaten.

        Rückgabe: {"answer": str | Generator, "sources": list[dict]}.
        Globale Fragen laufen über die gespeicherten Dokument-Zusammenfassungen,
        Detailfragen über Vektorsuche + Re-Ranking. `source` schränkt auf ein
        Dokument ein, `history` ermöglicht Follow-up-Fragen.
        """
        history = history or []

        if is_summary_question(question):
            # Zusammenfassungen werden bewusst NICHT beim Upload erzeugt (das würde den
            # Chat unnötig blockieren), sondern erst hier, beim ersten tatsächlichen Bedarf.
            # Einmal gebaut, bleiben sie gespeichert und werden nur noch abgerufen.
            sources_to_summarize = [source] if source else self.list_sources()
            for doc_source in sources_to_summarize:
                if not self._get_stored_summaries(doc_source):
                    self.build_document_summary(doc_source)

            summaries = self._get_stored_summaries(source)
            if summaries:
                summary_chunks = [
                    f"--- SUMMARY OF DOCUMENT '{doc_name}' ---\n{summary_text}"
                    for doc_name, summary_text in summaries
                ]
                return {
                    "answer": self.generate_answer(
                        question, summary_chunks, summary_mode=True, history=history, stream=stream
                    ),
                    "sources": [
                        {"source": doc_name, "page": None, "page_end": None,
                         "type": "summary", "image_path": None, "text": summary_text, "score": None}
                        for doc_name, summary_text in summaries
                    ],
                }
            # Kein Dokument im gewählten Umfang: breiter suchen statt leer zu antworten
            search_query = self._rewrite_question(question, history)
            retrieved = self.retrieve_relevant_chunks(search_query, max_chunks=10, source=source)
        else:
            search_query = self._rewrite_question(question, history)
            retrieved = self.retrieve_relevant_chunks(search_query, source=source)

        if not retrieved:
            return {
                "answer": (
                    "Es konnten keine relevanten Informationen in der Datenbank gefunden werden. "
                    "Bitte lade zuerst ein PDF hoch."
                ),
                "sources": [],
            }

        return {
            "answer": self.generate_answer(
                question,
                self._format_chunks_for_llm(retrieved),
                history=history,
                stream=stream,
            ),
            "sources": self._sources_from_chunks(retrieved),
        }

    def answer_question_with_rag(self, question: str, source: str | None = None) -> str:
        """Abwärtskompatible String-Variante (CLI, Tests, ältere Aufrufer)."""
        return self.answer(question, source=source, stream=False)["answer"]


# --- Lazy Singleton für App und Skripte ---
_pipeline_instance = None


def get_pipeline() -> GWDGRagPipeline:
    """Erzeugt die Pipeline erst beim ersten Zugriff.

    Vermeidet den teuren Import-Side-Effect (LLM-Client, Vektor-DB und
    Embedding-Modell wurden früher schon beim bloßen Import geladen) und lässt
    Aufrufer Initialisierungsfehler (fehlende .env) sauber behandeln.
    """
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = GWDGRagPipeline()
    return _pipeline_instance
