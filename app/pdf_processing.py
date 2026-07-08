import io
import os
import re
import threading
from pathlib import Path

import chromadb
import fitz  # Das ist PyMuPDF
from PIL import Image
from chromadb.utils import embedding_functions

# Alle Pfade an der Projektwurzel verankern, damit App und Skripte unabhängig
# vom Arbeitsverzeichnis immer dieselbe Datenbank und dieselben Ordner nutzen.
BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / "pdfs"
IMAGE_DIR = BASE_DIR / "extracted_images"
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "eco_reports"
# Mehrsprachiges State-of-the-Art-Retrieval-Modell (braucht keine Query-/Passage-Präfixe)
EMBEDDING_MODEL = "BAAI/bge-m3"

# Zielgröße eines Chunks in Zeichen und Überlappung zwischen benachbarten Chunks
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Bilder unterhalb dieser Kantenlänge (Logos, Icons, Deko) werden übersprungen
MIN_IMAGE_SIZE = 100


# Serialisiert alle schreibenden Zugriffe auf die (SQLite-gestützte) Collection.
# Nötig, seit Bildbeschreibungen in einem Hintergrund-Thread nachindexiert werden
# und dabei mit Chunking/Löschungen im Hauptthread kollidieren könnten.
DB_WRITE_LOCK = threading.Lock()

_embedding_function = None


def get_embedding_function():
    """Lädt das Embedding-Modell nur einmal pro Prozess (teurer Modell-Load)."""
    global _embedding_function
    if _embedding_function is None:
        _embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL,
            normalize_embeddings=True,
        )
    return _embedding_function


def get_collection(chroma_client):
    """Öffnet bzw. erstellt die zentrale Collection mit einheitlicher Konfiguration.

    Cosine-Distanz passt zu den normalisierten bge-m3-Embeddings; der Helper stellt
    sicher, dass PDFProcessor und RAG-Pipeline identische Einstellungen verwenden.
    """
    return chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def _is_noise_block(raw_text: str) -> bool:
    """Erkennt Blöcke ohne inhaltlichen Wert (Seitenzahlen, Inhaltsverzeichnis-Einträge)."""
    stripped = raw_text.strip()
    if not stripped:
        return True
    # Nackte Seitenzahl
    if re.fullmatch(r"\d{1,4}", stripped):
        return True
    # Inhaltsverzeichnis-Eintrag wie "12\tKapiteltitel"
    if re.match(r"^\d{1,3}\s*\t", stripped):
        return True
    # Punktleisten wie "Kapitel .......... 12"
    if re.search(r"\.{4,}\s*\d{1,4}$", stripped):
        return True
    return False


def _clean_block_text(raw_text: str) -> str:
    """Entfernt Steuerzeichen und normalisiert Whitespace innerhalb eines Blocks."""
    text = raw_text.replace("\x07", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _recursive_split(text: str, separators: tuple[str, ...] = (". ", "; ", ", ", " ")) -> list[str]:
    """Zerteilt überlange Texte rekursiv an natürlichen Grenzen in Stücke <= CHUNK_SIZE."""
    if len(text) <= CHUNK_SIZE:
        return [text]

    for sep_index, sep in enumerate(separators):
        if sep not in text:
            continue

        parts = text.split(sep)
        # Separator wieder anhängen, damit kein Text verloren geht
        parts = [part + sep for part in parts[:-1]] + [parts[-1]]

        pieces: list[str] = []
        buffer = ""
        for part in parts:
            if len(buffer) + len(part) > CHUNK_SIZE and buffer:
                pieces.append(buffer.strip())
                buffer = buffer[-CHUNK_OVERLAP:]
            if len(part) > CHUNK_SIZE:
                if buffer:
                    pieces.append(buffer.strip())
                deeper = _recursive_split(part, separators[sep_index + 1:])
                pieces.extend(piece.strip() for piece in deeper[:-1])
                buffer = deeper[-1]
            else:
                buffer += part
        if buffer.strip():
            pieces.append(buffer.strip())
        return [piece for piece in pieces if piece]

    # Kein Separator mehr vorhanden: hart schneiden
    step = CHUNK_SIZE - CHUNK_OVERLAP
    return [text[i:i + CHUNK_SIZE] for i in range(0, len(text), step)]


class PDFProcessor:
    def __init__(self, pdf_dir=None, image_dir=None, persist_directory=None):
        self.pdf_dir = Path(pdf_dir) if pdf_dir else PDF_DIR
        self.image_dir = Path(image_dir) if image_dir else IMAGE_DIR

        for directory in [self.pdf_dir, self.image_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(
            path=str(persist_directory) if persist_directory else str(CHROMA_DIR)
        )
        self.collection = get_collection(self.chroma_client)

    # ==========================================
    # Chunking
    # ==========================================

    def _extract_text_units(self, doc) -> list[tuple[int, str]]:
        """Liefert bereinigte (Seite, Text)-Einheiten für alle brauchbaren Textblöcke."""
        units = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            # sort=True stellt die Lesereihenfolge auch bei mehrspaltigen Layouts her
            for block in page.get_text("blocks", sort=True):
                if block[6] != 0:  # nur Textblöcke, keine Bildblöcke
                    continue
                if _is_noise_block(block[4]):
                    continue
                text = _clean_block_text(block[4])
                if text:
                    units.append((page_num + 1, text))
        return units

    def _build_chunks(self, units: list[tuple[int, str]]) -> list[tuple[str, int, int]]:
        """Fasst Blöcke zu ~CHUNK_SIZE großen Chunks mit Überlappung zusammen.

        Rückgabe: Liste aus (Chunk-Text, Startseite, Endseite).
        """
        # Überlange Einzelblöcke vorab rekursiv zerteilen
        normalized: list[tuple[int, str]] = []
        for page, text in units:
            if len(text) > CHUNK_SIZE:
                normalized.extend((page, piece) for piece in _recursive_split(text))
            else:
                normalized.append((page, text))

        chunks: list[tuple[str, int, int]] = []
        current: list[tuple[int, str]] = []
        current_len = 0

        def emit_current() -> list[tuple[int, str]]:
            """Gibt den aktuellen Chunk aus und liefert die Überlappungs-Blöcke zurück."""
            text = "\n".join(block_text for _, block_text in current)
            chunks.append((text, current[0][0], current[-1][0]))

            # Die letzten Blöcke bis CHUNK_OVERLAP Zeichen in den nächsten Chunk übernehmen
            tail: list[tuple[int, str]] = []
            tail_len = 0
            for page, block_text in reversed(current):
                if tail_len + len(block_text) <= CHUNK_OVERLAP:
                    tail.insert(0, (page, block_text))
                    tail_len += len(block_text)
                else:
                    if not tail:
                        tail.insert(0, (page, block_text[-CHUNK_OVERLAP:]))
                    break
            return tail

        for page, text in normalized:
            if current and current_len + len(text) > CHUNK_SIZE:
                current = emit_current()
                current_len = sum(len(block_text) for _, block_text in current)
            current.append((page, text))
            current_len += len(text)

        if current:
            emit_current()

        return chunks

    # ==========================================
    # Bilder
    # ==========================================

    def _extract_images(self, doc, filename: str) -> int:
        """Speichert die Bilder eines PDFs auf der Festplatte (für spätere Bildbeschreibung).

        Es werden bewusst KEINE Platzhalter-Chunks in die Vektordatenbank geschrieben,
        weil Text wie "[BILD] Abbildung auf Seite 5" das Retrieval nur verschmutzt.
        """
        saved = 0
        seen_xrefs = set()

        for page_num in range(len(doc)):
            for img_index, img_info in enumerate(doc[page_num].get_images(full=True)):
                xref = img_info[0]
                # Dasselbe Bild (z. B. Logo auf jeder Seite) nur einmal speichern
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)

                try:
                    base_image = doc.extract_image(xref)
                    if base_image["width"] < MIN_IMAGE_SIZE or base_image["height"] < MIN_IMAGE_SIZE:
                        continue

                    image = Image.open(io.BytesIO(base_image["image"]))
                    image_filename = f"{filename}_p{page_num + 1}_img_{img_index}.{base_image['ext']}"
                    image.save(self.image_dir / image_filename)
                    saved += 1
                except Exception as e:
                    print(f"Fehler beim Speichern eines Bildes auf Seite {page_num + 1}: {e}")

        return saved

    # ==========================================
    # Ingest / Verwaltung
    # ==========================================

    def process_pdf(self, filename: str) -> dict:
        """Verarbeitet genau ein PDF aus dem PDF-Ordner und indexiert es in ChromaDB.

        Vorhandene Chunks desselben Dokuments (Text, Summary, Alt-Bestände) werden
        vorher gelöscht, damit keine verwaisten Einträge zurückbleiben.
        """
        filepath = self.pdf_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"PDF nicht gefunden: {filepath}")

        print(f"\nVerarbeite: {filename} ...")
        doc = fitz.open(filepath)

        units = self._extract_text_units(doc)
        chunks = self._build_chunks(units)
        images_saved = self._extract_images(doc, filename)
        page_count = len(doc)
        doc.close()

        with DB_WRITE_LOCK:
            # Alte Einträge dieses Dokuments entfernen (verhindert verwaiste Chunks)
            self.collection.delete(where={"source": filename})

            if chunks:
                self.collection.upsert(
                    documents=[text for text, _, _ in chunks],
                    metadatas=[
                        {
                            "source": filename,
                            "type": "text",
                            "page": page_start,
                            "page_end": page_end,
                        }
                        for _, page_start, page_end in chunks
                    ],
                    ids=[f"{filename}_c{index}" for index in range(len(chunks))],
                )

        stats = {"chunks": len(chunks), "pages": page_count, "images": images_saved}
        print(
            f"'{filename}': {stats['chunks']} Chunks aus {stats['pages']} Seiten indexiert, "
            f"{stats['images']} Bilder extrahiert."
        )
        return stats

    def process_all_pdfs(self):
        """Verarbeitet alle PDFs im PDF-Ordner (für Batch-/CLI-Nutzung)."""
        pdf_files = [f for f in os.listdir(self.pdf_dir) if f.endswith(".pdf")]

        if not pdf_files:
            print(f"Keine PDFs im Ordner '{self.pdf_dir}' gefunden.")
            return

        for filename in pdf_files:
            self.process_pdf(filename)

        print("\nVerarbeitung abgeschlossen! Daten sind bereit für die RAG-Pipeline.")

    def delete_pdf(self, filename: str):
        """Entfernt ein Dokument vollständig: DB-Chunks, PDF-Datei und extrahierte Bilder."""
        with DB_WRITE_LOCK:
            self.collection.delete(where={"source": filename})

        pdf_path = self.pdf_dir / filename
        pdf_path.unlink(missing_ok=True)

        for image_path in self.image_dir.glob(f"{filename}_p*"):
            image_path.unlink()

        print(f"'{filename}' wurde aus Datenbank und Dateisystem entfernt.")

    def list_sources(self) -> list[str]:
        """Listet die Dateinamen aller aktuell indexierten Dokumente."""
        data = self.collection.get(include=["metadatas"])
        return sorted({meta["source"] for meta in data["metadatas"] if meta.get("source")})


# --- Ausführung ---
if __name__ == "__main__":
    processor = PDFProcessor()
    processor.process_all_pdfs()
