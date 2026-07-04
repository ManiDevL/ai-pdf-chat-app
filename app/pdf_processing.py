import os
import chromadb
import fitz  # Das ist PyMuPDF
from PIL import Image
import io
from chromadb.utils import embedding_functions


class PDFProcessor:
    def __init__(self, pdf_dir="./pdfs", image_dir="./extracted_images", persist_directory="./chroma_db"):
        self.pdf_dir = pdf_dir
        self.image_dir = image_dir

        # Ordnerstrukturen sicherstellen
        for directory in [self.pdf_dir, self.image_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Ordner '{directory}' wurde erstellt.")

        # Datenbankverbindung
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        # Definiere das mehrsprachige Modell
        multilingual_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )

        # Übergebe das Modell an die Datenbank
        self.collection = self.chroma_client.get_or_create_collection(
            name="eco_reports",
            embedding_function=multilingual_ef
        )

    def process_all_pdfs(self):
        """Durchläuft alle PDFs und wendet text- und bildspezifisches Chunking an."""
        pdf_files = [f for f in os.listdir(self.pdf_dir) if f.endswith(".pdf")]

        if not pdf_files:
            print(f"Keine PDFs im Ordner '{self.pdf_dir}' gefunden.")
            return

        all_chunks = []
        all_metadatas = []
        all_ids = []

        chunk_counter = 0

        for filename in pdf_files:
            filepath = os.path.join(self.pdf_dir, filename)
            print(f"\nVerarbeite: {filename} ...")

            # Öffne die PDF mit PyMuPDF
            doc = fitz.open(filepath)

            for page_num in range(len(doc)):
                page = doc[page_num]

                # ==========================================
                # 1. HIERARCHICAL CHUNKING (Text)
                # ==========================================
                # PyMuPDF's "blocks" erkennt Absätze und Layout-Blöcke automatisch
                text_blocks = page.get_text("blocks")

                for block_index, block in enumerate(text_blocks):
                    # block[4] enthält den eigentlichen Text des Absatzes
                    text_content = block[4].strip()

                    if text_content:
                        all_chunks.append(text_content)
                        # Hierarchie in den Metadaten abbilden:
                        all_metadatas.append({
                            "source": filename,
                            "type": "text",
                            "page": page_num + 1,
                            "hierarchy_level": "paragraph",
                            "block_index": block_index
                        })
                        all_ids.append(f"{filename}_p{page_num + 1}_text_{block_index}")
                        chunk_counter += 1

                # ==========================================
                # 2. MODALITY SPECIFIC CHUNKING (Bilder/Grafiken)
                # ==========================================
                image_list = page.get_images(full=True)

                for img_index, img_info in enumerate(image_list):
                    xref = img_info[0]  # Referenznummer des Bildes in der PDF
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    # Bild lokal speichern
                    image_filename = f"{filename}_p{page_num + 1}_img_{img_index}.{image_ext}"
                    image_filepath = os.path.join(self.image_dir, image_filename)

                    try:
                        image = Image.open(io.BytesIO(image_bytes))
                        image.save(image_filepath)

                        # Stellvertreter-Text für die Vektordatenbank erstellen
                        # (Idealfall: Hier eine KI das Bild beschreiben lassen!)
                        image_description = (
                            f"[BILD/GRAFIK] Eine Abbildung aus dem Dokument '{filename}', "
                            f"befindlich auf Seite {page_num + 1}. "
                            f"Dateipfad: {image_filepath}"
                        )

                        all_chunks.append(image_description)
                        all_metadatas.append({
                            "source": filename,
                            "type": "image",
                            "page": page_num + 1,
                            "image_path": image_filepath
                        })
                        all_ids.append(f"{filename}_p{page_num + 1}_img_{img_index}")
                        chunk_counter += 1

                    except Exception as e:
                        print(f"Fehler beim Speichern des Bildes auf Seite {page_num + 1}: {e}")

        # ==========================================
        # 3. UPSERT IN DIE DATENBANK
        # ==========================================
        if all_chunks:
            print(f"\nSpeichere {chunk_counter} strukturierte Chunks in die Datenbank...")
            self.collection.upsert(
                documents=all_chunks,
                metadatas=all_metadatas,
                ids=all_ids
            )
            print("Verarbeitung abgeschlossen! Daten sind bereit für die RAG-Pipeline.")


# --- Ausführung ---
if __name__ == "__main__":
    processor = PDFProcessor()
    processor.process_all_pdfs()