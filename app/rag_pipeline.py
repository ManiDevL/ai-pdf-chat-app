import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
import chromadb

# Lade Umgebungsvariablen (API Key aus der .env Datei)
load_dotenv()


class GWDGRagPipeline:
    def __init__(self, persist_directory="./chroma_db"):
        # Sucht die .env Datei automatisch im Hauptverzeichnis
        load_dotenv(find_dotenv())

        # 1. Wir laden alle drei Werte exakt so, wie sie in der .env stehen
        api_key = os.getenv("SAIA_API_KEY")
        base_url = os.getenv("SAIA_BASE_URL")
        self.model_name = os.getenv("SAIA_MODEL")  # Speichern wir für die generate_answer Funktion

        # Sicherheits-Check
        if not api_key:
            raise ValueError("SAIA_API_KEY fehlt in der .env Datei!")

        # 2. Wir übergeben sie an OpenAI mit den Namen, die OpenAI erwartet!
        self.llm_client = OpenAI(
            api_key=api_key,  # OpenAI erwartet hier 'api_key'
            base_url=base_url  # OpenAI erwartet hier 'base_url'
        )

        # 3. Lokale Datenbank (ChromaDB)
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.chroma_client.get_or_create_collection(name="eco_reports")

    def generate_answer(self, query, context_chunks):
        """Sendet die echte Anfrage an den GWDG Server."""
        context_text = "\n\n---\n\n".join(context_chunks)

        system_prompt = (
            "Du bist ein analytischer Experte für Nachhaltigkeitsberichte. "
            "Beantworte die Frage nur basierend auf dem Kontext.\n\n"
            f"KONTEXT:\n{context_text}"
        )

        # HIER WIEDER DEN ECHTEN API-CALL NUTZEN:
        response = self.llm_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content

    def add_chunks_to_db(self, chunks, metadata_list, ids):
        """Speichert die zerkleinerten PDF-Texte (Chunks) in der Datenbank."""
        self.collection.upsert(
            documents=chunks,
            metadatas=metadata_list,
            ids=ids
        )
        print(f"Erfolgreich {len(chunks)} Chunks zur Datenbank hinzugefügt.")

    def retrieve(self, query, n_results=3):
        """Sucht in der Datenbank nach den relevantesten Textabschnitten zur Frage."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        # results["documents"][0] enthält eine Liste der gefundenen Texte
        return results["documents"][0] if results["documents"] else []


    def ask(self, query):
        """Die Hauptfunktion: Orchestriert den RAG-Prozess (Retrieve -> Generate)."""
        print(f"Suche nach relevanten Abschnitten für: '{query}'...")
        context = self.retrieve(query)

        if not context:
            return "Keine relevanten Informationen in der Datenbank gefunden."

        print("Generiere Antwort über GWDG Server...")
        answer = self.generate_answer(query, context)
        return answer


# --- Test-Bereich (wird nur ausgeführt, wenn ihr diese Datei direkt startet) ---
if __name__ == "__main__":
    pipeline = GWDGRagPipeline()

    # 1. Wir simulieren Person B (PDF Processing) und werfen Dummy-Daten in die DB
    print("Fülle Datenbank mit Dummy-Daten...")
    pipeline.add_chunks_to_db(
        chunks=[
            "Im Jahr 2023 hat die Firma EcoCorp ihre CO2-Emissionen um 15% auf 10.000 Tonnen gesenkt.",
            "Die Flotte der EcoCorp besteht aktuell aus 500 Elektrofahrzeugen (Number_of_Electric_Vehicles).",
            "Ein großes Risiko für das Unternehmen ist die Wasserknappheit in den Produktionsländern."
        ],
        metadata_list=[{"source": "ecocorp_report.pdf"}, {"source": "ecocorp_report.pdf"},
                       {"source": "ecocorp_report.pdf"}],
        ids=["chunk_1", "chunk_2", "chunk_3"]
    )

    # 2. Wir testen eine Anfrage
    frage = "Wie viele Elektrofahrzeuge besitzt EcoCorp?"
    antwort = pipeline.ask(frage)

    print("\n" + "=" * 50)
    print("FRAGE:", frage)
    print("ANTWORT:\n", antwort)
    print("=" * 50)
