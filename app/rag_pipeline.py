import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

# Lade Umgebungsvariablen
load_dotenv(find_dotenv(), override=True)


class GWDGRagPipeline:
    def __init__(self, persist_directory="./chroma_db"):
        api_key = os.getenv("SAIA_API_KEY")
        base_url = os.getenv("SAIA_BASE_URL")
        self.model_name = os.getenv("SAIA_MODEL")

        if not api_key or not base_url or not self.model_name:
            raise ValueError("Fehlende SAIA Umgebungsvariablen in der .env Datei!")

        # GWDG LLM Client
        self.llm_client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        # Verbinde mit der lokalen Chroma-Datenbank, die von pdf_processing.py gefüllt wurde
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

    def retrieve_relevant_chunks(self, question: str, max_chunks: int = 5) -> list[str]:
        """Sucht in der ChromaDB nach den relevantesten Chunks und fügt Seiten-Metadaten hinzu."""
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_texts=[question],
            n_results=max_chunks
        )

        if not results["documents"] or not results["documents"][0]:
            return []

        formatted_chunks = []
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        # Verbinde jeden Text-Chunk mit seiner Seitenzahl und dem Dateinamen
        for doc, meta in zip(documents, metadatas):
            page = meta.get("page", "Unbekannt")
            source = meta.get("source", "Unbekannt")

            # Wir rahmen den Text für das LLM klar ein
            chunk_with_meta = f"--- ABSCHNITT START (Quelle: {source}, Seite: {page}) ---\n{doc}\n--- ABSCHNITT ENDE ---"
            formatted_chunks.append(chunk_with_meta)

        return formatted_chunks

    def generate_answer(self, question: str, context_chunks: list[str]) -> str:
        """Sendet die Frage und die mit Metadaten versehenen Chunks an das Modell."""
        context_text = "\n\n".join(context_chunks)

        system_prompt = (
            "You are a highly qualified academic assistant and subject matter expert for analyzing corporate and sustainability reports. "
            "Your task is to answer user questions precisely, objectively, and factually.\n\n"
            "STRICTLY adhere to the following rules:\n"
            "1. FACT-BASED ONLY: Answer the question SOLELY based on the provided PDF context below. Do not use any external knowledge.\n"
            "2. ANTI-HALLUCINATION: If the provided context does not contain sufficient information to answer the question, do not guess. Reply exactly with: 'The provided document does not contain information on this topic.' (or the equivalent in the user's language).\n"
            "3. STYLE & TONE: Formulate your answers in a clean, professional, and objective academic tone. Use complete sentences.\n"
            "4. MULTILINGUAL (CRITICAL): You MUST answer in the EXACT SAME LANGUAGE as the user's question. If the user asks in English, write your entire response in English. If the user asks in German, write your entire response in German. Match the language perfectly.\n"
            "5. IN-TEXT CITATIONS: Each context section is provided with a document name and page number. You MUST cite your sources directly in the text. Place the corresponding source immediately at the end of the respective sentence in parentheses, using exactly this format: (Source: [Document Name], Page [X]). NEVER just list the sources at the end of your response!"
        )

        user_prompt = f"Context from the document:\n{context_text}\n\nUser question: {question}"

        response = self.llm_client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )

        return response.choices[0].message.content

    def answer_question_with_rag(self, question: str) -> str:
        """Hauptfunktion, die von Streamlit aufgerufen wird."""
        relevant_chunks = self.retrieve_relevant_chunks(question)

        if not relevant_chunks:
            return "Es konnten keine relevanten Informationen in der Datenbank gefunden werden. Hast du pdf_processing.py ausgeführt?"

        return self.generate_answer(question, relevant_chunks)


# --- Instanz für den Import in Streamlit ---
# Wenn Streamlit diese Datei importiert, kann es direkt 'pipeline.answer_question_with_rag()' nutzen
pipeline = GWDGRagPipeline()