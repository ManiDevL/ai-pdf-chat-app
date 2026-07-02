import os
import re

from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

from pdf_processing import chunk_text


# Load environment variables from the .env file in the project root.
# override=True makes sure the latest values from .env are used.
# Chunking is intentionally imported from pdf_processing.py, because PDF extraction
# and text preparation belong together.

load_dotenv(find_dotenv(), override=True)


def retrieve_relevant_chunks(question: str, chunks: list[str], max_chunks: int = 4) -> list[str]:
    """
    Select the most relevant chunks for the user's question.

    This is a simple prototype retrieval method:
    - Convert the question into words.
    - Convert each chunk into words.
    - Count how many words overlap.
    - Return the chunks with the highest overlap score.

    Later, this can be replaced by embeddings or ChromaDB for a more advanced RAG setup.
    """
    if not question or not chunks:
        return []

    question_words = set(re.findall(r"\w+", question.lower()))
    scored_chunks = []

    for chunk in chunks:
        chunk_words = set(re.findall(r"\w+", chunk.lower()))
        score = len(question_words.intersection(chunk_words))
        scored_chunks.append((score, chunk))

    # Sort chunks by relevance score, highest score first.
    scored_chunks.sort(key=lambda item: item[0], reverse=True)

    # Keep only chunks that have at least one matching word with the question.
    relevant_chunks = [chunk for score, chunk in scored_chunks[:max_chunks] if score > 0]

    # Fallback: if no keyword match is found, use the first chunks of the document.
    if relevant_chunks:
        return relevant_chunks

    return chunks[:max_chunks]


def generate_answer(question: str, context_chunks: list[str]) -> str:
    """
    Send the user's question and the retrieved PDF chunks to the SAIA/Gemma API.

    This function replaces the old llm_chat.py logic.
    The LLM call now belongs inside the RAG pipeline because the model should answer
    based on the retrieved chunks, not on the full raw PDF text.
    """
    api_key = os.getenv("SAIA_API_KEY")
    base_url = os.getenv("SAIA_BASE_URL")
    model = os.getenv("SAIA_MODEL")

    # Safety checks for missing .env values.
    if not api_key:
        return "Error: SAIA_API_KEY is missing in the .env file."

    if not base_url:
        return "Error: SAIA_BASE_URL is missing in the .env file."

    if not model:
        return "Error: SAIA_MODEL is missing in the .env file."

    # The OpenAI client is used only as a compatible API client.
    # The request is sent to the SAIA/GWDG endpoint from SAIA_BASE_URL.
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    # Combine the selected chunks into one context block for the model.
    context_text = "\n\n---\n\n".join(context_chunks)

    prompt = f"""
    You are an assistant for analyzing sustainability reports.
    Answer the user's question based only on the provided PDF context.

    PDF context:
    {context_text}

    User question:
    {question}
    """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You answer questions about uploaded PDF documents."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    return response.choices[0].message.content


def answer_question_with_rag(question: str, document_text: str) -> str:
    """
    Main function used by the Streamlit app.

    Full flow:
    1. Take the extracted PDF text.
    2. Split it into chunks.
    3. Retrieve the most relevant chunks for the user's question.
    4. Send only those chunks to SAIA/Gemma.
    5. Return the generated answer.
    """
    chunks = chunk_text(document_text)
    relevant_chunks = retrieve_relevant_chunks(question, chunks)

    if not relevant_chunks:
        return "No relevant information could be found in the uploaded PDF."

    return generate_answer(question, relevant_chunks)
