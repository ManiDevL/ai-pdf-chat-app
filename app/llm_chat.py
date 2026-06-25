import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


def ask_llm(question: str, context: str) -> str:
    """
    Send the user's question and the extracted PDF context to the SAIA/Gemma API.
    """
    api_key = os.getenv("SAIA_API_KEY")
    base_url = os.getenv("SAIA_BASE_URL")
    model = os.getenv("SAIA_MODEL")

    if not api_key:
        return "Error: SAIA_API_KEY is missing in the .env file."

    if not base_url:
        return "Error: SAIA_BASE_URL is missing in the .env file."

    if not model:
        return "Error: SAIA_MODEL is missing in the .env file."

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    prompt = f"""
    You are an assistant for analyzing sustainability reports.
    Answer the user's question based only on the provided PDF context.

    PDF context:
    {context[:12000]}

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

