import json
import re

# Felder des Sustainability-Exports; leerer String, wenn das Dokument nichts dazu enthält
EXPORT_FIELDS = [
    "name",
    "CO2",
    "NOX",
    "Number_of_Electric_Vehicles",
    "Impact",
    "Risks",
    "Opportunities",
    "Strategy",
    "Actions",
    "Adopted_policies",
    "Targets",
]

# Gezielte Suchanfragen, um die für die Felder relevanten Stellen einzusammeln
_FIELD_QUERIES = (
    "CO2 carbon emissions figures",
    "NOx nitrogen oxide emissions",
    "electric vehicles fleet",
    "environmental and social impact",
    "risks and opportunities",
    "sustainability strategy, actions, policies, and targets",
)


def _as_string(value) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False)


def create_json_export(pipeline, source: str) -> dict:
    """Extrahiert strukturierte Nachhaltigkeitsdaten eines Dokuments.

    Nutzt die gespeicherte Dokument-Zusammenfassung plus gezielte Retrievals als
    Kontext und lässt das LLM die Export-Felder als striktes JSON befüllen.
    """
    context_parts = [text for _, text in pipeline._get_stored_summaries(source)]
    seen = set(context_parts)
    for query in _FIELD_QUERIES:
        for chunk in pipeline.retrieve_relevant_chunks(query, max_chunks=4, source=source):
            if chunk["text"] not in seen:
                seen.add(chunk["text"])
                context_parts.append(chunk["text"])

    context = "\n\n".join(context_parts)
    field_list = ", ".join(EXPORT_FIELDS)

    response = pipeline.llm_client.chat.completions.create(
        model=pipeline.model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You extract structured sustainability data from report excerpts. "
                    "Reply with ONE valid JSON object only - no markdown, no explanations."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Fill this JSON template with data about the company from the report "
                    f"excerpts below. Keys: {field_list}. "
                    "Use concise strings (numbers with units where given). "
                    'Use an empty string "" for fields the excerpts do not cover.'
                    f"\n\nReport excerpts:\n{context}"
                ),
            },
        ],
        temperature=0.0,
    )

    raw = (response.choices[0].message.content or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        result = {field: "" for field in EXPORT_FIELDS}
        result["_error"] = "The language model returned invalid JSON."
        return result

    # Nur die erwarteten Felder in fester Reihenfolge zurückgeben
    return {field: _as_string(data.get(field, "")) for field in EXPORT_FIELDS}


def convert_json_export_to_string(export_data: dict | list) -> str:
    """Serialisiert den Export formatiert für den Download-Button."""
    return json.dumps(export_data, indent=2, ensure_ascii=False)


def create_chat_history_export(messages: list[dict]) -> list[dict]:
    """Paart jede Nutzerfrage mit der direkt folgenden Assistenzantwort für den Export.

    Nimmt st.session_state.messages entgegen (Liste aus {"role", "content", "sources"}).
    """
    turns = []
    pending_question = None
    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if role == "user":
            pending_question = content
        elif role == "assistant" and pending_question is not None:
            turns.append(
                {
                    "question": pending_question,
                    "answer": content,
                    "sources": [
                        {"source": entry.get("source"), "page": entry.get("page")}
                        for entry in (message.get("sources") or [])
                    ],
                }
            )
            pending_question = None
    return turns
