import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

# Lade die Umgebungsvariablen
load_dotenv(find_dotenv())

# Verbinde dich mit dem GWDG Server
client = OpenAI(
    api_key=os.getenv("SAIA_API_KEY"),
    base_url=os.getenv("SAIA_BASE_URL")
)

print("Frage GWDG-Server nach verfügbaren Modellen...\n")

# Rufe die Liste aller Modelle ab
try:
    models = client.models.list()
    print("Der Server hat aktuell folgende Modelle geladen:")
    print("-" * 40)
    for model in models.data:
        print(f"- {model.id}")
    print("-" * 40)
except Exception as e:
    print(f"Fehler beim Abrufen: {e}")