from functools import cache

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"

@cache
def _client() -> OpenAI:
    return OpenAI()

def embed(texts: list[str]) -> list[list[float]]:
    client = _client()
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
