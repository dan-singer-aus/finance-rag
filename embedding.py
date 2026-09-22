from functools import cache

from dotenv import load_dotenv
from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"


@cache
def _client() -> OpenAI:
    # Loaded here, not at module scope, so importing this module stays free —
    # the same reason the client itself is lazy.
    load_dotenv()
    return OpenAI()


def embed(texts: list[str]) -> list[list[float]]:
    client = _client()
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [
        item.embedding for item in sorted(response.data, key=lambda item: item.index)
    ]
