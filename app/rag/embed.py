# app/rag/embed.py
import os
from typing import List, Optional
from openai import OpenAI

_client: Optional[OpenAI] = None

def _get_client() -> OpenAI:
    """
    Lazy-init Azure OpenAI client after env vars are loaded.
    """
    global _client
    if _client is not None:
        return _client

    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")

    if not azure_endpoint or not azure_key:
        raise RuntimeError("Missing AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY")

    _client = OpenAI(
        api_key=azure_key,
        base_url=f"{azure_endpoint.rstrip('/')}/openai/v1/",
    )
    return _client

def embed_texts(texts: List[str]) -> List[List[float]]:
    deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
    client = _get_client()
    resp = client.embeddings.create(
        model=deployment,  # Azure: deployment name
        input=texts,
    )
    return [item.embedding for item in resp.data]

def embed_text(text: str) -> List[float]:
    return embed_texts([text])[0]