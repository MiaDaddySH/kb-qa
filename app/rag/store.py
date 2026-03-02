# app/rag/store.py
from __future__ import annotations

from typing import List, Dict, Any
import json

from psycopg.types.json import Jsonb
from app.db.pg import get_conn

def insert_chunk(
    workspace_id: str,
    doc_id: str,
    chunk_index: int,
    content: str,
    embedding: List[float],
    metadata: Dict[str, Any] | None = None,
):
    metadata = metadata or {}

    print("[insert_chunk] metadata type:", type(metadata), "keys:", list(metadata.keys())[:5])

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chunks (workspace_id, doc_id, chunk_index, content, embedding, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    workspace_id,
                    doc_id,
                    chunk_index,
                    content,
                    embedding,
                    Jsonb(metadata),  # ✅ JSONB 用 Jsonb
                ),
            )

def search_similar(workspace_id: str, query_embedding: List[float], top_k: int = 5):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, doc_id, chunk_index, content, metadata,
                       (embedding <=> (%s)::vector) AS distance
                FROM chunks
                WHERE workspace_id = %s
                ORDER BY embedding <=> (%s)::vector
                LIMIT %s
                """,
                (query_embedding, workspace_id, query_embedding, top_k),
            )
            rows = cur.fetchall()

    results = []
    for r in rows:
        meta = r[4]
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {"_raw": meta}

        results.append(
            {
                "id": r[0],
                "doc_id": r[1],
                "chunk_index": r[2],
                "content": r[3],
                "metadata": meta,
                "distance": float(r[5]),
            }
        )
    return results