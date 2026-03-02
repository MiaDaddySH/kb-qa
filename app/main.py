import os
from dotenv import load_dotenv
from fastapi import FastAPI, Body
from pydantic import BaseModel
from openai import OpenAI
from app.rag.embed import embed_text
from app.rag.store import insert_chunk, search_similar

load_dotenv()

app = FastAPI(title="KB-QA MVP (Azure OpenAI)")

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

if not AZURE_ENDPOINT or not AZURE_KEY or not DEPLOYMENT:
    # 让你一眼知道是不是 .env 没读到
    missing = [k for k, v in {
        "AZURE_OPENAI_ENDPOINT": AZURE_ENDPOINT,
        "AZURE_OPENAI_API_KEY": AZURE_KEY,
        "AZURE_OPENAI_DEPLOYMENT": DEPLOYMENT,
    }.items() if not v]
    raise RuntimeError(f"Missing env vars: {missing}")

client = OpenAI(
    api_key=AZURE_KEY,
    base_url=f"{AZURE_ENDPOINT.rstrip('/')}/openai/v1/",
)

class ChatRequest(BaseModel):
    question: str

@app.get("/health")
def health():
    return {"status": "ok", "using": "azure-openai"}

@app.post("/chat")
def chat(payload: ChatRequest = Body(...)):
    completion = client.chat.completions.create(
        model=DEPLOYMENT,  # Azure: 这里必须是部署名
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": payload.question},
        ],
        temperature=0.7,
    )
    return {"answer": completion.choices[0].message.content}

class EmbedTestRequest(BaseModel):
    text: str

@app.post("/embed_test")
def embed_test(payload: EmbedTestRequest = Body(...)):
    vec = embed_text(payload.text)
    return {
        "dim": len(vec),
        "preview": vec[:8],   # 只返回前几个数字，避免返回太大
    }


class IngestTestRequest(BaseModel):
    workspace_id: str = "demo"
    doc_id: str = "doc1"
    chunk_index: int = 0
    content: str

@app.post("/ingest_test")
def ingest_test(payload: IngestTestRequest = Body(...)):
    vec = embed_text(payload.content)
    insert_chunk(
        workspace_id=payload.workspace_id,
        doc_id=payload.doc_id,
        chunk_index=payload.chunk_index,
        content=payload.content,
        embedding=vec,
        metadata={"source": "manual"},
    )
    return {"status": "inserted", "dim": len(vec)}

class SearchTestRequest(BaseModel):
    workspace_id: str = "demo"
    query: str

@app.post("/search_test")
def search_test(payload: SearchTestRequest = Body(...)):
    qvec = embed_text(payload.query)
    hits = search_similar(payload.workspace_id, qvec, top_k=3)
    return {"hits": hits}

class RagChatRequest(BaseModel):
    workspace_id: str = "demo"
    question: str
    top_k: int = 3

@app.post("/rag_chat")
def rag_chat(payload: RagChatRequest = Body(...)):
    # 1) Embed the question
    qvec = embed_text(payload.question)

    # 2) Retrieve top-k chunks
    hits = search_similar(payload.workspace_id, qvec, top_k=payload.top_k)

    # 3) Build evidence context (numbered, easy to cite)
    evidence_lines = []
    for i, h in enumerate(hits, start=1):
        evidence_lines.append(f"[{i}] (doc={h['doc_id']}, chunk={h['chunk_index']}) {h['content']}")
    evidence = "\n".join(evidence_lines) if evidence_lines else "(no evidence found)"

    # 4) Ask Azure OpenAI (use your existing client / deployment)
    #    如果你已经在 main.py 里有 client 和 DEPLOYMENT，就直接用
    completion = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an enterprise knowledge base assistant.\n"
                    "Answer ONLY using the provided EVIDENCE.\n"
                    "If the evidence is insufficient, say you don't know.\n"
                    "Cite sources in the form [1], [2] referring to evidence items."
                ),
            },
            {"role": "user", "content": f"QUESTION:\n{payload.question}\n\nEVIDENCE:\n{evidence}"},
        ],
        temperature=0.2,
    )

    answer = completion.choices[0].message.content

    return {
        "answer": answer,
        "citations": hits,   # 调试用：把命中的 chunks 返回
        "evidence": evidence # 调试用：你可以在响应里看到上下文是什么
    }