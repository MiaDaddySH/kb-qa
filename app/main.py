import os
from dotenv import load_dotenv
from fastapi import FastAPI, Body
from pydantic import BaseModel
from openai import OpenAI

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