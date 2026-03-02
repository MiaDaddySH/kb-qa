# Enterprise KB-QA System

Enterprise-ready Knowledge Base Question Answering (RAG) system built with FastAPI and Azure OpenAI.

---

## 🚀 Overview

This project implements a production-style backend service for enterprise knowledge base question answering.

It integrates:

- Azure OpenAI (Chat + Embeddings)
- Retrieval-Augmented Generation (RAG)
- FastAPI REST API
- Environment-based configuration
- Clean modular project structure

The goal of this project is to demonstrate real-world AI application engineering practices suitable for enterprise environments.

---

## 🏗 Architecture

High-level architecture:

User → FastAPI API → Retrieval Layer → Azure OpenAI → Response

Planned full RAG pipeline:

1. Document Upload  
2. Text Extraction  
3. Chunking  
4. Embedding Generation  
5. Vector Storage (pgvector)  
6. Similarity Search  
7. Context-augmented LLM Response  

---

## 🧰 Tech Stack

- Python 3.11  
- FastAPI  
- Pydantic  
- Azure OpenAI  
- Uvicorn  
- python-dotenv  

Planned additions:

- PostgreSQL  
- pgvector  
- Docker  
- Authentication layer  

---

## 📦 Project Structure

    kb-qa/
    │
    ├── app/
    │   └── main.py
    │
    ├── .env.example
    ├── requirements.txt
    ├── LICENSE
    └── README.md

---

## ⚙️ Setup & Run

### 1️⃣ Clone repository

    git clone https://github.com/<your-username>/kb-qa.git
    cd kb-qa

### 2️⃣ Create virtual environment

    python -m venv .venv
    source .venv/bin/activate

### 3️⃣ Install dependencies

    pip install -r requirements.txt

### 4️⃣ Configure environment variables

    cp .env.example .env

Fill in:

- AZURE_OPENAI_API_KEY  
- AZURE_OPENAI_ENDPOINT  
- AZURE_OPENAI_DEPLOYMENT  

### 5️⃣ Start the server

    uvicorn app.main:app --reload

Visit:

    http://127.0.0.1:8000/docs

---

## 📡 API Endpoints

### GET /health

Health check endpoint.

---

### POST /chat

Request body:

    {
      "question": "What is the capital of France?"
    }

Response:

    {
      "answer": "Paris"
    }

---

## 🧠 Why This Project?

This repository demonstrates:

- Enterprise-ready AI integration  
- Clean backend architecture  
- Proper dependency management  
- Secure configuration handling  
- Practical Azure OpenAI usage  

It is designed as a portfolio-level project for AI engineering roles.

---

## 🗺 Roadmap

- [x] Azure OpenAI Chat integration  
- [x] GitHub project setup  
- [x] Environment configuration management  
- [ ] Embedding integration  
- [ ] PostgreSQL + pgvector  
- [ ] Document upload API  
- [ ] Full RAG pipeline  
- [ ] Docker support  
- [ ] Authentication & multi-tenant support  

---

## 📜 License

MIT License