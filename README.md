# Industrial Maintenance Manual RAG System

An AI-powered system that allows field technicians to describe machine faults in plain language and receive instant, cited repair instructions extracted from official maintenance manuals. Supports multiple languages including Tamil, Hindi, Arabic, and 50+ others.

---

## Demo

> Upload a PDF maintenance manual, describe a fault in any language, and receive structured repair instructions with source citations in seconds.

---

## Features

- **PDF Manual Upload** — Upload any industrial maintenance manual directly from the browser
- **Hybrid Search** — Combines BM25 keyword search and vector semantic search using Reciprocal Rank Fusion
- **Multilingual Support** — Ask questions in Tamil, Hindi, Arabic, French, or any language and receive answers in the same language
- **Source Citations** — Every answer includes the exact manual page and section it was retrieved from
- **Manual Filter** — Restrict search to a specific uploaded manual
- **Query History** — All queries and feedback logged to SQLite database
- **REST API** — Full FastAPI backend with Swagger documentation

---

## Architecture

PDF Manuals
↓
Text Extraction (pypdf)
↓
Smart Chunking with Overlap
↓
Embeddings → ChromaDB Vector Store

Query (any language)
↓
Language Detection + Translation to English
↓
BM25 Keyword Search + Vector Semantic Search
↓
Reciprocal Rank Fusion (RRF)
↓
Groq LLM (llama-3.1-8b-instant) generates answer
↓
Answer translated back to user's language
↓
Structured response with source citations


---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq API — llama-3.1-8b-instant |
| Orchestration | LangChain |
| Vector Database | ChromaDB |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Keyword Search | BM25 (rank-bm25) |
| Result Ranking | Reciprocal Rank Fusion |
| PDF Processing | pypdf |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Translation | deep-translator + langdetect |
| Logging | SQLite + SQLAlchemy |
| Environment | python-dotenv |

---

## Project Structure

industrial-maintenance-rag/
├── ingestion/
│ ├── pdf_extractor.py # PDF text extraction
│ ├── chunker.py # Smart text chunking
│ ├── embedder.py # Embedding generation and storage
│ └── ingest_pipeline.py # Master ingestion runner
├── retrieval/
│ ├── vector_search.py # ChromaDB semantic search
│ ├── bm25_search.py # BM25 keyword search
│ ├── hybrid_retriever.py # RRF fusion of both searches
│ └── reranker.py # Final result selection
├── generation/
│ ├── prompt_templates.py # LLM prompt engineering
│ ├── llm_chain.py # Groq LLM integration
│ └── translator.py # Multilingual support
├── api/
│ ├── main.py # FastAPI application
│ ├── schemas.py # Pydantic request/response models
│ └── logger.py # Query and feedback logging
├── frontend/
│ └── app.py # Streamlit UI
├── data/
│ ├── raw_manuals/ # PDF manuals (not tracked in git)
│ ├── processed_chunks/ # Extracted chunks (not tracked in git)
│ └── chroma_db/ # Vector database (not tracked in git)
├── database/ # SQLite logs (not tracked in git)
├── requirements.txt
└── .env # Configuration (not tracked in git)


---

## Setup and Installation

### Prerequisites

- Python 3.11+
- Groq API key (free at console.groq.com)

### Step 1 — Clone the Repository

```bash
git clone https://github.com/subash2000-eng/industrial-maintenance-rag.git
cd industrial-maintenance-rag
```

### Step 2 — Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Configure Environment

Create a `.env` file in the project root:

GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_DB_PATH=./data/chroma_db
CHROMA_COLLECTION_NAME=maintenance_manuals
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K_RETRIEVAL=20
TOP_K_RERANK=3
API_HOST=0.0.0.0
API_PORT=8000
SQLITE_DB_PATH=./database/feedback.db


### Step 5 — Add PDF Manuals

Place PDF maintenance manuals in the `data/raw_manuals/` folder.

### Step 6 — Run Ingestion Pipeline

```bash
python ingestion/ingest_pipeline.py
```

### Step 7 — Build BM25 Index

```bash
python retrieval/bm25_search.py
```

### Step 8 — Start the API

```bash
python api/main.py
```

API available at: `http://localhost:8000`
Swagger docs at: `http://localhost:8000/docs`

### Step 9 — Start the Frontend

Open a new terminal:

```bash
streamlit run frontend/app.py
```

App available at: `http://localhost:8501`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | /health | System health check |
| POST | /query | Submit fault, receive repair instructions |
| POST | /upload-manual | Upload a PDF manual |
| GET | /manuals | List all ingested manuals |
| POST | /feedback | Submit rating for a response |
| GET | /history | View recent query history |

### Example Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "motor bearing overheating after 2 hours of operation"}'
```

### Example Response

```json
{
  "query_id": "q_3f8a2c9d1b4e",
  "query": "motor bearing overheating after 2 hours",
  "answer": "## Fault Diagnosis\nBearing overheating is likely caused by...",
  "sources": [
    {
      "source_num": 1,
      "manual_name": "SIEMENS_Motor_Manual",
      "page_number": 16,
      "section_title": "Motor Troubleshooting Chart",
      "confidence": "High"
    }
  ],
  "confidence": "High",
  "model": "llama-3.1-8b-instant",
  "chunks_used": 3,
  "response_time_s": 3.2,
  "detected_language": "English"
}
```

---

## Multilingual Support

The system automatically detects the input language and translates queries and responses.

Supported languages include Tamil, Hindi, Arabic, French, German, Spanish, Chinese, Japanese, Korean, Portuguese, Russian, Italian, and 50+ others.

Example Tamil query:

மோட்டார் தாங்கி சூடாகிறது, என்ன செய்வது?


The system will:
1. Detect the language as Tamil
2. Translate the query to English for searching
3. Retrieve relevant manual sections
4. Generate the answer in English
5. Translate the answer back to Tamil

---

## Skills Demonstrated

| Skill | Implementation |
|---|---|
| RAG Architecture | Full pipeline from ingestion to generation |
| Vector Databases | ChromaDB with cosine similarity search |
| LangChain | LLM orchestration and prompt management |
| LLM APIs | Groq API with llama-3.1-8b-instant |
| Hybrid Search | BM25 + Vector search with RRF fusion |
| FastAPI | Production REST API with Pydantic validation |
| Streamlit | Interactive frontend with dark industrial theme |
| Multilingual NLP | Language detection and translation pipeline |
| SQLite Logging | Query and feedback observability |
| PDF Processing | Text extraction and smart chunking |
| Python OOP | Modular, maintainable codebase |

---

## Author

**Subash T**
Data Science and AI Professional

- GitHub: [github.com/subash2000-eng](https://github.com/subash2000-eng)
- LinkedIn: [linkedin.com/in/subasht2000](https://linkedin.com/in/subasht2000)
- Email: subashsdr2000@gmail.com