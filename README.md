# Production-ready AI Chatbot Backend (FastAPI + Ollama + RAG + Scraping + Postgres)

## Prereqs
- Python 3.10+
- Ollama running locally with `llama3` pulled
- PostgreSQL running locally

## Setup
1. Create a virtualenv and install deps:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2. Install Playwright browser (needed for scraping JS pages):

```bash
python -m playwright install chromium
```

2. Configure env:
- Copy `.env.example` to `.env` and edit values as needed.

3. Ensure Postgres database exists (example):
- DB name: `chatbot`

## Build the RAG vector DB

```bash
python ingest.py https://pokaktech.com/
```

You should see:
- `[crawler] pages_scraped=...`
- `[ingest] vector DB built ...`

## Run the API

```bash
uvicorn main:app --reload
```

## Chat API
`POST /chat`

Request:
```json
{
  "message": "Hi, I need a website and want pricing. My name is John Doe, email john@acme.com",
  "session_id": "abc123"
}
```

Response:
```json
{
  "response": "..."
}
```

