from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import chromadb
from dotenv import load_dotenv

from crawler import crawl
from utils import chunk_text, clean_text


class OllamaClient:
    def __init__(self, base_url: str, model: str, *, timeout_seconds: int = 180):
        import requests

        self._requests = requests
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def embed(self, text: str) -> list[float]:
        url = f"{self.base_url}/api/embeddings"
        r = self._requests.post(
            url,
            json={"model": self.model, "prompt": text},
            timeout=self.timeout_seconds,
        )
        r.raise_for_status()
        data = r.json()
        emb = data.get("embedding")
        if not isinstance(emb, list):
            raise RuntimeError("Invalid embedding response from Ollama")
        return [float(x) for x in emb]


class OllamaEmbeddingFunction:
    def __init__(self, client: OllamaClient):
        self._client = client

    def name(self) -> str:
        return f"ollama:{self._client.model}"

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return [self._client.embed(t) for t in input]

    def embed_query(self, input) -> list[list[float]]:
        if isinstance(input, str):
            return [self._client.embed(input)]
        if isinstance(input, list) and input:
            return [self._client.embed(input[0])]
        return [[]]

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.embed_documents(input)


def main() -> int:
    load_dotenv()

    if len(sys.argv) < 2:
        print("Usage: python ingest.py https://pokaktech.com/")
        return 2

    base_url = sys.argv[1].strip()
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./vectorstore")
    collection_name = os.getenv("CHROMA_COLLECTION", "docs")

    max_pages = int(os.getenv("CRAWL_MAX_PAGES", "50"))
    timeout_seconds = int(os.getenv("CRAWL_TIMEOUT_SECONDS", "15"))
    max_chunks = int(os.getenv("INGEST_MAX_CHUNKS", "40"))

    ollama_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3")
    ollama_timeout = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))
    ollama = OllamaClient(ollama_base, ollama_model, timeout_seconds=ollama_timeout)

    # Build documents
    docs: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    # Optional seed knowledge: include the brochure text file so the chatbot
    # has “extended company knowledge” even if some website sections are missing.
    include_seed = os.getenv("INCLUDE_BROCHURE_SEED", "true").strip().lower()
    include_seed = include_seed in ("1", "true", "yes", "y", "on")
    seed_path = os.getenv(
        "BROCHURE_SEED_PATH",
        "./seed_docs/pokak_enterprise_chatbot_brochure.txt",
    ).strip()

    if include_seed:
        sp = Path(seed_path)
        if sp.exists() and sp.is_file():
            seed_title = "Pokak Enterprise Chatbot Brochure"
            try:
                seed_text = clean_text(sp.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                seed_text = clean_text(sp.read_text(errors="ignore"))

            for chunk in chunk_text(seed_text, max_words=140, overlap_words=20):
                chunk = clean_text(chunk)
                if not chunk:
                    continue
                docs.append(chunk)
                metadatas.append({"source_url": "seed_docs://brochure", "title": seed_title})
                ids.append(str(uuid.uuid4()))
                if len(docs) >= max_chunks:
                    break
        else:
            print(f"[ingest] seed brochure not found: {sp.resolve()}")

    pages = crawl(base_url, max_pages=max_pages, timeout_seconds=timeout_seconds)
    for p in pages:
        text = clean_text(f"{p.title}\n{p.content}")
        # Prevent extremely long SPA-rendered pages from creating too many chunks.
        if len(text) > 5000:
            text = text[:5000]
        for chunk in chunk_text(text, max_words=140, overlap_words=20):
            chunk = clean_text(chunk)
            if not chunk:
                continue
            docs.append(chunk)
            metadatas.append({"source_url": p.url, "title": p.title})
            ids.append(str(uuid.uuid4()))
            if len(docs) >= max_chunks:
                break
        if len(docs) >= max_chunks:
            break

    client = chromadb.PersistentClient(path=persist_dir)
    col = client.get_or_create_collection(
        name=collection_name,
        embedding_function=OllamaEmbeddingFunction(ollama),
    )

    # Simple reset for repeatable ingests
    try:
        existing = col.count()
        if existing:
            col.delete(where={})
    except Exception:
        pass

    # Add in batches for stability
    batch = 128
    for i in range(0, len(docs), batch):
        col.add(
            documents=docs[i : i + batch],
            metadatas=metadatas[i : i + batch],
            ids=ids[i : i + batch],
        )

    print(f"[ingest] vector DB built persist_dir={persist_dir} collection={collection_name} chunks={len(docs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

