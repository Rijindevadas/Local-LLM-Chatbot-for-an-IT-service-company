from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass
from typing import Any, Optional

import chromadb
import requests

from database import Lead, save_lead
from utils import (
    clean_text,
    detect_human_handoff,
    detect_contact_request_intent,
    extract_email,
    extract_name,
    extract_phone,
    score_lead,
)


PROMPT_TEMPLATE = """You are an AI assistant for an IT company.

You must represent Pokak Technologies (an IT services company).

Goals:
- Help users with Pokak Technologies IT services
- Suggest solutions based on the provided Context
- Convert users into leads when lead_detected is true by encouraging them to share contact details

Answer rules:
- Use the Context whenever it’s relevant.
- If the Context does not contain the answer, say so and ask a short clarifying question.
- Keep the response very short (1-2 sentences).
- If lead_detected is true, end with a short request for contact details (name/email/phone); otherwise do NOT ask for contact details.

Context:
{retrieved_context}

Conversation:
{chat_history}

User:
{user_input}

lead_detected: {lead_detected}
"""


@dataclass(frozen=True)
class ChatResult:
    response: str
    lead_id: Optional[int] = None
    lead_score: Optional[int] = None


class OllamaClient:
    def __init__(self, base_url: str, model: str, *, timeout_seconds: int = 180, max_tokens: int = 300):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        url = f"{self.base_url}/api/generate"
        try:
            r = requests.post(
                url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        # Keep responses short and reduce timeouts
                        "num_predict": int(self.max_tokens),
                    },
                },
                timeout=self.timeout_seconds,
            )
            r.raise_for_status()
            data = r.json()
            out = data.get("response", "")
            return out if isinstance(out, str) else str(out)
        except Exception as e:
            raise RuntimeError(f"Ollama generate error: {e}") from e

    def embed(self, text: str) -> list[float]:
        url = f"{self.base_url}/api/embeddings"
        try:
            r = requests.post(
                url,
                json={"model": self.model, "prompt": text},
                timeout=self.timeout_seconds,
            )
            r.raise_for_status()
            data = r.json()
            emb = data.get("embedding")
            if not isinstance(emb, list):
                raise RuntimeError("Invalid embedding response")
            return [float(x) for x in emb]
        except Exception as e:
            raise RuntimeError(f"Ollama embeddings error: {e}") from e


class OllamaEmbeddingFunction:
    def __init__(self, client: OllamaClient):
        self._client = client

    def name(self) -> str:  # required by newer chroma validators
        return f"ollama:{self._client.model}"

    # Newer chromadb uses embed_documents / embed_query with `input` keyword
    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return [self._client.embed(t) for t in input]

    def embed_query(self, input) -> list[list[float]]:
        # Chroma may pass a single string or a list[str]; always return list[list[float]]
        if isinstance(input, str):
            return [self._client.embed(input)]
        if isinstance(input, list) and input:
            # Use the first query string
            return [self._client.embed(input[0])]
        return [[]]

    # Maintain backward-compatible __call__ signature for Chroma validators
    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.embed_documents(input)


class ChatMemory:
    def __init__(self, *, max_messages: int = 10):
        self._max = max_messages
        self._lock = threading.Lock()
        self._sessions: dict[str, list[dict[str, str]]] = {}

    def append(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            hist = self._sessions.setdefault(session_id, [])
            hist.append({"role": role, "content": content})
            if len(hist) > self._max:
                self._sessions[session_id] = hist[-self._max :]

    def get(self, session_id: str) -> list[dict[str, str]]:
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def format_for_prompt(self, session_id: str) -> str:
        hist = self.get(session_id)
        return "\n".join([f'{m["role"]}: {m["content"]}' for m in hist]).strip()


class RAGStore:
    def __init__(self, *, persist_dir: str, collection: str, embedding_fn: Any):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection,
            embedding_function=embedding_fn,
        )

    def query(self, query_text: str, *, top_k: int = 3) -> list[str]:
        try:
            res = self._collection.query(query_texts=[query_text], n_results=top_k)
            docs = res.get("documents") or []
            if not docs or not isinstance(docs, list) or not docs[0]:
                return []
            return [clean_text(d) for d in docs[0] if isinstance(d, str)]
        except Exception as e:
            raise RuntimeError(f"Chroma query error: {e}") from e


class AIEngine:
    def __init__(self):
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "llama3")
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./vectorstore")
        self._top_k = int(os.getenv("RAG_TOP_K", "3"))
        self._collection = os.getenv("CHROMA_COLLECTION", "docs")
        self._memory = ChatMemory(max_messages=int(os.getenv("CHAT_MEMORY_MAX", "10")))

        timeout_seconds = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180"))
        max_tokens = int(os.getenv("OLLAMA_MAX_TOKENS", "300"))
        self._ollama = OllamaClient(base_url=base_url, model=model, timeout_seconds=timeout_seconds, max_tokens=max_tokens)
        self._rag = RAGStore(
            persist_dir=persist_dir,
            collection=self._collection,
            embedding_fn=OllamaEmbeddingFunction(self._ollama),
        )

    def chat(self, *, session_id: str, user_input: str) -> ChatResult:
        user_input = clean_text(user_input or "")
        if not user_input:
            return ChatResult(response="Please enter a message.")

        # Human handoff detection (must be immediate)
        wants_human = detect_human_handoff(user_input)

        # Lead extraction + scoring
        name = extract_name(user_input)
        email = extract_email(user_input)
        phone = extract_phone(user_input)
        score = score_lead(message=user_input, email=email, phone=phone)
        if wants_human:
            score += 50

        # Only ask for contact details when user explicitly signals they want
        # more details about services/projects (or pricing/quote, call, contact, etc.).
        lead_detected = bool(detect_contact_request_intent(user_input))

        # Update memory with user message
        self._memory.append(session_id, "user", user_input)

        if wants_human:
            lead_id = None
            if name or email or phone:
                lead_id = save_lead(Lead(name=name, email=email, phone=phone, message=user_input, score=score))
            self._memory.append(session_id, "assistant", "Our team will contact you shortly")
            return ChatResult(response="Our team will contact you shortly", lead_id=lead_id, lead_score=score)

        # RAG retrieve
        retrieved_chunks = self._rag.query(user_input, top_k=self._top_k)
        retrieved_context = "\n\n".join([f"- {c}" for c in retrieved_chunks]).strip() or "No relevant context found."

        prompt = PROMPT_TEMPLATE.format(
            retrieved_context=retrieved_context,
            chat_history=self._memory.format_for_prompt(session_id) or "(empty)",
            user_input=user_input,
            lead_detected=str(lead_detected),
        )

        # Ollama generate
        try:
            reply = clean_text(self._ollama.generate(prompt))
            # Some models may include an extra debugging note like:
            # "Note: No request for contact details since lead_detected is false"
            # Strip it so the chat UI shows only the real answer.
            reply = re.sub(r"\s*\(Note:.*?\)\s*$", "", reply, flags=re.IGNORECASE | re.DOTALL)
            reply = re.sub(r"\s*Note:.*$", "", reply, flags=re.IGNORECASE | re.DOTALL)
            reply = clean_text(reply)
        except Exception as e:
            reply = f"Sorry — I’m having trouble generating a response right now. ({e})"

        # Save lead if we found contact info
        lead_id = None
        if name or email or phone:
            try:
                lead_id = save_lead(Lead(name=name, email=email, phone=phone, message=user_input, score=score))
            except Exception:
                # DB errors should not break chat
                lead_id = None

        self._memory.append(session_id, "assistant", reply)
        return ChatResult(response=reply, lead_id=lead_id, lead_score=score if (name or email or phone) else None)

