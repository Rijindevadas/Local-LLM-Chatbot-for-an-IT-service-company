from __future__ import annotations

import re
from typing import Optional


_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE_RE = re.compile(
    r"\b(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{4}\b"
)


HUMAN_HANDOFF_KEYWORDS = (
    "talk to human",
    "contact team",
    "call me",
)


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_text(text: str, *, max_words: int = 220, overlap_words: int = 40) -> list[str]:
    words = text.split()
    if not words:
        return []
    if max_words <= 0:
        return [text]
    overlap_words = max(0, min(overlap_words, max_words - 1))

    chunks: list[str] = []
    start = 0
    n = len(words)
    while start < n:
        end = min(n, start + max_words)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap_words)
    return chunks


def extract_email(text: str) -> Optional[str]:
    m = _EMAIL_RE.search(text or "")
    return m.group(0) if m else None


def extract_phone(text: str) -> Optional[str]:
    m = _PHONE_RE.search(text or "")
    return m.group(0) if m else None


def extract_name(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"\b(?:my name is|i am|i'm)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"\bname\s*[:\-]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", text, re.IGNORECASE)
    if m2:
        return m2.group(1).strip()
    return None


def detect_human_handoff(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in HUMAN_HANDOFF_KEYWORDS)


def detect_contact_request_intent(text: str) -> bool:
    """
    When true, the assistant should ask for contact details.
    This should be triggered by explicit intent like: pricing/quote,
    project requirements, or asking for more details.
    """
    t = (text or "").lower()

    # Explicit lead / purchase intent
    lead_keywords = (
        "price",
        "pricing",
        "cost",
        "quote",
        "estimate",
        "budget",
        "how much",
        "project",
        "requirements",
        "need a",
        "need an",
        "want a",
        "want to",
        "interested",
        "talk to human",
        "contact team",
        "call me",
        "call",
        "contact",
        "email",
        "phone",
        "schedule a call",
        "book",
        "meeting",
    )

    # Users often say "more details" / "tell me more"
    more_details_phrases = (
        "more details",
        "tell me more",
        "additional details",
        "details about",
        "details for",
    )

    if any(k in t for k in more_details_phrases):
        return True

    return any(k in t for k in lead_keywords)


def score_lead(*, message: str, email: Optional[str], phone: Optional[str]) -> int:
    score = 0
    t = (message or "").lower()

    service_words = ("service", "services", "support", "website", "app", "cloud", "devops", "security", "it")
    if any(w in t for w in service_words):
        score += 10

    pricing_words = ("price", "pricing", "cost", "quote", "estimate", "rates", "budget")
    if any(w in t for w in pricing_words):
        score += 20

    if email:
        score += 30
    if phone:
        score += 40

    return score
