from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor


LEADS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS leads(
  id SERIAL PRIMARY KEY,
  name TEXT,
  email TEXT,
  phone TEXT,
  message TEXT,
  score INT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


@dataclass(frozen=True)
class Lead:
    name: Optional[str]
    email: Optional[str]
    phone: Optional[str]
    message: str
    score: int


def _database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    return url


@contextmanager
def get_conn():
    conn = psycopg2.connect(_database_url())
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(LEADS_TABLE_DDL)
            conn.commit()
        print("[db] leads table ready")
    except Exception as e:
        print(f"[db] init error: {e}")
        raise


def save_lead(lead: Lead) -> int:
    try:
        with get_conn() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    INSERT INTO leads(name, email, phone, message, score)
                    VALUES(%s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (lead.name, lead.email, lead.phone, lead.message, lead.score),
                )
                row = cur.fetchone()
            conn.commit()
        lead_id = int(row["id"]) if row and row.get("id") is not None else -1
        print(f"[db] lead saved id={lead_id} score={lead.score}")
        return lead_id
    except Exception as e:
        print(f"[db] save error: {e}")
        raise
