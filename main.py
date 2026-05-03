from __future__ import annotations

from typing import Optional
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

from ai_engine import AIEngine
from database import init_db


load_dotenv()

app = FastAPI(title="AI Chatbot Backend", version="1.0.0")
engine = AIEngine()


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: Optional[str] = Field(default=None, description="Client-provided session id (optional)")


class ChatResponse(BaseModel):
    response: str
    session_id: str


@app.on_event("startup")
def _startup():
    # DB init is best-effort; keep API up if DB is down
    try:
        init_db()
    except Exception:
        pass


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request, response: Response):
    try:
        cookie_session_id = request.cookies.get("session_id")
        session_id = (req.session_id or cookie_session_id or str(uuid4())).strip()

        # If the client didn't send a session_id, set a cookie so next calls keep the memory.
        if not req.session_id:
            response.set_cookie(
                key="session_id",
                value=session_id,
                httponly=True,
                samesite="lax",
            )

        result = engine.chat(session_id=session_id, user_input=req.message)
        return ChatResponse(response=result.response, session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

