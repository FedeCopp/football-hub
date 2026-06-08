"""
api/chat_router.py
Endpoint FastAPI per il chatbot — REST + Server-Sent Events (streaming).

Registra questo router in main.py con:
    from api.chat_router import chat_router
    app.include_router(chat_router)
"""
import json
import logging
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from chatbot.agent import chat_manager

logger = logging.getLogger(__name__)

chat_router = APIRouter(prefix="/api/chat", tags=["chat"])


# ─── Modelli Pydantic ─────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: str = ""          # se vuoto ne genera uno nuovo


class ChatResponse(BaseModel):
    response: str
    session_id: str


class SessionInfo(BaseModel):
    session_id: str
    active_sessions: int


# ─── Endpoint ────────────────────────────────────────────────

@chat_router.post("/", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Endpoint chat principale — risposta completa (non streaming).
    Il frontend invia il messaggio e riceve la risposta completa.

    Body: { "message": "...", "session_id": "opzionale" }
    """
    session_id = req.session_id or str(uuid.uuid4())

    if not req.message.strip():
        raise HTTPException(400, "Il messaggio non può essere vuoto")
    if len(req.message) > 2000:
        raise HTTPException(400, "Messaggio troppo lungo (max 2000 caratteri)")

    bot = chat_manager.get_or_create(session_id)

    try:
        response = await bot.chat_async(req.message)
        return ChatResponse(response=response, session_id=session_id)
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(503, f"Chatbot temporaneamente non disponibile: {str(e)[:100]}")


@chat_router.get("/stream")
async def chat_stream(message: str, session_id: str = ""):
    """
    Endpoint streaming con Server-Sent Events.
    Il frontend riceve i token man mano che vengono generati.

    Uso dal frontend:
        const es = new EventSource(`/api/chat/stream?message=...&session_id=...`)
        es.onmessage = (e) => { /* aggiunge token alla UI */ }
        es.addEventListener('done', () => es.close())
    """
    if not message.strip():
        raise HTTPException(400, "Messaggio vuoto")

    sid = session_id or str(uuid.uuid4())
    bot = chat_manager.get_or_create(sid)

    async def event_generator() -> AsyncGenerator[str, None]:
        # Prima manda il session_id
        yield f"event: session\ndata: {json.dumps({'session_id': sid})}\n\n"

        try:
            async for token in bot.stream_response(message):
                # SSE format: data: <payload>\n\n
                yield f"data: {json.dumps({'token': token})}\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"event: error\ndata: {json.dumps({'error': str(e)[:100]})}\n\n"
        finally:
            yield f"event: done\ndata: {json.dumps({'session_id': sid})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disabilita buffering Nginx
        },
    )


@chat_router.delete("/{session_id}")
async def clear_session(session_id: str):
    """Cancella la memoria di una sessione (reset conversazione)."""
    chat_manager.delete(session_id)
    return {"status": "cleared", "session_id": session_id}


@chat_router.get("/sessions/info", response_model=SessionInfo)
async def sessions_info(session_id: str = ""):
    """Info sulle sessioni attive (per debug)."""
    return SessionInfo(
        session_id=session_id,
        active_sessions=chat_manager.active_sessions,
    )


@chat_router.post("/session/new", response_model=SessionInfo)
async def new_session():
    """Crea esplicitamente una nuova sessione e restituisce il suo ID."""
    session_id = str(uuid.uuid4())
    chat_manager.get_or_create(session_id)
    return SessionInfo(
        session_id=session_id,
        active_sessions=chat_manager.active_sessions,
    )
