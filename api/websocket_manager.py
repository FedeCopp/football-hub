"""
api/websocket_manager.py
Manager WebSocket condiviso — usato da tasks.py per i broadcast.
"""
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    Gestisce le connessioni WebSocket attive e invia broadcast a tutti i client.
    L'istanza viene importata sia da main.py che da tasks.py.
    In produzione con più worker, usa Redis pub/sub per il broadcast cross-process.
    """

    def __init__(self):
        self.active = []

    async def connect(self, ws):
        await ws.accept()
        self.active.append(ws)
        logger.info(f"WS connesso. Totale connessioni: {len(self.active)}")

    def disconnect(self, ws):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: dict):
        """Invia un messaggio JSON a tutti i client connessi."""
        if not self.active:
            return

        data = json.dumps(message, default=str)
        disconnected = []

        for ws in self.active:
            try:
                await ws.send_text(data)
            except Exception as e:
                logger.debug(f"WS send error (disconnessione): {e}")
                disconnected.append(ws)

        for ws in disconnected:
            self.disconnect(ws)

        if self.active:
            logger.info(
                f"Broadcast '{message.get('type', '?')}' → {len(self.active)} client"
            )

    async def send_to_one(self, ws, message: dict):
        """Invia un messaggio a un singolo client."""
        try:
            await ws.send_text(json.dumps(message, default=str))
        except Exception as e:
            logger.warning(f"WS send_to_one error: {e}")
            self.disconnect(ws)


# ─── Redis pub/sub (per deployment multi-worker) ──────────────
class RedisWebSocketManager(WebSocketManager):
    """
    Versione con Redis pub/sub per supportare più worker Uvicorn.
    Da usare in produzione con `--workers 4`.
    """

    def __init__(self, redis_url: str):
        super().__init__()
        self.redis_url  = redis_url
        self._redis     = None
        self._pubsub    = None

    async def _get_redis(self):
        if not self._redis:
            import aioredis
            self._redis = await aioredis.from_url(self.redis_url)
        return self._redis

    async def publish(self, message: dict):
        """Pubblica su Redis invece di broadcast diretto."""
        r = await self._get_redis()
        await r.publish("ws_broadcast", json.dumps(message, default=str))

    async def start_listener(self):
        """
        Avvia il listener Redis in background.
        Chiama questo all'avvio dell'app con asyncio.create_task().
        """
        r = await self._get_redis()
        pubsub = r.pubsub()
        await pubsub.subscribe("ws_broadcast")

        async for msg in pubsub.listen():
            if msg["type"] == "message":
                try:
                    data = json.loads(msg["data"])
                    await super().broadcast(data)
                except Exception as e:
                    logger.error(f"Redis listener error: {e}")


# Singleton — importato da main.py e tasks.py
ws_manager = WebSocketManager()
