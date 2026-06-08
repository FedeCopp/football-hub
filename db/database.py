"""
db/database.py
Connessione al database, sessioni, e funzioni di init/seed.
"""
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
import logging

from db.models import Base
from config import settings

logger = logging.getLogger(__name__)

# ─── Engine ──────────────────────────────────────────────────
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,         # controlla connessione prima di usarla
    pool_size=10,
    max_overflow=20,
    echo=settings.DEBUG,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ─── Context manager per le sessioni ─────────────────────────
@contextmanager
def get_db_session() -> Session:
    """Usa questo nei task Celery e script."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ─── Dependency FastAPI ───────────────────────────────────────
def get_db():
    """Usa questo come Depends() nelle route FastAPI."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ─── Setup ───────────────────────────────────────────────────
def init_db():
    """Crea tutte le tabelle se non esistono."""
    logger.info("Inizializzazione database...")
    Base.metadata.create_all(bind=engine)
    logger.info("Tabelle create con successo.")


def health_check() -> bool:
    """Testa la connessione al DB."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"DB health check fallito: {e}")
        return False
