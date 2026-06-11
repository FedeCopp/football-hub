"""
scheduler.py
Scheduler in-process (APScheduler) per i sync periodici.

Railway esegue solo il Dockerfile (uvicorn), senza worker/beat Celery
separati: i job schedulati in tasks.py non vengono mai eseguiti in
produzione. Questo modulo copre i sync essenziali (partite live,
prossimo turno, mercato, previsioni) direttamente nel processo API.
"""
import logging
from datetime import datetime, timedelta

from apscheduler.schedulers.background import BackgroundScheduler

from config import settings

logger = logging.getLogger(__name__)

COMPETITIONS = ["SA", "CL", "PL", "PD"]


def _sync_live_matches():
    try:
        from scraper.football_data_client import football_data
        updated = football_data.sync_live_matches()
        if updated:
            logger.info(f"[scheduler] Live update: {len(updated)} partite")
    except Exception as e:
        logger.error(f"[scheduler] sync_live_matches error: {e}")


def _bootstrap_if_empty():
    """Se il DB non ha ancora partite, importa lo storico (una tantum)."""
    if not settings.FOOTBALL_DATA_API_KEY:
        return
    try:
        from db.database import get_db_session
        from db.models import Match
        from scraper.football_data_client import football_data

        with get_db_session() as db:
            has_matches = db.query(Match.id).first() is not None
        if has_matches:
            return

        logger.info("[scheduler] DB vuoto: avvio import storico iniziale (3 stagioni)")
        for comp in COMPETITIONS:
            football_data.import_historical_matches(comp, seasons_back=3)
    except Exception as e:
        logger.error(f"[scheduler] bootstrap import error: {e}")


def _sync_upcoming_matches():
    if not settings.FOOTBALL_DATA_API_KEY:
        return
    try:
        from scraper.football_data_client import football_data
        for comp in COMPETITIONS:
            football_data.import_upcoming_matches(comp, days_ahead=120)
    except Exception as e:
        logger.error(f"[scheduler] sync_upcoming_matches error: {e}")


def _scrape_transfers():
    try:
        from scraper.transfer_scraper import TransferScraper
        scraper = TransferScraper()

        news_count = scraper.scrape_news_sources()
        logger.info(f"[scheduler] Notizie mercato salvate: {news_count}")

        from nlp.transfer_analyzer import transfer_analyzer
        processed = transfer_analyzer.process_all_unprocessed(limit=200)
        logger.info(f"[scheduler] Notizie processate (NLP): {processed}")

        if settings.API_FOOTBALL_KEY:
            count = scraper.scrape_all_sources()
            logger.info(f"[scheduler] Trasferimenti API-Football importati: {count}")
    except Exception as e:
        logger.error(f"[scheduler] scrape_transfers error: {e}")


def _update_predictions():
    try:
        from ml.predictor import predictor
        from db.database import get_db_session
        from db.models import Match

        now = datetime.utcnow()
        in_week = now + timedelta(days=7)
        with get_db_session() as db:
            match_ids = [
                m.id for m in db.query(Match).filter(
                    Match.kickoff >= now,
                    Match.kickoff <= in_week,
                    Match.status.in_(["scheduled", "timed"]),
                ).all()
            ]
        for mid in match_ids:
            try:
                predictor.predict_match(mid)
            except Exception as e:
                logger.warning(f"[scheduler] previsione match {mid} fallita: {e}")
        logger.info(f"[scheduler] Previsioni aggiornate: {len(match_ids)}")
    except Exception as e:
        logger.error(f"[scheduler] update_predictions error: {e}")


_scheduler: BackgroundScheduler | None = None


def start_scheduler() -> BackgroundScheduler:
    global _scheduler
    if _scheduler is not None:
        return _scheduler

    sched = BackgroundScheduler(timezone="UTC")
    sched.add_job(_bootstrap_if_empty, "date",
                   run_date=datetime.utcnow() + timedelta(seconds=10), id="bootstrap_if_empty")
    sched.add_job(_sync_live_matches, "interval", minutes=2, id="sync_live_matches",
                   next_run_time=datetime.utcnow() + timedelta(seconds=30))
    sched.add_job(_sync_upcoming_matches, "interval", hours=6, id="sync_upcoming_matches",
                   next_run_time=datetime.utcnow() + timedelta(seconds=60))
    sched.add_job(_scrape_transfers, "interval", hours=3, id="scrape_transfers",
                   next_run_time=datetime.utcnow() + timedelta(minutes=2))
    sched.add_job(_update_predictions, "interval", hours=6, id="update_predictions",
                   next_run_time=datetime.utcnow() + timedelta(minutes=3))
    sched.start()
    _scheduler = sched
    logger.info("Scheduler in-process avviato")
    return sched


def stop_scheduler():
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
