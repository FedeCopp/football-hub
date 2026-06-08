"""
tasks.py
Task Celery per tutti i job asincroni e schedulati.

Avvio worker:
    celery -A tasks worker --loglevel=info

Avvio scheduler (beat):
    celery -A tasks beat --loglevel=info
"""
import logging
from datetime import datetime, timedelta

from celery import Celery
from celery.schedules import crontab

from config import settings

logger = logging.getLogger(__name__)

# ─── Celery App ───────────────────────────────────────────────
celery_app = Celery(
    "football_hub",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Europe/Rome",
    enable_utc=True,
    task_track_started=True,
    worker_prefetch_multiplier=1,
)

# ─── Schedule ────────────────────────────────────────────────
celery_app.conf.beat_schedule = {
    # Aggiorna risultati live ogni minuto
    "sync-live-matches": {
        "task": "tasks.sync_live_matches",
        "schedule": 60.0,
    },
    # Controlla formazioni ufficiali ogni 60s (attivo nelle 2h pre-gara)
    "check-official-lineups": {
        "task": "tasks.check_official_lineups",
        "schedule": 60.0,
    },
    # Aggiorna probabili formazioni ogni 4h
    "scrape-probable-lineups": {
        "task": "tasks.scrape_all_probable_lineups",
        "schedule": crontab(minute=0, hour="*/4"),
    },
    # Aggiorna quote ogni 30 minuti
    "sync-odds": {
        "task": "tasks.sync_all_odds",
        "schedule": crontab(minute="*/30"),
    },
    # Scraping notizie mercato ogni 15 minuti
    "scrape-transfer-news": {
        "task": "tasks.scrape_transfer_news",
        "schedule": crontab(minute="*/15"),
    },
    # Ricalcola previsioni ogni mattina alle 8
    "update-predictions": {
        "task": "tasks.update_all_predictions",
        "schedule": crontab(hour=8, minute=0),
    },
    # Sincronizza infortuni ogni mattina alle 7
    "sync-injuries": {
        "task": "tasks.sync_injuries",
        "schedule": crontab(hour=7, minute=0),
    },
}


# ─────────────────────────────────────────────────────────────
# TASK: PARTITE LIVE
# ─────────────────────────────────────────────────────────────

@celery_app.task(name="tasks.sync_live_matches", bind=True, max_retries=3)
def sync_live_matches(self):
    """
    Aggiorna risultati e stato delle partite in corso.
    Ogni minuto — molto leggero, usa API football-data.org.
    """
    try:
        from scraper.football_data_client import football_data
        from api.websocket_manager import ws_manager

        updated = football_data.sync_live_matches()

        if updated:
            # Notifica tutti i client connessi via WebSocket
            import asyncio
            asyncio.run(ws_manager.broadcast({
                "type": "live_update",
                "matches": updated,
                "timestamp": datetime.utcnow().isoformat(),
            }))
            logger.info(f"Live update: {len(updated)} partite")

        return {"updated": len(updated)}

    except Exception as exc:
        logger.error(f"sync_live_matches error: {exc}")
        raise self.retry(exc=exc, countdown=30)


# ─────────────────────────────────────────────────────────────
# TASK: FORMAZIONI UFFICIALI
# ─────────────────────────────────────────────────────────────

@celery_app.task(name="tasks.check_official_lineups", bind=True, max_retries=3)
def check_official_lineups(self):
    """
    Controlla e scarica le formazioni ufficiali per le partite
    che iniziano entro le prossime 2 ore.
    """
    try:
        from scraper.api_football_client import api_football, LEAGUE_IDS
        from db.database import get_db_session
        from db.models import Match, Lineup
        from api.websocket_manager import ws_manager
        import asyncio

        now       = datetime.utcnow()
        threshold = now + timedelta(hours=2)
        new_lineups = []

        with get_db_session() as db:
            # Partite nelle prossime 2 ore senza formazione ufficiale
            upcoming = db.query(Match).filter(
                Match.kickoff >= now,
                Match.kickoff <= threshold,
                Match.status == "scheduled",
            ).all()

            for match in upcoming:
                # Controlla se abbiamo già entrambe le formazioni ufficiali
                official_count = db.query(Lineup).filter_by(
                    match_id=match.id,
                    is_official=True,
                ).count()

                if official_count >= 2:
                    continue   # già scaricate

                if not match.ext_id:
                    continue

                # Prova a scaricare le formazioni ufficiali
                saved = api_football.save_lineup_to_db(
                    fixture_id=int(match.ext_id),
                    match_id=match.id,
                )

                if saved:
                    new_lineups.append(match.id)
                    logger.info(f"✓ Formazioni ufficiali per match {match.id}")

        if new_lineups:
            # Push WebSocket ai client
            asyncio.run(ws_manager.broadcast({
                "type": "official_lineups",
                "match_ids": new_lineups,
                "timestamp": datetime.utcnow().isoformat(),
            }))

        return {"new_lineups": len(new_lineups)}

    except Exception as exc:
        logger.error(f"check_official_lineups error: {exc}")
        raise self.retry(exc=exc, countdown=60)


# ─────────────────────────────────────────────────────────────
# TASK: PROBABILI FORMAZIONI
# ─────────────────────────────────────────────────────────────

@celery_app.task(name="tasks.scrape_all_probable_lineups")
def scrape_all_probable_lineups():
    """
    Scraping probabili formazioni da Gazzetta e Fantacalcio
    per le partite dei prossimi 7 giorni.
    """
    import asyncio
    from scraper.lineup_scraper import lineup_scraper
    from db.database import get_db_session
    from db.models import Match

    now     = datetime.utcnow()
    in_week = now + timedelta(days=7)
    count   = 0

    with get_db_session() as db:
        upcoming = db.query(Match).filter(
            Match.kickoff >= now,
            Match.kickoff <= in_week,
        ).all()

    for match in upcoming:
        try:
            home_name = match.home_team.short_name or match.home_team.name
            away_name = match.away_team.short_name or match.away_team.name

            lineups = asyncio.run(
                lineup_scraper.scrape_fantacalcio(home_name, away_name)
            )

            for lineup_data in lineups:
                lineup_scraper.save_probable_lineup(match.id, lineup_data)
                count += 1

        except Exception as e:
            logger.warning(f"Errore scraping formazione match {match.id}: {e}")

    logger.info(f"Aggiornate {count} probabili formazioni")
    return {"count": count}


# ─────────────────────────────────────────────────────────────
# TASK: QUOTE
# ─────────────────────────────────────────────────────────────

@celery_app.task(name="tasks.sync_all_odds")
def sync_all_odds():
    """
    Aggiorna le quote per tutte le competizioni monitorate.
    Usa The Odds API — attento al limite mensile!
    """
    from scraper.odds_client import odds_client

    competitions = ["serie_a", "champions_league", "premier_league"]
    total = 0

    for comp in competitions:
        try:
            count = odds_client.sync_odds_for_competition(comp)
            total += count
            logger.info(f"  {comp}: {count} partite aggiornate")

            # Controlla richieste rimanenti
            remaining = odds_client.requests_remaining
            if remaining is not None and remaining < 50:
                logger.warning(
                    f"⚠️ Odds API: solo {remaining} richieste rimanenti questo mese!"
                )
                break

        except Exception as e:
            logger.error(f"Errore sync odds {comp}: {e}")

    return {"total": total}


# ─────────────────────────────────────────────────────────────
# TASK: NOTIZIE MERCATO
# ─────────────────────────────────────────────────────────────

@celery_app.task(name="tasks.scrape_transfer_news")
def scrape_transfer_news():
    """
    Scarica le ultime notizie di mercato da Sky Sport, TMW, Calciomercato.com.
    Poi passa i titoli non ancora processati al modulo NLP.
    """
    from scraper.transfer_scraper import transfer_scraper

    count = transfer_scraper.scrape_all_sources()
    logger.info(f"Scraped {count} nuove notizie mercato")

    # Avvia il processing NLP in background
    process_unprocessed_news.delay()
    return {"new_articles": count}


@celery_app.task(name="tasks.process_unprocessed_news")
def process_unprocessed_news():
    """
    Passa le notizie non processate al modulo NLP
    per estrarre entità e aggiornare i transfer.
    """
    from nlp.transfer_analyzer import transfer_analyzer
    from db.database import get_db_session
    from db.models import NewsItem

    with get_db_session() as db:
        unprocessed = db.query(NewsItem).filter_by(processed=False).limit(50).all()

    count = 0
    for item in unprocessed:
        try:
            transfer_analyzer.process_news_item(item)
            count += 1
        except Exception as e:
            logger.warning(f"Errore NLP su news {item.id}: {e}")

    return {"processed": count}


# ─────────────────────────────────────────────────────────────
# TASK: PREVISIONI ML
# ─────────────────────────────────────────────────────────────

@celery_app.task(name="tasks.update_all_predictions")
def update_all_predictions():
    """
    Ricalcola le previsioni ML per tutte le partite
    programmate nei prossimi 7 giorni.
    """
    from ml.predictor import predictor
    from db.database import get_db_session
    from db.models import Match

    now     = datetime.utcnow()
    in_week = now + timedelta(days=7)

    with get_db_session() as db:
        matches = db.query(Match).filter(
            Match.kickoff >= now,
            Match.kickoff <= in_week,
            Match.status == "scheduled",
        ).all()
        match_ids = [m.id for m in matches]

    count = 0
    for mid in match_ids:
        try:
            predictor.predict_match(mid)
            count += 1
        except Exception as e:
            logger.warning(f"Errore previsione match {mid}: {e}")

    logger.info(f"Aggiornate {count} previsioni")
    return {"count": count}


# ─────────────────────────────────────────────────────────────
# TASK: INFORTUNI
# ─────────────────────────────────────────────────────────────

@celery_app.task(name="tasks.sync_injuries")
def sync_injuries():
    """Sincronizza infortuni da API-Football per le leghe monitorate."""
    from scraper.api_football_client import api_football

    season = datetime.utcnow().year
    totals = {}

    for name, lid in [("serie_a", 135), ("premier_league", 39)]:
        try:
            count = api_football.sync_injuries(lid, season)
            totals[name] = count
        except Exception as e:
            logger.error(f"Errore sync injuries {name}: {e}")

    return totals


# ─────────────────────────────────────────────────────────────
# TASK: IMPORT INIZIALE (run once)
# ─────────────────────────────────────────────────────────────

@celery_app.task(name="tasks.initial_data_import")
def initial_data_import():
    """
    Import iniziale di 3 stagioni di dati storici.
    Esegui UNA sola volta dopo aver configurato il DB:
        celery -A tasks call tasks.initial_data_import
    """
    from scraper.football_data_client import football_data

    competitions = ["SA", "CL", "PL"]
    results = {}

    for comp in competitions:
        try:
            logger.info(f"=== Import storico {comp} ===")
            count = football_data.import_historical_matches(comp, seasons_back=3)
            results[comp] = count
        except Exception as e:
            logger.error(f"Errore import {comp}: {e}")
            results[comp] = 0

    logger.info(f"Import completato: {results}")
    return results
