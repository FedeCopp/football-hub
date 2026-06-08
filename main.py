"""
main.py
FastAPI app principale — tutti gli endpoint REST + WebSocket.
"""
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from config import settings
from db.database import get_db, init_db, health_check
from db.models import Match, Team, Player, Lineup, LineupPlayer, Transfer, Prediction, Odds, Injury
from api.chat_router import chat_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── WebSocket Manager ────────────────────────────────────────
class WebSocketManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)
        logger.info(f"WS connesso. Totale: {len(self.active)}")

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: dict):
        import json
        data = json.dumps(message)
        disconnected = []
        for ws in self.active:
            try:
                await ws.send_text(data)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            self.disconnect(ws)


ws_manager = WebSocketManager()


# ─── Lifecycle ────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("⚽ FootballHub API avviato")
    init_db()
    yield
    logger.info("FootballHub API fermato")


# ─── App ──────────────────────────────────────────────────────
app = FastAPI(
    title="FootballHub API",
    version="1.0.0",
    description="Backend per probabili formazioni, mercato, previsioni ML",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)

app.include_router(chat_router)


# ─────────────────────────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok" if health_check() else "db_error",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
    }


# ─────────────────────────────────────────────────────────────
# WEBSOCKET
# ─────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    Client si connette qui per ricevere aggiornamenti real-time:
    - formazioni ufficiali disponibili
    - risultati live
    - nuovi rumors mercato
    """
    await ws_manager.connect(ws)
    try:
        while True:
            data = await ws.receive_text()
            # Ping/pong keepalive
            if data == "ping":
                await ws.send_text("pong")
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)


# ─────────────────────────────────────────────────────────────
# PARTITE
# ─────────────────────────────────────────────────────────────

@app.get("/api/matches")
def get_matches(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    competition: Optional[str] = None,
    status: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """
    Lista partite con filtri.
    date_from/date_to: "YYYY-MM-DD"
    status: scheduled | live | finished
    """
    query = db.query(Match)

    if date_from:
        query = query.filter(Match.kickoff >= datetime.fromisoformat(date_from))
    if date_to:
        query = query.filter(
            Match.kickoff <= datetime.fromisoformat(date_to) + timedelta(days=1)
        )
    if status:
        query = query.filter(Match.status == status)

    matches = query.order_by(Match.kickoff).limit(50).all()
    return [_serialize_match(m, db) for m in matches]


@app.get("/api/matches/today")
def get_today_matches(db: Session = Depends(get_db)):
    """Partite di oggi."""
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    tomorrow = today + timedelta(days=1)
    matches = db.query(Match).filter(
        Match.kickoff >= today,
        Match.kickoff < tomorrow,
    ).order_by(Match.kickoff).all()
    return [_serialize_match(m, db) for m in matches]


@app.get("/api/matches/{match_id}")
def get_match(match_id: int, db: Session = Depends(get_db)):
    match = db.query(Match).filter_by(id=match_id).first()
    if not match:
        raise HTTPException(404, "Partita non trovata")
    return _serialize_match(match, db, detailed=True)


def _serialize_match(match: Match, db: Session, detailed: bool = False) -> dict:
    home = match.home_team
    away = match.away_team

    # Quote medie
    odds_row = db.query(Odds).filter_by(
        match_id=match.id, market="1x2", bookmaker="average"
    ).first()

    # Forma ultima 5 partite
    def get_form(team_id: int) -> list[str]:
        recent = db.query(Match).filter(
            ((Match.home_team_id == team_id) | (Match.away_team_id == team_id)),
            Match.status == "finished",
            Match.kickoff < match.kickoff,
        ).order_by(Match.kickoff.desc()).limit(5).all()

        form = []
        for m in recent:
            if m.home_team_id == team_id:
                if m.home_score > m.away_score:
                    form.append("W")
                elif m.home_score == m.away_score:
                    form.append("D")
                else:
                    form.append("L")
            else:
                if m.away_score > m.home_score:
                    form.append("W")
                elif m.home_score == m.away_score:
                    form.append("D")
                else:
                    form.append("L")
        return form

    result = {
        "id": match.id,
        "competition": match.competition.name if match.competition else "",
        "home": {"id": home.id, "name": home.name, "short": home.short_name},
        "away": {"id": away.id, "name": away.name, "short": away.short_name},
        "kickoff": match.kickoff.isoformat() if match.kickoff else None,
        "status": match.status,
        "score": {
            "home": match.home_score,
            "away": match.away_score,
            "ht_home": match.home_ht,
            "ht_away": match.away_ht,
        },
        "form": {
            "home": get_form(home.id),
            "away": get_form(away.id),
        },
        "odds": {
            "home": odds_row.home_win if odds_row else None,
            "draw": odds_row.draw if odds_row else None,
            "away": odds_row.away_win if odds_row else None,
            "impl_home": odds_row.impl_home if odds_row else None,
            "impl_draw": odds_row.impl_draw if odds_row else None,
            "impl_away": odds_row.impl_away if odds_row else None,
        },
    }

    if detailed:
        result["stats"] = {
            "home_xg": match.home_xg,
            "away_xg": match.away_xg,
            "home_possession": match.home_possession,
            "away_possession": match.away_possession,
            "home_shots": match.home_shots,
            "away_shots": match.away_shots,
        }

    return result


# ─────────────────────────────────────────────────────────────
# FORMAZIONI
# ─────────────────────────────────────────────────────────────

@app.get("/api/matches/{match_id}/lineups")
def get_lineups(match_id: int, db: Session = Depends(get_db)):
    """
    Formazioni per una partita.
    Restituisce quelle ufficiali se disponibili, altrimenti le probabili.
    """
    match = db.query(Match).filter_by(id=match_id).first()
    if not match:
        raise HTTPException(404, "Partita non trovata")

    # Prima cerca ufficiali
    lineups = db.query(Lineup).filter_by(match_id=match_id, is_official=True).all()
    if not lineups:
        # Fallback su probabili (prendi la più recente per team)
        lineups = (
            db.query(Lineup)
            .filter_by(match_id=match_id, is_official=False)
            .order_by(Lineup.fetched_at.desc())
            .all()
        )

    return [_serialize_lineup(l, db) for l in lineups]


def _serialize_lineup(lineup: Lineup, db: Session) -> dict:
    players = db.query(LineupPlayer).filter_by(lineup_id=lineup.id).all()

    starters = [
        {
            "id": lp.player_id,
            "name": lp.player.name if lp.player else "Unknown",
            "number": lp.shirt_num,
            "position": lp.position,
            "uncertain": lp.is_uncertain,
        }
        for lp in players if lp.role == "starter"
    ]
    subs = [
        {
            "id": lp.player_id,
            "name": lp.player.name if lp.player else "Unknown",
            "number": lp.shirt_num,
            "position": lp.position,
        }
        for lp in players if lp.role == "substitute"
    ]

    return {
        "team_id": lineup.team_id,
        "team_name": lineup.team.name if lineup.team else "",
        "formation": lineup.formation,
        "is_official": lineup.is_official,
        "source": lineup.source,
        "fetched_at": lineup.fetched_at.isoformat() if lineup.fetched_at else None,
        "starters": starters,
        "substitutes": subs,
    }


# ─────────────────────────────────────────────────────────────
# TRASFERIMENTI
# ─────────────────────────────────────────────────────────────

@app.get("/api/transfers")
def get_transfers(
    min_prob: float = 0,
    status: Optional[str] = None,
    team: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """
    Lista rumors di mercato.
    min_prob: filtra per probabilità minima (0-100)
    status: rumor | advanced | confirmed
    """
    from sqlalchemy import func

    query = db.query(Transfer).filter(Transfer.probability >= min_prob)

    if status:
        query = query.filter(Transfer.status == status)
    if team:
        query = query.filter(
            (func.lower(Transfer.from_team).contains(team.lower()))
            | (func.lower(Transfer.to_team).contains(team.lower()))
        )

    transfers = query.order_by(Transfer.probability.desc()).limit(50).all()

    return [
        {
            "id": t.id,
            "player": t.player_name,
            "from_team": t.from_team,
            "to_team": t.to_team,
            "fee": t.fee_estimate,
            "probability": t.probability,
            "status": t.status,
            "here_we_go": t.here_we_go,
            "detail": t.detail,
            "sources": t.sources,
            "updated_at": t.updated_at.isoformat() if t.updated_at else None,
        }
        for t in transfers
    ]


# ─────────────────────────────────────────────────────────────
# PREVISIONI
# ─────────────────────────────────────────────────────────────

@app.get("/api/matches/{match_id}/prediction")
def get_prediction(match_id: int, db: Session = Depends(get_db)):
    """Previsione ML per una partita."""
    pred = db.query(Prediction).filter_by(match_id=match_id).first()
    if not pred:
        # Genera al volo se non esiste
        try:
            from ml.predictor import predictor
            pred = predictor.predict_match(match_id)
        except Exception as e:
            raise HTTPException(503, f"Previsione non disponibile: {e}")

    if not pred:
        raise HTTPException(404, "Previsione non trovata")

    return {
        "match_id": match_id,
        "outcome": {
            "home": pred.prob_home,
            "draw": pred.prob_draw,
            "away": pred.prob_away,
        },
        "scores": pred.score_probs,
        "scorers": pred.scorer_probs,
        "booked": pred.booked_probs,
        "btts": pred.btts_prob,
        "over25": pred.over25_prob,
        "model": pred.model_version,
        "confidence": pred.confidence,
        "created_at": pred.created_at.isoformat() if pred.created_at else None,
    }


# ─────────────────────────────────────────────────────────────
# INFORTUNI
# ─────────────────────────────────────────────────────────────

@app.get("/api/injuries")
def get_injuries(
    team_id: Optional[int] = None,
    active_only: bool = True,
    db: Session = Depends(get_db),
):
    query = db.query(Injury)
    if active_only:
        query = query.filter(Injury.is_active == True)
    if team_id:
        query = query.join(Player).filter(Player.team_id == team_id)

    injuries = query.order_by(Injury.start_date.desc()).limit(50).all()

    return [
        {
            "player_id": inj.player_id,
            "player": inj.player.name if inj.player else "",
            "team": inj.player.team.name if inj.player and inj.player.team else "",
            "type": inj.type,
            "reason": inj.reason,
            "start_date": inj.start_date.isoformat() if inj.start_date else None,
            "end_date": inj.end_date.isoformat() if inj.end_date else None,
            "is_active": inj.is_active,
        }
        for inj in injuries
    ]


# ─────────────────────────────────────────────────────────────
# SQUADRE & GIOCATORI
# ─────────────────────────────────────────────────────────────

@app.get("/api/teams")
def get_teams(competition: Optional[str] = None, db: Session = Depends(get_db)):
    teams = db.query(Team).limit(100).all()
    return [
        {"id": t.id, "name": t.name, "short": t.short_name, "logo": t.logo_url}
        for t in teams
    ]


@app.get("/api/players/{player_id}/stats")
def get_player_stats(player_id: int, db: Session = Depends(get_db)):
    from db.models import PlayerStats
    player = db.query(Player).filter_by(id=player_id).first()
    if not player:
        raise HTTPException(404, "Giocatore non trovato")

    stats = db.query(PlayerStats).filter_by(player_id=player_id).all()
    return {
        "player": {"id": player.id, "name": player.name, "position": player.position},
        "stats": [
            {
                "season": s.season,
                "appearances": s.appearances,
                "minutes": s.minutes,
                "goals": s.goals,
                "assists": s.assists,
                "yellow_cards": s.yellow_cards,
                "xg": s.xg,
                "xa": s.xa,
            }
            for s in stats
        ],
    }


# ─────────────────────────────────────────────────────────────
# ADMIN / TRIGGER MANUALI
# ─────────────────────────────────────────────────────────────

@app.api_route("/api/admin/import", methods=["GET","POST"])
def trigger_import(secret: str, competition: str = "SA"):
    """Avvia import iniziale (solo con secret corretto)."""
    if secret != settings.SECRET_KEY:
        raise HTTPException(403, "Non autorizzato")
    from tasks import initial_data_import
    task = initial_data_import.delay()
    return {"task_id": task.id, "status": "queued"}


@app.api_route("/api/admin/sync-odds", methods=["GET","POST"])
def trigger_sync_odds(secret: str):
    if secret != settings.SECRET_KEY:
        raise HTTPException(403, "Non autorizzato")
    from tasks import sync_all_odds
    task = sync_all_odds.delay()
    return {"task_id": task.id, "status": "queued"}


@app.api_route("/api/admin/update-predictions", methods=["GET","POST"])
def trigger_predictions(secret: str):
    if secret != settings.SECRET_KEY:
        raise HTTPException(403, "Non autorizzato")
    from tasks import update_all_predictions
    task = update_all_predictions.delay()
    return {"task_id": task.id, "status": "queued"}


# ─── Avvio diretto ────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.DEBUG,
    )
