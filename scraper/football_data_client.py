"""
scraper/football_data_client.py
Client per football-data.org — dati storici e live.
Piano gratuito: 10 req/min, competizioni principali.

Docs: https://docs.football-data.org/general/v4/index.html
"""
import time
import logging
from datetime import datetime, timedelta
from typing import Optional

import httpx

from config import settings
from db.database import get_db_session
from db.models import Competition, Team, Player, Match, MatchEvent

logger = logging.getLogger(__name__)

BASE_URL = "https://api.football-data.org/v4"

# IDs competizioni su football-data.org
COMPETITION_IDS = {
    "serie_a":         "SA",
    "champions_league":"CL",
    "premier_league":  "PL",
    "la_liga":         "PD",
    "bundesliga":      "BL1",
    "ligue_1":         "FL1",
    "europa_league":   "EL",
}


class FootballDataClient:
    def __init__(self):
        self.headers = {
            "X-Auth-Token": settings.FOOTBALL_DATA_API_KEY,
            "Accept": "application/json",
        }
        self._last_call = 0.0
        self._min_interval = 6.5   # 10 req/min → aspetta 6.5s tra chiamate

    def _get(self, endpoint: str, params: dict = None) -> dict:
        """GET con rate limiting automatico."""
        elapsed = time.time() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

        url = f"{BASE_URL}/{endpoint}"
        try:
            with httpx.Client(timeout=30) as client:
                resp = client.get(url, headers=self.headers, params=params or {})
                resp.raise_for_status()
                self._last_call = time.time()
                return resp.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP {e.response.status_code} su {url}: {e}")
            raise
        except Exception as e:
            logger.error(f"Errore richiesta {url}: {e}")
            raise

    # ── Competizioni ─────────────────────────────────────────

    def get_competition(self, code: str) -> dict:
        """Info su una competizione (squadre, stagione corrente)."""
        return self._get(f"competitions/{code}")

    def get_standings(self, code: str) -> dict:
        """Classifica attuale."""
        return self._get(f"competitions/{code}/standings")

    # ── Partite ───────────────────────────────────────────────

    def get_matches(
        self,
        competition_code: str,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        status: Optional[str] = None,
    ) -> list[dict]:
        """
        Partite di una competizione.
        date_from/date_to: "YYYY-MM-DD"
        status: SCHEDULED | LIVE | IN_PLAY | PAUSED | FINISHED | POSTPONED
        """
        params = {}
        if date_from:
            params["dateFrom"] = date_from
        if date_to:
            params["dateTo"] = date_to
        if status:
            params["status"] = status

        data = self._get(f"competitions/{competition_code}/matches", params)
        return data.get("matches", [])

    def get_match(self, match_id: int) -> dict:
        """Dettaglio singola partita con eventi (gol, cartellini)."""
        return self._get(f"matches/{match_id}")

    def get_today_matches(self) -> list[dict]:
        """Tutte le partite di oggi su tutte le competizioni monitorate."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        return self.get_matches_by_date(today, today)

    def get_matches_by_date(self, date_from: str, date_to: str) -> list[dict]:
        params = {"dateFrom": date_from, "dateTo": date_to}
        data = self._get("matches", params)
        return data.get("matches", [])

    # ── Squadre e Giocatori ───────────────────────────────────

    def get_team(self, team_id: int) -> dict:
        """Rosa completa di una squadra."""
        return self._get(f"teams/{team_id}")

    def get_competition_teams(self, code: str) -> list[dict]:
        """Tutte le squadre di una competizione."""
        data = self._get(f"competitions/{code}/teams")
        return data.get("teams", [])

    # ── Import DB ─────────────────────────────────────────────

    def import_competition(self, code: str, season: str = "2023/24") -> int:
        """
        Importa (o aggiorna) una competizione nel database.
        Restituisce l'ID della competizione.
        """
        data = self.get_competition(code)
        with get_db_session() as db:
            comp = db.query(Competition).filter_by(ext_id=code).first()
            if not comp:
                comp = Competition(
                    ext_id=code,
                    name=data["name"],
                    country=data.get("area", {}).get("name", ""),
                    season=season,
                )
                db.add(comp)
                db.flush()
                logger.info(f"Competizione importata: {comp.name}")
            return comp.id

    def import_teams(self, competition_code: str) -> list[Team]:
        """
        Importa tutte le squadre di una competizione.
        """
        teams_data = self.get_competition_teams(competition_code)
        imported = []

        with get_db_session() as db:
            for t in teams_data:
                ext_id = str(t["id"])
                team = db.query(Team).filter_by(ext_id=ext_id).first()
                if not team:
                    team = Team(
                        ext_id=ext_id,
                        name=t["name"],
                        short_name=t.get("shortName", t["name"][:20]),
                        country=t.get("area", {}).get("name", ""),
                        logo_url=t.get("crest", ""),
                    )
                    db.add(team)
                    logger.info(f"  Team importato: {team.name}")
                else:
                    team.logo_url = t.get("crest", team.logo_url)
                imported.append(team)

        logger.info(f"Importate {len(imported)} squadre per {competition_code}")
        return imported

    def _save_matches(self, matches: list[dict], comp_id: int) -> int:
        """
        Salva (o aggiorna) una lista di partite restituite da football-data.org.
        Restituisce il numero di nuove partite inserite.
        """
        count = 0
        for m in matches:
            ext_id = str(m["id"])
            try:
                with get_db_session() as db:
                    existing = db.query(Match).filter_by(ext_id=ext_id).first()

                    home = db.query(Team).filter_by(
                        ext_id=str(m["homeTeam"]["id"])
                    ).first()
                    away = db.query(Team).filter_by(
                        ext_id=str(m["awayTeam"]["id"])
                    ).first()

                    if not home or not away:
                        continue

                    score = m.get("score", {})
                    full  = score.get("fullTime", {})
                    ht    = score.get("halfTime", {})

                    if existing:
                        # Aggiorna stato/risultato/data (es. rinvii, orari confermati)
                        existing.status     = m.get("status", existing.status).lower()
                        existing.kickoff    = self._parse_date(m.get("utcDate")) or existing.kickoff
                        existing.matchday   = m.get("matchday", existing.matchday)
                        existing.home_score = full.get("home", existing.home_score)
                        existing.away_score = full.get("away", existing.away_score)
                        existing.home_ht    = ht.get("home", existing.home_ht)
                        existing.away_ht    = ht.get("away", existing.away_ht)
                        continue

                    match = Match(
                        ext_id=ext_id,
                        competition_id=comp_id,
                        home_team_id=home.id,
                        away_team_id=away.id,
                        matchday=m.get("matchday"),
                        kickoff=self._parse_date(m.get("utcDate")),
                        status=m.get("status", "FINISHED").lower(),
                        home_score=full.get("home"),
                        away_score=full.get("away"),
                        home_ht=ht.get("home"),
                        away_ht=ht.get("away"),
                    )
                    db.add(match)
                    count += 1
            except Exception as e:
                if "duplicate" in str(e).lower() or "unique" in str(e).lower():
                    continue  # partita già inserita da altro thread
                logger.warning(f"Skip match {ext_id}: {e}")

        return count

    def import_historical_matches(
        self,
        competition_code: str,
        seasons_back: int = 3,
    ) -> int:
        """
        Importa le partite storiche (N stagioni indietro).
        Restituisce il numero di partite importate.
        """
        count = 0
        now = datetime.utcnow()

        # Importa comp e team prima
        comp_id = self.import_competition(competition_code)
        self.import_teams(competition_code)

        for i in range(seasons_back):
            year_from = now.year - i - 1
            year_to   = now.year - i
            date_from = f"{year_from}-07-01"
            date_to   = f"{year_to}-06-30"

            logger.info(f"Importo {competition_code} stagione {year_from}/{year_to}...")

            try:
                matches = self.get_matches(competition_code, date_from, date_to)
            except Exception as e:
                logger.warning(f"Errore fetch stagione {year_from}: {e}")
                continue

            count += self._save_matches(matches, comp_id)

        logger.info(f"Importate {count} partite storiche per {competition_code}")
        return count

    def import_upcoming_matches(
        self,
        competition_code: str,
        days_ahead: int = 120,
    ) -> int:
        """
        Importa (o aggiorna) le partite programmate nei prossimi `days_ahead`
        giorni, indipendentemente dai confini di stagione (utile per avere
        sempre il prossimo turno anche a stagione conclusa/non ancora iniziata).
        """
        comp_id = self.import_competition(competition_code)
        self.import_teams(competition_code)

        now = datetime.utcnow()
        date_from = now.strftime("%Y-%m-%d")
        date_to = (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

        try:
            matches = self.get_matches(competition_code, date_from, date_to)
        except Exception as e:
            logger.warning(f"Errore fetch prossime partite {competition_code}: {e}")
            return 0

        count = self._save_matches(matches, comp_id)
        logger.info(f"Importate/aggiornate {count} prossime partite per {competition_code}")
        return count

    def sync_live_matches(self) -> list[dict]:
        """
        Aggiorna le partite in corso o di oggi.
        Da chiamare ogni minuto via Celery.
        """
        today = datetime.utcnow().strftime("%Y-%m-%d")
        matches = self.get_matches_by_date(today, today)
        updated = []

        with get_db_session() as db:
            for m in matches:
                ext_id = str(m["id"])
                match = db.query(Match).filter_by(ext_id=ext_id).first()
                if not match:
                    continue

                score = m.get("score", {})
                full  = score.get("fullTime", {})

                match.status     = m.get("status", match.status).lower()
                match.home_score = full.get("home", match.home_score)
                match.away_score = full.get("away", match.away_score)
                match.updated_at = datetime.utcnow()
                updated.append(ext_id)

        logger.info(f"Sincronizzate {len(updated)} partite live")
        return updated

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
        if not date_str:
            return None
        try:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except ValueError:
            return None


# Istanza singleton
football_data = FootballDataClient()
