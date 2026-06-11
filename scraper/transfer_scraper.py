"""
scraper/transfer_scraper.py
Trasferimenti ufficiali via API-Football con rate limiting corretto.
"""
import logging
import time
from datetime import datetime
from typing import Optional

import httpx

from config import settings
from db.database import get_db_session
from db.models import Transfer

logger = logging.getLogger(__name__)

BASE_URL = "https://api-football-v1.p.rapidapi.com/v3"

# Costanti usate da nlp/transfer_analyzer.py
SOURCE_WEIGHTS = {
    "api_football_confirmed": 3.0,
    "api_football_loan": 2.0,
    "api_football_rumor": 1.5,
}
CONFIRMATION_PATTERNS = [
    r"official", r"confirmed", r"completed", r"signed",
    r"ufficiale", r"confermato", r"here we go",
]
RUMOR_PATTERNS = [
    r"interest", r"linked", r"rumour", r"could",
    r"interesse", r"piace", r"potrebbe",
]

# Solo le squadre principali — max 5 per rispettare 10 req/min
MAIN_TEAMS = {
    489: "AC Milan",
    505: "Inter",
    496: "Juventus",
    492: "Napoli",
    497: "AS Roma",
}


class TransferScraper:
    def __init__(self):
        self.headers = {
            "X-RapidAPI-Key": settings.API_FOOTBALL_KEY,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com",
        }

    def _get(self, endpoint: str, params: dict) -> list:
        """GET con retry e rispetto del rate limit."""
        for attempt in range(2):
            try:
                time.sleep(7)  # 7 secondi tra chiamate = max 8 req/min (sotto il limite di 10)
                url = f"{BASE_URL}/{endpoint}"
                r = httpx.get(url, headers=self.headers, params=params, timeout=15)
                if r.status_code == 429:
                    logger.warning(f"Rate limit raggiunto, attendo 30s...")
                    time.sleep(30)
                    continue
                if r.status_code == 403:
                    logger.warning(f"403 Forbidden — piano free non permette questo endpoint")
                    return []
                r.raise_for_status()
                return r.json().get("response", [])
            except Exception as e:
                logger.warning(f"API-Football error {endpoint} (tentativo {attempt+1}): {e}")
                if attempt == 0:
                    time.sleep(15)
        return []

    def get_team_transfers(self, team_id: int, season: int) -> list:
        return self._get("transfers", {"team": team_id, "season": season})

    def scrape_all_sources(self) -> int:
        """Scarica trasferimenti per le squadre principali."""
        if not settings.API_FOOTBALL_KEY:
            logger.warning("API_FOOTBALL_KEY non configurata")
            return 0

        current_year = datetime.utcnow().year
        count = 0

        # Solo stagione corrente per minimizzare le chiamate
        season = current_year if datetime.utcnow().month >= 6 else current_year - 1

        for team_id, team_name in MAIN_TEAMS.items():
            logger.info(f"Fetching trasferimenti {team_name} stagione {season}...")
            transfers = self.get_team_transfers(team_id, season)

            for item in transfers:
                try:
                    if self._save_transfer(item, team_name):
                        count += 1
                except Exception as e:
                    logger.debug(f"Skip transfer: {e}")

        logger.info(f"Trasferimenti importati: {count}")
        return count

    def _save_transfer(self, item: dict, team_name: str) -> bool:
        player_data = item.get("player", {})
        transfers_list = item.get("transfers", [])

        if not player_data or not transfers_list:
            return False

        player_name = player_data.get("name", "")
        if not player_name:
            return False

        saved = False
        for t in transfers_list[-3:]:  # solo ultimi 3 trasferimenti per giocatore
            teams = t.get("teams", {})
            from_team = teams.get("out", {}).get("name", "")
            to_team = teams.get("in", {}).get("name", "")
            transfer_type = t.get("type", "")
            date_str = t.get("date", "")

            if not to_team:
                continue

            if transfer_type in ("Free", "Permanent"):
                probability = 95.0
                status = "confirmed"
            elif transfer_type == "Loan":
                probability = 90.0
                status = "confirmed"
            else:
                probability = 75.0
                status = "advanced"

            try:
                date = datetime.fromisoformat(date_str).strftime("%d/%m/%Y") if date_str else "?"
            except Exception:
                date = date_str[:10] if date_str else "?"

            fee = transfer_type or "N/D"
            detail = f"Trasferimento: {player_name} da {from_team} a {to_team}. Tipo: {fee}. Data: {date}."

            with get_db_session() as db:
                from sqlalchemy import func
                existing = db.query(Transfer).filter(
                    func.lower(Transfer.player_name) == player_name.lower(),
                    func.lower(Transfer.to_team) == to_team.lower(),
                ).first()

                if existing:
                    continue

                transfer = Transfer(
                    player_name=player_name,
                    from_team=from_team,
                    to_team=to_team,
                    fee_estimate=fee,
                    probability=probability,
                    status=status,
                    here_we_go=False,
                    detail=detail,
                    sources=[{
                        "source": "api_football_confirmed",
                        "title": detail,
                        "intensity": 0.9,
                        "published": datetime.utcnow().isoformat(),
                    }],
                )
                db.add(transfer)
                saved = True

        return saved


transfer_scraper = TransferScraper()
