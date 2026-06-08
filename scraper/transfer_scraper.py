"""
scraper/transfer_scraper.py
Dati trasferimenti tramite API-Football (RapidAPI).

Tutti i siti di notizie calcistiche bloccano richieste da IP cloud (403).
API-Football invece è un'API ufficiale che funziona da server.

Endpoint usati:
  /v3/transfers?team=ID&season=YYYY  — trasferimenti di una squadra
"""
import logging
from datetime import datetime
from typing import Optional

import httpx

from config import settings
from db.database import get_db_session
from db.models import NewsItem, Transfer, Team

logger = logging.getLogger(__name__)

BASE_URL = "https://api-football-v1.p.rapidapi.com/v3"

# Squadre Serie A su API-Football (ID RapidAPI)
SERIE_A_TEAMS = {
    489: "AC Milan", 505: "Inter", 496: "Juventus", 492: "Napoli",
    497: "AS Roma", 487: "Lazio", 488: "Atalanta", 502: "Fiorentina",
    867: "Bologna", 500: "Udinese", 494: "Cagliari", 486: "Parma",
    499: "Torino", 504: "Verona", 503: "Lecce", 495: "Genoa",
    511: "Como", 798: "Monza", 515: "Sassuolo", 514: "Cremonese",
}

# Squadre Premier League
PREMIER_TEAMS = {
    40: "Liverpool", 42: "Arsenal", 50: "Man City", 33: "Man United",
    49: "Chelsea", 47: "Tottenham", 66: "Aston Villa", 51: "Brighton",
    48: "West Ham", 46: "Leicester",
}

SOURCE_WEIGHTS = {
    "api_football_confirmed": 3.0,
    "api_football_loan": 2.0,
    "api_football_rumor": 1.5,
}


class TransferScraper:
    def __init__(self):
        self.headers = {
            "X-RapidAPI-Key": settings.API_FOOTBALL_KEY,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com",
        }

    def _get(self, endpoint: str, params: dict) -> list:
        try:
            url = f"{BASE_URL}/{endpoint}"
            r = httpx.get(url, headers=self.headers, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            return data.get("response", [])
        except Exception as e:
            logger.warning(f"API-Football error {endpoint}: {e}")
            return []

    def get_team_transfers(self, team_id: int, season: int) -> list[dict]:
        """Trasferimenti ufficiali di una squadra in una stagione."""
        return self._get("transfers", {"team": team_id, "season": season})

    def scrape_all_sources(self) -> int:
        """
        Scarica tutti i trasferimenti recenti da API-Football
        e li salva come Transfer nel database.
        """
        if not settings.API_FOOTBALL_KEY:
            logger.warning("API_FOOTBALL_KEY non configurata — skip transfer scraping")
            return 0

        current_year = datetime.utcnow().year
        seasons = [current_year, current_year - 1]
        count = 0

        all_teams = {**SERIE_A_TEAMS, **PREMIER_TEAMS}

        for team_id, team_name in list(all_teams.items())[:10]:  # max 10 squadre per rispettare rate limit
            for season in seasons:
                transfers = self.get_team_transfers(team_id, season)
                for item in transfers:
                    try:
                        saved = self._save_transfer(item, team_name)
                        if saved:
                            count += 1
                    except Exception as e:
                        logger.debug(f"Skip transfer: {e}")

        logger.info(f"Trasferimenti importati: {count}")
        return count

    def _save_transfer(self, item: dict, team_name: str) -> bool:
        """Salva un trasferimento dal formato API-Football nel DB."""
        player_data = item.get("player", {})
        transfers_list = item.get("transfers", [])

        if not player_data or not transfers_list:
            return False

        player_name = player_data.get("name", "")
        if not player_name:
            return False

        saved = False
        for t in transfers_list:
            teams = t.get("teams", {})
            from_team = teams.get("out", {}).get("name", "")
            to_team = teams.get("in", {}).get("name", "")
            transfer_type = t.get("type", "")
            date_str = t.get("date", "")

            if not to_team:
                continue

            # Determina probabilità in base al tipo
            if transfer_type in ("Free", "Permanent", "€"):
                probability = 95.0
                status = "confirmed"
            elif transfer_type == "Loan":
                probability = 90.0
                status = "confirmed"
            else:
                probability = 75.0
                status = "advanced"

            # Dettaglio
            fee = t.get("type", "N/D")
            try:
                date = datetime.fromisoformat(date_str).strftime("%d/%m/%Y") if date_str else "?"
            except Exception:
                date = date_str[:10] if date_str else "?"

            detail = f"Trasferimento ufficiale: {player_name} dal {from_team} al {to_team}. Tipo: {fee}. Data: {date}."

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
                    sources=[{"source": "api_football_confirmed", "title": detail, "intensity": 0.9}],
                )
                db.add(transfer)
                saved = True

        return saved
