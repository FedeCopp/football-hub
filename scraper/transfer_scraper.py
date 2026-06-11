"""
scraper/transfer_scraper.py
Trasferimenti ufficiali via API-Football + notizie/rumors di mercato
da feed RSS (Gazzetta, TMW, Calciomercato.com), processate poi via NLP.
"""
import logging
import time
from datetime import datetime
from typing import Optional

import httpx
from bs4 import BeautifulSoup

from config import settings
from db.database import get_db_session
from db.models import NewsItem, Transfer

logger = logging.getLogger(__name__)

BASE_URL = "https://api-football-v1.p.rapidapi.com/v3"

# Costanti usate da nlp/transfer_analyzer.py
SOURCE_WEIGHTS = {
    "api_football_confirmed": 3.0,
    "api_football_loan": 2.0,
    "api_football_rumor": 1.5,
    "gazzetta": 2.0,
    "tmw": 2.0,
    "calciomercato": 1.5,
}

# Feed RSS di notizie calciomercato — usati per popolare NewsItem,
# poi processati da nlp.transfer_analyzer per estrarre rumors con
# probabilità e spiegazione.
RSS_FEEDS = {
    "gazzetta": "https://www.gazzetta.it/rss/calciomercato.xml",
    "tmw": "https://www.tuttomercatoweb.com/rss",
    "calciomercato": "https://www.calciomercato.com/rss",
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

    # ── Notizie / rumors di mercato (RSS) ──────────────────────

    def scrape_news_sources(self) -> int:
        """
        Scarica le ultime notizie di calciomercato dai feed RSS configurati
        e le salva come NewsItem (in attesa di processing NLP).
        Restituisce il numero di notizie nuove salvate.
        """
        total = 0
        for source, url in RSS_FEEDS.items():
            try:
                total += self._scrape_rss_feed(source, url)
            except Exception as e:
                logger.warning(f"Errore scraping feed {source} ({url}): {e}")
        logger.info(f"Notizie mercato salvate: {total}")
        return total

    def _scrape_rss_feed(self, source: str, url: str) -> int:
        headers = {"User-Agent": settings.SCRAPER_USER_AGENT}
        r = httpx.get(url, headers=headers, timeout=15, follow_redirects=True)
        r.raise_for_status()

        soup = BeautifulSoup(r.content, "xml")
        items = soup.find_all("item")
        count = 0

        for item in items[:30]:
            title = (item.find("title").get_text(strip=True) if item.find("title") else "")
            link = (item.find("link").get_text(strip=True) if item.find("link") else "")
            description = (item.find("description").get_text(strip=True) if item.find("description") else "")
            pub_date_raw = (item.find("pubDate").get_text(strip=True) if item.find("pubDate") else "")

            if not title or not link:
                continue

            published = self._parse_rss_date(pub_date_raw)

            with get_db_session() as db:
                if db.query(NewsItem).filter_by(url=link).first():
                    continue

                news = NewsItem(
                    source=source,
                    title=title,
                    body=description,
                    url=link,
                    published=published,
                )
                db.add(news)
                count += 1

        return count

    @staticmethod
    def _parse_rss_date(date_str: str) -> datetime:
        if not date_str:
            return datetime.utcnow()
        from email.utils import parsedate_to_datetime
        try:
            dt = parsedate_to_datetime(date_str)
            return dt.replace(tzinfo=None) if dt.tzinfo else dt
        except (TypeError, ValueError):
            return datetime.utcnow()

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
