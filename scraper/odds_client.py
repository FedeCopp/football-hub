"""
scraper/odds_client.py
Client per The Odds API — quote aggregate da 20+ bookmaker.
Piano free: 500 req/mese (sufficienti per tutta la Serie A).

Docs: https://the-odds-api.com/liveapi/guides/v4/
"""
import logging
from datetime import datetime
from typing import Optional

import httpx

from config import settings
from db.database import get_db_session
from db.models import Match, Odds, Team

logger = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"

# Codici sport su The Odds API
SPORT_KEYS = {
    "serie_a":         "soccer_italy_serie_a",
    "champions_league":"soccer_uefa_champs_league",
    "premier_league":  "soccer_epl",
    "la_liga":         "soccer_spain_la_liga",
    "bundesliga":      "soccer_germany_bundesliga",
    "ligue_1":         "soccer_france_ligue_one",
    "europa_league":   "soccer_uefa_europa_league",
}

# Bookmaker preferiti da usare come fonte principale
PREFERRED_BOOKMAKERS = [
    "bet365", "williamhill", "unibet", "betfair_ex_eu",
    "pinnacle", "draftkings", "fanduel",
]


class OddsClient:
    def __init__(self):
        self.api_key = settings.ODDS_API_KEY
        self._remaining_requests = None

    def _get(self, endpoint: str, params: dict = None) -> dict | list:
        url = f"{BASE_URL}/{endpoint}"
        p = {"apiKey": self.api_key, **(params or {})}

        try:
            with httpx.Client(timeout=30) as client:
                resp = client.get(url, params=p)
                # Salva le richieste rimanenti dall'header
                self._remaining_requests = resp.headers.get(
                    "x-requests-remaining", self._remaining_requests
                )
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Odds API HTTP {e.response.status_code}: {e}")
            raise
        except Exception as e:
            logger.error(f"Odds API errore {url}: {e}")
            raise

    @property
    def requests_remaining(self) -> Optional[int]:
        if self._remaining_requests is not None:
            try:
                return int(self._remaining_requests)
            except ValueError:
                return None
        return None

    # ── Fetch quote ───────────────────────────────────────────

    def get_odds(
        self,
        sport_key: str,
        regions: str = "eu",
        markets: str = "h2h",
        odds_format: str = "decimal",
    ) -> list[dict]:
        """
        Quote per una competizione.
        markets: "h2h" (1X2) | "totals" (over/under) | "btts"
        regions: "eu" | "uk" | "us"
        """
        return self._get(
            f"sports/{sport_key}/odds",
            {
                "regions": regions,
                "markets": markets,
                "oddsFormat": odds_format,
                "bookmakers": ",".join(PREFERRED_BOOKMAKERS),
            },
        )

    def get_all_markets(self, sport_key: str) -> list[dict]:
        """Scarica h2h + totals + btts in un'unica chiamata (conta come 3 req)."""
        markets = "h2h,totals,btts"
        return self._get(
            f"sports/{sport_key}/odds",
            {"regions": "eu", "markets": markets, "oddsFormat": "decimal"},
        )

    # ── Calcolo probabilità implicita ─────────────────────────

    @staticmethod
    def implied_probs(home: float, draw: float, away: float) -> tuple[float, float, float]:
        """
        Converte le quote decimali in probabilità implicite,
        rimuovendo il margine del bookmaker (overround).

        Es. quote 2.40 / 3.20 / 2.90 → somma prob raw = 1.095 (margine 9.5%)
        Normalizziamo per ottenere prob pulite che sommano a 1.
        """
        raw_h = 1 / home if home else 0
        raw_d = 1 / draw if draw else 0
        raw_a = 1 / away if away else 0
        total = raw_h + raw_d + raw_a
        if total == 0:
            return 0.0, 0.0, 0.0
        return raw_h / total, raw_d / total, raw_a / total

    @staticmethod
    def average_odds(bookmakers_data: list[dict], market: str = "h2h") -> dict:
        """
        Calcola la media delle quote tra più bookmaker per un evento.
        Ritorna {"home": avg, "draw": avg, "away": avg}
        """
        home_odds, draw_odds, away_odds = [], [], []

        for bm in bookmakers_data:
            for m in bm.get("markets", []):
                if m["key"] != market:
                    continue
                outcomes = {o["name"]: o["price"] for o in m.get("outcomes", [])}
                # Le chiavi possono essere i nomi delle squadre o H/D/A
                values = list(outcomes.values())
                if len(values) == 3:
                    home_odds.append(values[0])
                    draw_odds.append(values[1])
                    away_odds.append(values[2])
                elif len(values) == 2:   # no draw (non dovrebbe per calcio)
                    home_odds.append(values[0])
                    away_odds.append(values[1])

        def safe_avg(lst):
            return round(sum(lst) / len(lst), 3) if lst else None

        return {
            "home": safe_avg(home_odds),
            "draw": safe_avg(draw_odds),
            "away": safe_avg(away_odds),
        }

    # ── Sync DB ───────────────────────────────────────────────

    def sync_odds_for_competition(self, competition_code: str) -> int:
        """
        Scarica le quote per una competizione e le salva nel DB.
        Collega ogni evento al Match corrispondente usando home/away team name.
        """
        sport_key = SPORT_KEYS.get(competition_code)
        if not sport_key:
            logger.warning(f"Sport key non trovata per {competition_code}")
            return 0

        logger.info(
            f"Fetching quote {competition_code} "
            f"(req rimanenti: {self.requests_remaining})..."
        )
        events = self.get_all_markets(sport_key)
        count  = 0

        with get_db_session() as db:
            for event in events:
                home_name = event.get("home_team", "")
                away_name = event.get("away_team", "")
                commence  = event.get("commence_time")

                # Trova il match nel DB per nome squadra + data approssimativa
                match = self._find_match(db, home_name, away_name, commence)
                if not match:
                    logger.debug(f"Match non trovato in DB: {home_name} vs {away_name}")
                    continue

                bookmakers = event.get("bookmakers", [])

                # ── 1X2 ──
                h2h_avg = self.average_odds(bookmakers, "h2h")
                if h2h_avg["home"] and h2h_avg["draw"] and h2h_avg["away"]:
                    impl_h, impl_d, impl_a = self.implied_probs(
                        h2h_avg["home"], h2h_avg["draw"], h2h_avg["away"]
                    )
                    odds_row = Odds(
                        match_id=match.id,
                        bookmaker="average",
                        market="1x2",
                        home_win=h2h_avg["home"],
                        draw=h2h_avg["draw"],
                        away_win=h2h_avg["away"],
                        impl_home=round(impl_h * 100, 1),
                        impl_draw=round(impl_d * 100, 1),
                        impl_away=round(impl_a * 100, 1),
                    )
                    db.add(odds_row)
                    count += 1

                # ── BTTS ──
                btts_avg = self.average_odds(bookmakers, "btts")
                if btts_avg.get("home"):
                    odds_btts = Odds(
                        match_id=match.id,
                        bookmaker="average",
                        market="btts",
                        btts_yes=btts_avg.get("home"),   # "Yes" mappato su home
                        btts_no=btts_avg.get("away"),
                    )
                    db.add(odds_btts)

                # ── Over/Under ──
                ou_avg = self.average_odds(bookmakers, "totals")
                if ou_avg.get("home"):
                    odds_ou = Odds(
                        match_id=match.id,
                        bookmaker="average",
                        market="over_under",
                        over_25=ou_avg.get("home"),
                        under_25=ou_avg.get("away"),
                    )
                    db.add(odds_ou)

        logger.info(
            f"Sincronizzate quote per {count} partite di {competition_code}. "
            f"Req rimanenti: {self.requests_remaining}"
        )
        return count

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _find_match(db, home_name: str, away_name: str, commence_time) -> Optional[Match]:
        """
        Cerca il match nel DB facendo un join sui nomi delle squadre.
        Usa ILIKE per gestire piccole differenze di nome.
        """
        from sqlalchemy import and_, func
        from datetime import timedelta

        if not home_name or not away_name:
            return None

        # Finestra temporale ±2 giorni per gestire fuso orario
        try:
            kickoff = datetime.fromisoformat(str(commence_time).replace("Z", "+00:00"))
            date_min = kickoff - timedelta(days=1)
            date_max = kickoff + timedelta(days=1)
        except Exception:
            date_min = date_max = None

        HomeTeam = Team.__table__.alias("home_t")
        AwayTeam = Team.__table__.alias("away_t")

        query = (
            db.query(Match)
            .join(HomeTeam, Match.home_team_id == HomeTeam.c.id)
            .join(AwayTeam, Match.away_team_id == AwayTeam.c.id)
            .filter(
                func.lower(HomeTeam.c.name).contains(home_name.lower()[:8])
            )
            .filter(
                func.lower(AwayTeam.c.name).contains(away_name.lower()[:8])
            )
        )

        if date_min and date_max:
            query = query.filter(
                and_(Match.kickoff >= date_min, Match.kickoff <= date_max)
            )

        return query.first()


# Istanza singleton
odds_client = OddsClient()
