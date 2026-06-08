"""
scraper/api_football_client.py
Client per API-Football (via RapidAPI).
Piano free: 100 req/giorno. Piano Pro: illimitato ~$10/mese.

Docs: https://www.api-football.com/documentation-v3
"""
import logging
from datetime import datetime
from typing import Optional

import httpx

from config import settings
from db.database import get_db_session
from db.models import Match, Team, Player, Lineup, LineupPlayer, Injury, PlayerStats

logger = logging.getLogger(__name__)

BASE_URL = "https://api-football-v1.p.rapidapi.com/v3"

# Mappa league_id API-Football → nostro codice
LEAGUE_MAP = {
    135: "serie_a",
    2:   "champions_league",
    39:  "premier_league",
    140: "la_liga",
    78:  "bundesliga",
    61:  "ligue_1",
    3:   "europa_league",
}

LEAGUE_IDS = {v: k for k, v in LEAGUE_MAP.items()}


class APIFootballClient:
    def __init__(self):
        self.headers = {
            "X-RapidAPI-Key": settings.API_FOOTBALL_KEY,
            "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com",
        }

    def _get(self, endpoint: str, params: dict = None) -> dict:
        url = f"{BASE_URL}/{endpoint}"
        try:
            with httpx.Client(timeout=30) as client:
                resp = client.get(url, headers=self.headers, params=params or {})
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"API-Football HTTP {e.response.status_code}: {e}")
            raise
        except Exception as e:
            logger.error(f"API-Football errore {url}: {e}")
            raise

    # ── Partite ───────────────────────────────────────────────

    def get_fixtures(
        self,
        league_id: int,
        season: int,
        date: Optional[str] = None,
        status: Optional[str] = None,
    ) -> list[dict]:
        """
        Partite di una competizione/stagione.
        status: "NS" (not started) | "LIVE" | "FT" | "HT" ecc.
        """
        params = {"league": league_id, "season": season}
        if date:
            params["date"] = date
        if status:
            params["status"] = status
        data = self._get("fixtures", params)
        return data.get("response", [])

    def get_fixture_by_id(self, fixture_id: int) -> dict:
        data = self._get("fixtures", {"id": fixture_id})
        resp = data.get("response", [])
        return resp[0] if resp else {}

    def get_live_fixtures(self) -> list[dict]:
        """Tutte le partite live in questo momento."""
        data = self._get("fixtures", {"live": "all"})
        return data.get("response", [])

    def get_today_fixtures(self, league_id: Optional[int] = None) -> list[dict]:
        today = datetime.utcnow().strftime("%Y-%m-%d")
        params = {"date": today}
        if league_id:
            params["league"] = league_id
            params["season"] = datetime.utcnow().year
        data = self._get("fixtures", params)
        return data.get("response", [])

    # ── Formazioni ────────────────────────────────────────────

    def get_lineups(self, fixture_id: int) -> list[dict]:
        """
        Formazioni ufficiali di una partita.
        Disponibili ~60 minuti prima del calcio d'inizio.
        Ritorna lista con 2 elementi (home + away).
        """
        data = self._get("fixtures/lineups", {"fixture": fixture_id})
        return data.get("response", [])

    def save_lineup_to_db(self, fixture_id: int, match_id: int) -> bool:
        """
        Scarica e salva le formazioni ufficiali per una partita.
        Ritorna True se le formazioni erano disponibili.
        """
        lineups_data = self.get_lineups(fixture_id)
        if not lineups_data:
            logger.info(f"Formazioni non ancora disponibili per fixture {fixture_id}")
            return False

        with get_db_session() as db:
            for lineup_data in lineups_data:
                team_ext_id = str(lineup_data["team"]["id"])
                team = db.query(Team).filter_by(ext_id=team_ext_id).first()
                if not team:
                    logger.warning(f"Team {team_ext_id} non trovato in DB")
                    continue

                # Controlla se esiste già
                existing = db.query(Lineup).filter_by(
                    match_id=match_id,
                    team_id=team.id,
                    is_official=True,
                ).first()
                if existing:
                    continue

                lineup = Lineup(
                    match_id=match_id,
                    team_id=team.id,
                    formation=lineup_data.get("formation", ""),
                    is_official=True,
                    source="api_football",
                )
                db.add(lineup)
                db.flush()

                # Titolari
                for p in lineup_data.get("startXI", []):
                    pdata = p["player"]
                    player = self._get_or_create_player(db, pdata, team.id)
                    lp = LineupPlayer(
                        lineup_id=lineup.id,
                        player_id=player.id,
                        role="starter",
                        position=pdata.get("pos", ""),
                        shirt_num=pdata.get("number"),
                    )
                    db.add(lp)

                # Panchina
                for p in lineup_data.get("substitutes", []):
                    pdata = p["player"]
                    player = self._get_or_create_player(db, pdata, team.id)
                    lp = LineupPlayer(
                        lineup_id=lineup.id,
                        player_id=player.id,
                        role="substitute",
                        position=pdata.get("pos", ""),
                        shirt_num=pdata.get("number"),
                    )
                    db.add(lp)

                logger.info(f"✓ Formazione ufficiale salvata: {team.name}")

        return True

    # ── Statistiche partita ───────────────────────────────────

    def get_fixture_stats(self, fixture_id: int) -> list[dict]:
        """
        Statistiche live/post-gara: possesso, tiri, corner, falli ecc.
        """
        data = self._get("fixtures/statistics", {"fixture": fixture_id})
        return data.get("response", [])

    def get_fixture_events(self, fixture_id: int) -> list[dict]:
        """
        Tutti gli eventi: gol, cartellini, sostituzioni con minuto.
        """
        data = self._get("fixtures/events", {"fixture": fixture_id})
        return data.get("response", [])

    def get_player_fixture_stats(self, fixture_id: int) -> list[dict]:
        """Statistiche individuali per una partita."""
        data = self._get("fixtures/players", {"fixture": fixture_id})
        return data.get("response", [])

    # ── Infortuni ─────────────────────────────────────────────

    def get_injuries(
        self, league_id: int, season: int, fixture_id: Optional[int] = None
    ) -> list[dict]:
        """
        Infortuni e squalifiche.
        Se passi fixture_id ottieni quelli specifici per quella partita.
        """
        params = {"league": league_id, "season": season}
        if fixture_id:
            params["fixture"] = fixture_id
        data = self._get("injuries", params)
        return data.get("response", [])

    def sync_injuries(self, league_id: int, season: int) -> int:
        """Sincronizza infortuni nel DB."""
        injuries_data = self.get_injuries(league_id, season)
        count = 0

        with get_db_session() as db:
            for item in injuries_data:
                player_data = item["player"]
                ext_id = str(player_data["id"])
                player = db.query(Player).filter_by(ext_id=ext_id).first()
                if not player:
                    continue

                inj = Injury(
                    player_id=player.id,
                    type=item.get("type", "Unknown"),
                    reason=item.get("reason", ""),
                    start_date=self._parse_date(item.get("fixture", {}).get("date")),
                    is_active=True,
                    source="api_football",
                )
                db.add(inj)
                count += 1

        logger.info(f"Sincronizzati {count} infortuni")
        return count

    # ── Statistiche giocatori (stagione) ─────────────────────

    def get_player_season_stats(
        self, player_id: int, season: int, league_id: int
    ) -> dict:
        params = {"id": player_id, "season": season, "league": league_id}
        data = self._get("players", params)
        resp = data.get("response", [])
        return resp[0] if resp else {}

    def sync_team_player_stats(
        self, team_id_ext: str, league_id: int, season: int
    ) -> int:
        """
        Scarica e salva le stat stagionali di tutti i giocatori di una squadra.
        """
        params = {"team": team_id_ext, "league": league_id, "season": season}
        data = self._get("players", params)
        players_data = data.get("response", [])
        count = 0

        with get_db_session() as db:
            team = db.query(Team).filter_by(ext_id=team_id_ext).first()
            if not team:
                return 0

            for item in players_data:
                p_data  = item["player"]
                s_list  = item.get("statistics", [{}])
                stats   = s_list[0] if s_list else {}

                ext_id  = str(p_data["id"])
                player  = db.query(Player).filter_by(ext_id=ext_id).first()
                if not player:
                    player = Player(
                        ext_id=ext_id,
                        team_id=team.id,
                        name=p_data["name"],
                        position=stats.get("games", {}).get("position", ""),
                        nationality=p_data.get("nationality", ""),
                        birth_date=self._parse_date(
                            p_data.get("birth", {}).get("date")
                        ),
                    )
                    db.add(player)
                    db.flush()

                games    = stats.get("games", {})
                goals    = stats.get("goals", {})
                cards    = stats.get("cards", {})
                shots_s  = stats.get("shots", {})
                dribbles = stats.get("dribbles", {})
                tackles  = stats.get("tackles", {})
                passes   = stats.get("passes", {})

                ps = PlayerStats(
                    player_id=player.id,
                    competition_id=None,
                    season=str(season),
                    appearances=games.get("appearences") or 0,
                    minutes=games.get("minutes") or 0,
                    goals=goals.get("total") or 0,
                    assists=goals.get("assists") or 0,
                    yellow_cards=cards.get("yellow") or 0,
                    red_cards=cards.get("red") or 0,
                    shots=shots_s.get("total") or 0,
                    shots_on_target=shots_s.get("on") or 0,
                    dribbles_succ=dribbles.get("success") or 0,
                    tackles=tackles.get("total") or 0,
                    key_passes=passes.get("key") or 0,
                )
                db.add(ps)
                count += 1

        return count

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _get_or_create_player(db, pdata: dict, team_id: int) -> Player:
        ext_id = str(pdata.get("id", "")) if pdata.get("id") else None
        if ext_id:
            player = db.query(Player).filter_by(ext_id=ext_id).first()
            if player:
                return player

        player = Player(
            ext_id=ext_id,
            team_id=team_id,
            name=pdata.get("name", "Unknown"),
            shirt_number=pdata.get("number"),
        )
        db.add(player)
        db.flush()
        return player

    @staticmethod
    def _parse_date(date_str) -> Optional[datetime]:
        if not date_str:
            return None
        try:
            return datetime.fromisoformat(str(date_str).replace("Z", "+00:00"))
        except ValueError:
            return None


api_football = APIFootballClient()
