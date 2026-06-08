"""
scraper/lineup_scraper.py
Scraping delle probabili formazioni da Gazzetta.it e Fantacalcio.it.
Usa Playwright per pagine JavaScript-heavy.

Esegui prima: playwright install chromium
"""
import asyncio
import logging
import re
import time
from datetime import datetime
from typing import Optional

from bs4 import BeautifulSoup

from config import settings
from db.database import get_db_session
from db.models import Match, Team, Player, Lineup, LineupPlayer

logger = logging.getLogger(__name__)


class LineupScraper:
    """
    Scraper asincrono per probabili formazioni.
    Ogni sorgente ha il suo metodo; i dati vengono normalizzati
    in un formato comune prima di essere salvati nel DB.
    """

    # Formato comune output:
    # {
    #   "team_name": str,
    #   "formation": str,           # es. "4-3-3"
    #   "source": str,
    #   "fetched_at": datetime,
    #   "starters": [
    #       {"name": str, "position": str, "number": int, "uncertain": bool}
    #   ],
    #   "substitutes": [...]
    # }

    def __init__(self):
        self._delay = settings.SCRAPER_DELAY
        self._ua    = settings.SCRAPER_USER_AGENT

    # ── Gazzetta.it ───────────────────────────────────────────

    async def scrape_gazzetta(self, match_slug: str) -> Optional[dict]:
        """
        Scraping da Gazzetta.it.
        match_slug: parte URL es. "milan-inter-serie-a"

        La pagina è JS-heavy, usiamo Playwright.
        """
        url = f"https://www.gazzetta.it/calcio/pagellone/{match_slug}"
        logger.info(f"Scraping Gazzetta: {url}")

        try:
            from playwright.async_api import async_playwright

            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(user_agent=self._ua)
                page    = await context.new_page()

                await page.goto(url, wait_until="networkidle", timeout=30000)
                await asyncio.sleep(2)   # attesa rendering JS

                html = await page.content()
                await browser.close()

            return self._parse_gazzetta_html(html)

        except Exception as e:
            logger.error(f"Errore scraping Gazzetta {url}: {e}")
            return None

    def _parse_gazzetta_html(self, html: str) -> Optional[dict]:
        soup = BeautifulSoup(html, "lxml")
        result = {}

        # Cerca il blocco formazione (Gazzetta usa classi CSS specifiche)
        formation_blocks = soup.find_all("div", class_=re.compile(r"formation|lineup", re.I))
        if not formation_blocks:
            logger.warning("Blocco formazione non trovato in Gazzetta HTML")
            return None

        players = []
        for block in formation_blocks:
            player_divs = block.find_all("div", class_=re.compile(r"player|giocatore", re.I))
            for div in player_divs:
                name = div.get_text(strip=True)
                if name and len(name) > 2:
                    players.append({
                        "name": self._normalize_player_name(name),
                        "position": "",
                        "number": None,
                        "uncertain": "?" in name or "dubbio" in name.lower(),
                    })

        if not players:
            return None

        return {
            "team_name": "unknown",
            "formation": self._detect_formation(html),
            "source": "gazzetta",
            "fetched_at": datetime.utcnow(),
            "starters": players[:11],
            "substitutes": players[11:],
        }

    # ── Fantacalcio.it ────────────────────────────────────────

    async def scrape_fantacalcio(self, home_team: str, away_team: str) -> list[dict]:
        """
        Scraping da Fantacalcio.it — ottima fonte per probabili formazioni
        con indicazione dei giocatori incerti.

        Restituisce lista di 2 dict (home + away).
        """
        # Normalizza nomi per URL
        def slugify(s):
            return s.lower().replace(" ", "-").replace(".", "")

        url = (
            f"https://www.fantacalcio.it/probabili-formazioni-serie-a/"
            f"{slugify(home_team)}-{slugify(away_team)}"
        )
        logger.info(f"Scraping Fantacalcio: {url}")

        try:
            from playwright.async_api import async_playwright

            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(user_agent=self._ua)
                page    = await context.new_page()

                # Blocca immagini/font per velocizzare
                await page.route(
                    "**/*.{png,jpg,jpeg,gif,webp,woff,woff2}",
                    lambda route: route.abort()
                )

                await page.goto(url, wait_until="domcontentloaded", timeout=25000)
                await asyncio.sleep(1.5)

                html = await page.content()
                await browser.close()

            return self._parse_fantacalcio_html(html, home_team, away_team)

        except Exception as e:
            logger.error(f"Errore scraping Fantacalcio {url}: {e}")
            return []

    def _parse_fantacalcio_html(
        self, html: str, home_team: str, away_team: str
    ) -> list[dict]:
        soup    = BeautifulSoup(html, "lxml")
        results = []

        # Fantacalcio usa struttura a 2 colonne per le 2 squadre
        team_sections = soup.find_all("div", class_=re.compile(r"team-lineup|squadra", re.I))

        teams = [home_team, away_team]

        for i, section in enumerate(team_sections[:2]):
            starters = []
            uncertain_players = set()

            # Giocatori con dubbio hanno classe "uncertain" o icona specifica
            for player_el in section.find_all(
                "span", class_=re.compile(r"player-name|nome-giocatore", re.I)
            ):
                raw_name = player_el.get_text(strip=True)
                is_uncertain = bool(
                    player_el.find_parent(class_=re.compile(r"uncertain|dubbio", re.I))
                )

                starters.append({
                    "name": self._normalize_player_name(raw_name),
                    "position": self._extract_position(player_el),
                    "number": None,
                    "uncertain": is_uncertain,
                })

            # Modulo (es. "4-3-3")
            formation_el = section.find(
                text=re.compile(r"\d-\d-\d|\d-\d-\d-\d")
            )
            formation = formation_el.strip() if formation_el else ""

            if starters:
                results.append({
                    "team_name": teams[i] if i < len(teams) else f"team_{i}",
                    "formation": formation,
                    "source": "fantacalcio",
                    "fetched_at": datetime.utcnow(),
                    "starters": starters[:11],
                    "substitutes": starters[11:],
                })

        return results

    # ── Salvataggio DB ────────────────────────────────────────

    def save_probable_lineup(
        self, match_id: int, lineup_data: dict
    ) -> Optional[Lineup]:
        """
        Salva una formazione probabile nel DB.
        Se esiste già una formazione per questo match/team/source, la aggiorna.
        """
        with get_db_session() as db:
            match = db.query(Match).filter_by(id=match_id).first()
            if not match:
                logger.error(f"Match {match_id} non trovato")
                return None

            # Trova la squadra
            team = self._find_team(db, lineup_data["team_name"])
            if not team:
                logger.warning(f"Team non trovato: {lineup_data['team_name']}")
                return None

            # Rimuovi eventuale formazione probabile precedente dalla stessa fonte
            existing = db.query(Lineup).filter_by(
                match_id=match_id,
                team_id=team.id,
                is_official=False,
                source=lineup_data["source"],
            ).first()
            if existing:
                db.delete(existing)
                db.flush()

            lineup = Lineup(
                match_id=match_id,
                team_id=team.id,
                formation=lineup_data.get("formation", ""),
                is_official=False,
                source=lineup_data["source"],
                fetched_at=lineup_data.get("fetched_at", datetime.utcnow()),
            )
            db.add(lineup)
            db.flush()

            for p_data in lineup_data.get("starters", []):
                player = self._find_or_create_player(db, p_data["name"], team.id)
                lp = LineupPlayer(
                    lineup_id=lineup.id,
                    player_id=player.id,
                    role="starter",
                    position=p_data.get("position", ""),
                    shirt_num=p_data.get("number"),
                    is_uncertain=p_data.get("uncertain", False),
                )
                db.add(lp)

            for p_data in lineup_data.get("substitutes", []):
                player = self._find_or_create_player(db, p_data["name"], team.id)
                lp = LineupPlayer(
                    lineup_id=lineup.id,
                    player_id=player.id,
                    role="substitute",
                    position=p_data.get("position", ""),
                    shirt_num=p_data.get("number"),
                    is_uncertain=p_data.get("uncertain", False),
                )
                db.add(lp)

            logger.info(
                f"✓ Probabile formazione salvata: {team.name} "
                f"({lineup_data['source']}) [{lineup_data.get('formation','')}]"
            )
            return lineup

    # ── Fusione sorgenti ──────────────────────────────────────

    def merge_lineups(self, lineups: list[dict]) -> Optional[dict]:
        """
        Combina le formazioni da più sorgenti per la stessa squadra,
        dando priorità a: api_football > gazzetta > fantacalcio.
        I giocatori presenti in 2+ fonti sono considerati certi.
        """
        if not lineups:
            return None
        if len(lineups) == 1:
            return lineups[0]

        priority = {"api_football": 3, "gazzetta": 2, "fantacalcio": 1}
        lineups.sort(key=lambda x: priority.get(x["source"], 0), reverse=True)
        best  = lineups[0]
        other = lineups[1:]

        # Conta quante volte ogni giocatore appare
        name_count: dict[str, int] = {}
        for lineup in lineups:
            for p in lineup.get("starters", []):
                key = p["name"].lower()
                name_count[key] = name_count.get(key, 0) + 1

        # Aggiorna uncertain: se compare in 2+ sorgenti → certo
        merged_starters = []
        for p in best.get("starters", []):
            p_copy = dict(p)
            if name_count.get(p["name"].lower(), 0) >= 2:
                p_copy["uncertain"] = False
            merged_starters.append(p_copy)

        result = dict(best)
        result["starters"]  = merged_starters
        result["source"]    = "merged"
        result["sources"]   = [l["source"] for l in lineups]
        return result

    # ── Helpers privati ───────────────────────────────────────

    @staticmethod
    def _normalize_player_name(name: str) -> str:
        """Rimuove caratteri spurii e standardizza il nome."""
        name = re.sub(r"\?|\*|†|⚠", "", name)
        name = re.sub(r"\s+", " ", name).strip()
        # Rimuovi numero di maglia se presente all'inizio
        name = re.sub(r"^\d{1,2}\.\s*", "", name)
        return name.title()

    @staticmethod
    def _detect_formation(html: str) -> str:
        """Cerca il pattern del modulo nel testo HTML."""
        match = re.search(r"\b(\d[-–]\d[-–]\d(?:[-–]\d)?)\b", html)
        return match.group(1) if match else ""

    @staticmethod
    def _extract_position(element) -> str:
        """Tenta di estrarre la posizione dal contesto HTML."""
        parent = element.find_parent()
        if parent:
            data_pos = parent.get("data-position", "")
            if data_pos:
                return data_pos
        return ""

    @staticmethod
    def _find_team(db, name: str) -> Optional[Team]:
        from sqlalchemy import func
        # Cerca per nome esatto o per corrispondenza parziale
        team = db.query(Team).filter(
            func.lower(Team.name) == name.lower()
        ).first()
        if not team:
            team = db.query(Team).filter(
                func.lower(Team.name).contains(name.lower()[:5])
            ).first()
        return team

    @staticmethod
    def _find_or_create_player(db, name: str, team_id: int) -> Player:
        from sqlalchemy import func
        player = db.query(Player).filter(
            func.lower(Player.name) == name.lower(),
            Player.team_id == team_id,
        ).first()
        if not player:
            player = Player(name=name, team_id=team_id)
            db.add(player)
            db.flush()
        return player


lineup_scraper = LineupScraper()
