"""
nlp/transfer_analyzer.py
Analisi NLP delle notizie di mercato — versione senza spaCy.

Usa esclusivamente regex per l'estrazione entità, eliminando
il conflitto spaCy/typer/fastapi-cli.
La qualità è equivalente per notizie di mercato calcistico
perché le fonti (Romano, Sky, TMW) scrivono in modo strutturato.
"""
import logging
import re
from datetime import datetime, timedelta
from typing import Optional

from db.database import get_db_session
from db.models import NewsItem, Transfer, Player, Team
from scraper.transfer_scraper import SOURCE_WEIGHTS, CONFIRMATION_PATTERNS, RUMOR_PATTERNS

logger = logging.getLogger(__name__)


# ─── Dizionario alias squadre ─────────────────────────────────
TEAM_ALIASES: dict[str, str] = {
    # Serie A
    "milan": "AC Milan", "rossoneri": "AC Milan",
    "inter": "Inter", "nerazzurri": "Inter", "internazionale": "Inter",
    "juve": "Juventus", "juventus": "Juventus", "bianconeri": "Juventus",
    "napoli": "Napoli", "partenopei": "Napoli",
    "roma": "AS Roma", "giallorossi": "AS Roma",
    "lazio": "Lazio", "biancocelesti": "Lazio",
    "atalanta": "Atalanta", "fiorentina": "Fiorentina",
    "viola": "Fiorentina", "torino": "Torino",
    "bologna": "Bologna", "monza": "Monza",
    # Premier
    "city": "Manchester City", "man city": "Manchester City",
    "united": "Manchester United", "man utd": "Manchester United",
    "chelsea": "Chelsea", "arsenal": "Arsenal",
    "liverpool": "Liverpool", "tottenham": "Tottenham",
    "spurs": "Tottenham", "newcastle": "Newcastle",
    # La Liga
    "real": "Real Madrid", "real madrid": "Real Madrid",
    "blancos": "Real Madrid", "barca": "Barcellona",
    "barcellona": "Barcellona", "atletico": "Atletico Madrid",
    # Bundesliga
    "bayern": "Bayern Monaco", "dortmund": "Borussia Dortmund",
    "bvb": "Borussia Dortmund", "leverkusen": "Bayer Leverkusen",
    # Ligue 1
    "psg": "PSG", "paris": "PSG",
    # Altro
    "porto": "Porto", "benfica": "Benfica",
    "ajax": "Ajax", "sporting": "Sporting CP",
}

# Pattern per estrarre il nome del giocatore dal testo
PLAYER_NAME_PATTERNS = [
    # "Mbappé al Real Madrid" / "Osimhen verso il PSG"
    r"^([A-ZÁÉÍÓÚÀÈÌÒÙÑÇ][a-záéíóúàèìòùñç\-\']+(?:\s+[A-ZÁÉÍÓÚÀÈÌÒÙÑÇ][a-záéíóúàèìòùñç\-\']+){1,3})\s+(?:al|verso|a|in)",
    # "Il Milan vuole Leao" → prende l'ultimo nome proprio dopo verbo
    r"(?:vuole|cerca|offre|punta su|tratta|segue)\s+([A-ZÁÉÍÓÚÀÈÌÒÙÑÇ][a-záéíóúàèìòùñç\-\']+(?:\s+[A-ZÁÉÍÓÚÀÈÌÒÙÑÇ][a-záéíóúàèìòùñç\-\']+)?)",
    # "Kvaratskhelia: accordo vicino" → nome prima dei due punti
    r"^([A-ZÁÉÍÓÚÀÈÌÒÙÑÇ][a-záéíóúàèìòùñç\-\']+(?:\s+[A-ZÁÉÍÓÚÀÈÌÒÙÑÇ][a-záéíóúàèìòùñç\-\']+){1,2})\s*:",
    # Nomi tutto maiuscolo stile Romano → "MBAPPÉ to Real Madrid"
    r"^([A-ZÁÉÍÓÚÀÈÌÒÙÑÇ]{2,}(?:\s+[A-ZÁÉÍÓÚÀÈÌÒÙÑÇ]{2,})?)\s+to\s+",
]


class TransferAnalyzer:

    def process_news_item(self, item: NewsItem) -> Optional[Transfer]:
        """Processa una notizia e aggiorna/crea il Transfer nel DB."""
        text = f"{item.title} {item.body}"

        player_name = self._extract_player(text)
        from_team   = self._extract_team(text, role="from")
        to_team     = self._extract_team(text, role="to")

        if not player_name:
            self._mark_processed(item.id)
            return None

        intensity  = self._score_intensity(text, item.source)
        here_we_go = bool(re.search(r"here we go", text, re.I))

        transfer = self._upsert_transfer(
            player_name=player_name,
            from_team=from_team or "",
            to_team=to_team or "",
            source=item.source,
            intensity=intensity,
            here_we_go=here_we_go,
            news_url=item.url or "",
            news_title=item.title or "",
            published=item.published or datetime.utcnow(),
        )

        self._mark_processed(item.id)
        return transfer

    # ── Estrazione entità via regex ───────────────────────────

    def _extract_player(self, text: str) -> Optional[str]:
        """Estrae il nome del giocatore con pattern regex."""
        for pattern in PLAYER_NAME_PATTERNS:
            m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if m:
                candidate = m.group(1).strip()
                # Esclude se è il nome di una squadra
                if candidate.lower() not in TEAM_ALIASES:
                    return self._normalize_name(candidate)

        # Fallback: primo nome proprio maiuscolo che non è una squadra nota
        names = re.findall(
            r"\b([A-ZÁÉÍÓÚÀÈÌÒÙÑÇ][a-záéíóúàèìòùñç\-\']+(?:\s+[A-ZÁÉÍÓÚÀÈÌÒÙÑÇ][a-záéíóúàèìòùñç\-\']+){1,2})\b",
            text
        )
        team_words = set(TEAM_ALIASES.keys())
        for name in names:
            if name.lower() not in team_words and len(name) > 4:
                return self._normalize_name(name)

        return None

    def _extract_team(self, text: str, role: str = "to") -> Optional[str]:
        """Estrae squadra di destinazione o provenienza."""
        text_lower = text.lower()

        if role == "to":
            patterns = [
                r"(?:al|verso|a)\s+([A-ZÁÉÍÓÚÀÈÌÒÙÑÇ][a-záéíóúàèìòùñç\s]+?)(?:\s|,|\.|per|a)",
                r"(?:to|joining|signs for)\s+([A-ZÁÉÍÓÚÀÈÌÒÙÑÇ][a-záéíóúàèìòùñç\s]+?)(?:\s|,|\.)",
                r"(?:destinazione|direzione)\s+([A-ZÁÉÍÓÚÀÈÌÒÙÑÇ][a-záéíóúàèìòùñç\s]+?)(?:\s|,|\.)",
            ]
        else:
            patterns = [
                r"(?:dal|lascia|ex)\s+([A-ZÁÉÍÓÚÀÈÌÒÙÑÇ][a-záéíóúàèìòùñç\s]+?)(?:\s|,|\.)",
                r"(?:from|leaving)\s+([A-ZÁÉÍÓÚÀÈÌÒÙÑÇ][a-záéíóúàèìòùñç\s]+?)(?:\s|,|\.)",
            ]

        for pattern in patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                raw = m.group(1).strip().rstrip(".,;:")
                normalized = self._normalize_team(raw)
                if normalized:
                    return normalized

        # Cerca alias direttamente nel testo
        for alias, canonical in sorted(TEAM_ALIASES.items(), key=lambda x: -len(x[0])):
            if alias in text_lower:
                return canonical

        return None

    # ── Scoring intensità ─────────────────────────────────────

    def _score_intensity(self, text: str, source: str) -> float:
        score = 0.0
        text_lower = text.lower()

        source_weight = SOURCE_WEIGHTS.get(source, 1.0)
        score += min(source_weight / 3.0, 1.0) * 0.3

        conf_hits = sum(
            1 for p in CONFIRMATION_PATTERNS
            if re.search(p, text_lower, re.I)
        )
        score += min(conf_hits * 0.15, 0.45)

        if re.search(r"here we go", text_lower, re.I):
            score += 0.35

        if re.search(r"€\s*\d+|\$\s*\d+|\d+\s*(?:milioni|million)", text_lower):
            score += 0.08
        if re.search(r"\d+\s*(?:anni|years)\s*(?:di contratto|contract)", text_lower):
            score += 0.07

        rumor_hits = sum(
            1 for p in RUMOR_PATTERNS
            if re.search(p, text_lower, re.I)
        )
        score -= min(rumor_hits * 0.05, 0.2)

        return max(0.0, min(1.0, score))

    # ── Calcolo probabilità ───────────────────────────────────

    def calculate_probability(self, transfer, all_news: list[dict]) -> float:
        if not all_news:
            return 0.0

        now = datetime.utcnow()
        weighted_sum = 0.0
        weight_total = 0.0
        has_here_we_go = False

        for news in all_news:
            source    = news.get("source", "unknown")
            intensity = news.get("intensity", 0.3)
            published = news.get("published", now)
            hwg       = news.get("here_we_go", False)

            if hwg:
                has_here_we_go = True

            age_days   = (now - published).total_seconds() / 86400
            time_decay = max(0.1, 1.0 - (age_days / 30.0))
            source_w   = SOURCE_WEIGHTS.get(source, 1.0)
            w = source_w * time_decay
            weighted_sum  += intensity * w
            weight_total  += w

        if weight_total == 0:
            return 0.0

        base_prob = (weighted_sum / weight_total) * 100

        if has_here_we_go:
            base_prob = min(99.0, base_prob * 1.5 + 30)

        n_sources    = len(set(n.get("source") for n in all_news))
        source_boost = min(n_sources * 3.0, 15.0)
        base_prob    = min(99.0, base_prob + source_boost)

        return round(base_prob, 1)

    # ── Generazione spiegazione ───────────────────────────────

    def generate_explanation(
        self, player_name, from_team, to_team,
        probability, sources, news_items
    ) -> str:
        lines = []

        if probability >= 90:
            lines.append("Trattativa in fase conclusiva.")
        elif probability >= 70:
            lines.append(f"Trattativa avanzata tra {from_team or '?'} e {to_team or '?'}.")
        elif probability >= 40:
            lines.append(f"Interesse concreto di {to_team or 'un club'} per {player_name}.")
        else:
            lines.append(f"Circolano voci su un possibile interesse per {player_name}.")

        for news in news_items:
            text = news.get("body", "") or news.get("title", "")
            fee_m = re.search(r"€\s*(\d+(?:[,\.]\d+)?)\s*(?:M|milion[ie]|million)?", text, re.I)
            if fee_m:
                lines.append(f"Cifre citate: {fee_m.group(0)}.")
                break

        if any(n.get("here_we_go") for n in news_items):
            lines.append("Fabrizio Romano ha confermato con il suo 'here we go'.")

        src_names = {
            "romano_twitter":  "Fabrizio Romano (Twitter)",
            "romano_substack": "Fabrizio Romano (Substack)",
            "sky_sport":       "Sky Sport",
            "tmw":             "TuttoMercatoWeb",
            "calciomercato":   "Calciomercato.com",
            "transfermarkt":   "Transfermarkt",
            "gazzetta":        "La Gazzetta dello Sport",
        }
        unique = list(set(sources))
        lines.append(f"Fonti: {', '.join(src_names.get(s, s) for s in unique)}.")

        return " ".join(lines)

    # ── Upsert Transfer ───────────────────────────────────────

    def _upsert_transfer(
        self, player_name, from_team, to_team,
        source, intensity, here_we_go,
        news_url, news_title, published
    ) -> Transfer:
        with get_db_session() as db:
            from sqlalchemy import func
            transfer = db.query(Transfer).filter(
                func.lower(Transfer.player_name).contains(
                    player_name.lower().split()[-1]
                )
            ).first()

            existing_news = []
            if transfer and transfer.sources:
                existing_news = transfer.sources if isinstance(transfer.sources, list) else []

            new_entry = {
                "source": source, "title": news_title[:200],
                "url": news_url, "intensity": intensity,
                "here_we_go": here_we_go,
                "published": published.isoformat(),
            }
            all_news = existing_news + [new_entry]

            all_news_dt = []
            for n in all_news:
                nc = dict(n)
                if isinstance(nc.get("published"), str):
                    try:
                        nc["published"] = datetime.fromisoformat(nc["published"])
                    except ValueError:
                        nc["published"] = datetime.utcnow()
                all_news_dt.append(nc)

            probability = self.calculate_probability(transfer, all_news_dt)

            if here_we_go or probability >= 92:
                status = "confirmed"
            elif probability >= 65:
                status = "advanced"
            else:
                status = "rumor"

            explanation = self.generate_explanation(
                player_name, from_team, to_team,
                probability, [n["source"] for n in all_news],
                all_news_dt,
            )

            if transfer:
                transfer.probability  = probability
                transfer.status       = status
                transfer.here_we_go   = transfer.here_we_go or here_we_go
                transfer.sources      = all_news[-20:]
                transfer.to_team      = to_team or transfer.to_team
                transfer.from_team    = from_team or transfer.from_team
                transfer.detail       = explanation
                transfer.updated_at   = datetime.utcnow()
            else:
                transfer = Transfer(
                    player_name=player_name,
                    from_team=from_team,
                    to_team=to_team,
                    probability=probability,
                    status=status,
                    here_we_go=here_we_go,
                    detail=explanation,
                    sources=all_news[-20:],
                )
                db.add(transfer)

            logger.info(f"Transfer: {player_name} → {to_team} ({probability}%)")
            return transfer

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _normalize_name(name: str) -> str:
        name = re.sub(r"\s+", " ", name).strip()
        return " ".join(w.capitalize() for w in name.split())

    @staticmethod
    def _normalize_team(name: str) -> Optional[str]:
        if not name or len(name) < 2:
            return None
        key = name.lower().strip()
        if key in TEAM_ALIASES:
            return TEAM_ALIASES[key]
        for alias, canonical in TEAM_ALIASES.items():
            if alias in key:
                return canonical
        return name.title() if len(name) > 3 else None

    @staticmethod
    def _mark_processed(news_id: int):
        with get_db_session() as db:
            item = db.query(NewsItem).filter_by(id=news_id).first()
            if item:
                item.processed = True

    def process_all_unprocessed(self, limit: int = 100) -> int:
        with get_db_session() as db:
            items = (
                db.query(NewsItem)
                .filter_by(processed=False)
                .order_by(NewsItem.published.desc())
                .limit(limit)
                .all()
            )

        count = 0
        for item in items:
            try:
                if self.process_news_item(item):
                    count += 1
            except Exception as e:
                logger.warning(f"Errore processing news {item.id}: {e}")

        logger.info(f"Processate {count} notizie")
        return count


transfer_analyzer = TransferAnalyzer()
