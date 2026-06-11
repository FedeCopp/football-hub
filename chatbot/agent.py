"""
chatbot/agent.py — LangChain 0.3.x + Groq
"""
import logging
import re
from datetime import datetime, timedelta
from typing import Optional

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

from config import settings
from db.database import get_db_session
from db.models import Match, Team, Lineup, LineupPlayer, Transfer, Prediction, Injury, Competition

logger = logging.getLogger(__name__)


def build_llm():
    provider = settings.llm_provider
    logger.info(f"LLM provider: {provider}")
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=settings.GROQ_MODEL, temperature=0.1, api_key=settings.GROQ_API_KEY)
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.1, api_key=settings.OPENAI_API_KEY)
    from langchain_community.chat_models import ChatOllama
    return ChatOllama(model=settings.OLLAMA_MODEL, base_url=settings.OLLAMA_BASE_URL, temperature=0.1)


# Alias per i nomi delle competizioni come restituiti da football-data.org,
# che spesso non corrispondono al nome comune usato dagli utenti.
COMPETITION_ALIASES: dict[str, str] = {
    "la liga": "Primera Division",
    "liga": "Primera Division",
    "spagna": "Primera Division",
    "spagnola": "Primera Division",
}


def _resolve_competition_name(competition: str) -> str:
    return COMPETITION_ALIASES.get(competition.lower().strip(), competition)


def get_system_prompt() -> str:
    now = datetime.utcnow().strftime("%d/%m/%Y %H:%M UTC")
    return (
        "Sei FootballHub AI, assistente esperto di calcio.\n\n"
        "REGOLE ASSOLUTE:\n"
        "1. Usa SEMPRE i tools per rispondere. MAI inventare partite, risultati, date o notizie.\n"
        "2. Se un tool restituisce dati vuoti o 'nessuna partita', rispondi SOLO con quello che il tool ha restituito. Non aggiungere partite inventate.\n"
        "3. Non mostrare mai tag XML, function= o simili nel testo.\n"
        "4. Rispondi in italiano.\n"
        f"5. Data attuale: {now}\n\n"
        "Il database contiene: Serie A, Premier League, La Liga, Champions League stagioni 2023/24, 2024/25, 2025/26.\n"
        "Se il database non ha dati per una domanda, dillo chiaramente senza inventare."
    )


@tool
def get_matches(days_back: int = 7, days_forward: int = 30, competition: str = "") -> str:
    """
    Cerca partite nel database. 
    days_back: quanti giorni indietro cercare (default 7)
    days_forward: quanti giorni avanti cercare (default 30)
    competition: filtra per competizione (es. 'Serie A', 'Premier League')
    Usa questo tool per qualsiasi domanda su partite passate o future.
    """
    from sqlalchemy import func
    now = datetime.utcnow()
    date_from = now - timedelta(days=days_back)
    date_to = now + timedelta(days=days_forward)

    with get_db_session() as db:
        query = db.query(Match).filter(
            Match.kickoff >= date_from,
            Match.kickoff <= date_to,
        )
        if competition:
            comp_name = _resolve_competition_name(competition)
            query = query.join(Competition).filter(
                func.lower(Competition.name).contains(comp_name.lower())
            )
        matches = query.order_by(Match.kickoff).limit(20).all()

        if not matches:
            return f"Nessuna partita trovata nel database per il periodo richiesto."

        lines = []
        for m in matches:
            home = m.home_team.short_name or m.home_team.name
            away = m.away_team.short_name or m.away_team.name
            comp = m.competition.name if m.competition else "?"
            d = m.kickoff.strftime("%d/%m/%Y %H:%M") if m.kickoff else "?"
            if m.status == "finished":
                lines.append(f"[FINITA] {comp}: {home} {m.home_score}-{m.away_score} {away} ({d})")
            elif m.status in ("live", "in_play"):
                lines.append(f"[LIVE] {comp}: {home} {m.home_score or 0}-{m.away_score or 0} {away}")
            else:
                lines.append(f"[PROGRAMMA] {comp}: {home} vs {away} ({d})")

        return "\n".join(lines)


@tool
def get_lineup(team_name: str) -> str:
    """Formazione probabile o ufficiale di una squadra per la prossima partita."""
    from sqlalchemy import func, or_
    with get_db_session() as db:
        team = db.query(Team).filter(func.lower(Team.name).contains(team_name.lower())).first()
        if not team:
            return f"Squadra '{team_name}' non trovata nel database."
        now = datetime.utcnow()
        match = db.query(Match).filter(
            or_(Match.home_team_id == team.id, Match.away_team_id == team.id),
            Match.kickoff >= now - timedelta(hours=2),
            Match.kickoff <= now + timedelta(days=7),
        ).order_by(Match.kickoff).first()
        if not match:
            return f"Nessuna partita in programma per {team.name} nei prossimi 7 giorni."
        lineup = db.query(Lineup).filter_by(match_id=match.id, team_id=team.id).order_by(Lineup.is_official.desc()).first()
        if not lineup:
            return f"Formazione non ancora disponibile per {team.name}."
        starters = db.query(LineupPlayer).filter_by(lineup_id=lineup.id, role="starter").all()
        tipo = "UFFICIALE" if lineup.is_official else "Probabile"
        opp_id = match.away_team_id if match.home_team_id == team.id else match.home_team_id
        opp = db.query(Team).filter_by(id=opp_id).first()
        lines = [f"{tipo} - {team.name} vs {opp.name if opp else '?'} ({lineup.formation or '?'})", ""]
        for i, lp in enumerate(starters, 1):
            name = lp.player.name if lp.player else "N/D"
            unc = " (dubbio)" if lp.is_uncertain else ""
            lines.append(f"{i:2}. {name}{unc}")
        return "\n".join(lines)


@tool
def get_prediction(home_team: str, away_team: str) -> str:
    """Previsione ML per una partita specifica. Cerca la partita nel DB e mostra le probabilità."""
    from sqlalchemy import func
    with get_db_session() as db:
        home = db.query(Team).filter(func.lower(Team.name).contains(home_team.lower())).first()
        away = db.query(Team).filter(func.lower(Team.name).contains(away_team.lower())).first()
        if not home or not away:
            missing = home_team if not home else away_team
            return f"Squadra '{missing}' non trovata nel database."
        now = datetime.utcnow()
        match = db.query(Match).filter(
            Match.home_team_id == home.id, Match.away_team_id == away.id,
            Match.kickoff >= now - timedelta(days=1),
            Match.kickoff <= now + timedelta(days=90),
        ).order_by(Match.kickoff).first()
        if not match:
            return f"Nessuna partita trovata tra {home.name} e {away.name} nei prossimi 90 giorni."
        pred = db.query(Prediction).filter_by(match_id=match.id).first()
        if not pred or pred.prob_home == 33.3:
            return (
                f"Partita trovata: {home.name} vs {away.name} ({match.kickoff.strftime('%d/%m/%Y') if match.kickoff else '?'})\n"
                f"Previsione ML non disponibile — il modello deve essere addestrato con i dati storici."
            )
        d = match.kickoff.strftime("%d/%m/%Y") if match.kickoff else "?"
        lines = [
            f"Previsione: {home.name} vs {away.name} ({d})",
            f"1 {home.short_name or home.name}: {pred.prob_home:.1f}%",
            f"X Pareggio: {pred.prob_draw:.1f}%",
            f"2 {away.short_name or away.name}: {pred.prob_away:.1f}%",
        ]
        if pred.btts_prob:
            lines.append(f"BTTS: {pred.btts_prob:.1f}%  Over 2.5: {pred.over25_prob:.1f}%")
        if pred.scorer_probs:
            top = sorted(pred.scorer_probs, key=lambda x: -x.get("prob", 0))[:3]
            lines.append("Marcatori probabili: " + ", ".join(f"{s['name']} {s['prob']:.0f}%" for s in top))
        return "\n".join(lines)


@tool
def get_transfers(query: str) -> str:
    """Rumors calciomercato. Cerca per nome giocatore o squadra."""
    from sqlalchemy import func, or_
    with get_db_session() as db:
        transfers = db.query(Transfer).filter(
            or_(
                func.lower(Transfer.player_name).contains(query.lower()),
                func.lower(Transfer.from_team).contains(query.lower()),
                func.lower(Transfer.to_team).contains(query.lower()),
            )
        ).order_by(Transfer.probability.desc()).limit(5).all()
        if not transfers:
            return f"Nessun rumor di mercato trovato per '{query}' nel database. I rumors vengono aggiornati dallo scraper delle notizie."
        lines = []
        for t in transfers:
            hwg = " HERE WE GO!" if t.here_we_go else ""
            prob = f"{t.probability:.0f}%" if t.probability else "?"
            lines.append(f"{t.player_name}{hwg}: {t.from_team or '?'} -> {t.to_team or '?'} ({prob})")
            if t.detail:
                lines.append(f"  {t.detail[:200]}")
        return "\n".join(lines)


@tool
def get_injuries(team_name: str) -> str:
    """Infortuni e squalifiche attive di una squadra."""
    from sqlalchemy import func
    from db.models import Player
    with get_db_session() as db:
        team = db.query(Team).filter(func.lower(Team.name).contains(team_name.lower())).first()
        if not team:
            return f"Squadra '{team_name}' non trovata."
        injuries = db.query(Injury).join(Player).filter(
            Player.team_id == team.id, Injury.is_active == True
        ).all()
        if not injuries:
            return f"{team.name}: nessun infortunio attivo nel database."
        return "\n".join(f"- {i.player.name}: {i.type} — {i.reason}" for i in injuries if i.player)


@tool
def get_standings(competition: str, season: str = "") -> str:
    """
    Classifica di una competizione.
    competition: es. 'Serie A', 'Premier League', 'Champions League'
    season: es. '2025' per stagione 2025/26, lascia vuoto per stagione corrente
    """
    from sqlalchemy import func
    comp_name = _resolve_competition_name(competition)
    with get_db_session() as db:
        comp = db.query(Competition).filter(
            func.lower(Competition.name).contains(comp_name.lower())
        ).first()
        if not comp:
            return f"Competizione '{competition}' non trovata. Disponibili: Serie A, Premier League, La Liga, Champions League."

        now = datetime.utcnow()
        # Determina stagione
        if season:
            try:
                year = int(season)
                season_start = datetime(year, 7, 1)
                season_end = datetime(year + 1, 6, 30)
            except ValueError:
                season_start = datetime(now.year - 1 if now.month < 7 else now.year, 7, 1)
                season_end = season_start.replace(year=season_start.year + 1, month=6, day=30)
        else:
            if now.month >= 7:
                season_start = datetime(now.year, 7, 1)
                season_end = datetime(now.year + 1, 6, 30)
            else:
                season_start = datetime(now.year - 1, 7, 1)
                season_end = datetime(now.year, 6, 30)

        teams = db.query(Team).join(
            Match, (Match.home_team_id == Team.id) | (Match.away_team_id == Team.id)
        ).filter(
            Match.competition_id == comp.id,
            Match.status == "finished",
            Match.kickoff >= season_start,
            Match.kickoff <= season_end,
        ).distinct().all()

        if not teams:
            return f"Nessun dato per {comp.name} nel periodo {season_start.year}/{season_end.year}."

        standings = []
        for team in teams:
            matches = db.query(Match).filter(
                Match.competition_id == comp.id,
                Match.status == "finished",
                Match.kickoff >= season_start,
                Match.kickoff <= season_end,
                (Match.home_team_id == team.id) | (Match.away_team_id == team.id),
            ).all()
            pts, gf, ga, w, d, l = 0, 0, 0, 0, 0, 0
            for m in matches:
                gf_m = (m.home_score or 0) if m.home_team_id == team.id else (m.away_score or 0)
                ga_m = (m.away_score or 0) if m.home_team_id == team.id else (m.home_score or 0)
                gf += gf_m; ga += ga_m
                if gf_m > ga_m: pts += 3; w += 1
                elif gf_m == ga_m: pts += 1; d += 1
                else: l += 1
            if matches:
                standings.append({"team": team.short_name or team.name, "pts": pts,
                                   "gd": gf-ga, "gf": gf, "ga": ga,
                                   "played": len(matches), "w": w, "d": d, "l": l})

        standings.sort(key=lambda x: (-x["pts"], -x["gd"], -x["gf"]))
        label = f"{season_start.year}/{str(season_end.year)[2:]}"
        return f"Classifica {comp.name} {label}:\n" + "\n".join(
            f"{i:2}. {s['team']:<20} {s['pts']}pt  {s['played']}G  {s['w']}V {s['d']}P {s['l']}S  GD{s['gd']:+d}"
            for i, s in enumerate(standings[:20], 1)
        )


@tool
def get_team_stats(team_name: str) -> str:
    """
    Statistiche complete di una squadra: forma recente, gol fatti/subiti, rendimento casa/trasferta.
    Utile per analisi e confronti tra squadre.
    """
    from sqlalchemy import func, or_
    with get_db_session() as db:
        team = db.query(Team).filter(func.lower(Team.name).contains(team_name.lower())).first()
        if not team:
            return f"Squadra '{team_name}' non trovata."

        now = datetime.utcnow()
        season_start = datetime(now.year - 1 if now.month < 7 else now.year, 7, 1)

        matches = db.query(Match).filter(
            or_(Match.home_team_id == team.id, Match.away_team_id == team.id),
            Match.status == "finished",
            Match.kickoff >= season_start,
        ).order_by(Match.kickoff.desc()).all()

        if not matches:
            return f"Nessuna partita trovata per {team.name} questa stagione."

        total = len(matches)
        pts, gf, ga, w, d, l = 0, 0, 0, 0, 0, 0
        home_w, home_d, home_l = 0, 0, 0
        away_w, away_d, away_l = 0, 0, 0
        last5 = []

        for m in matches:
            is_home = m.home_team_id == team.id
            g_for = (m.home_score or 0) if is_home else (m.away_score or 0)
            g_ag = (m.away_score or 0) if is_home else (m.home_score or 0)
            gf += g_for; ga += g_ag
            if g_for > g_ag:
                pts += 3; w += 1
                if is_home: home_w += 1
                else: away_w += 1
                if len(last5) < 5: last5.append("V")
            elif g_for == g_ag:
                pts += 1; d += 1
                if is_home: home_d += 1
                else: away_d += 1
                if len(last5) < 5: last5.append("P")
            else:
                l += 1
                if is_home: home_l += 1
                else: away_l += 1
                if len(last5) < 5: last5.append("S")

        avg_gf = gf / total if total > 0 else 0
        avg_ga = ga / total if total > 0 else 0

        return (
            f"Statistiche {team.name} (stagione corrente):\n"
            f"Partite: {total} | Punti: {pts} | {w}V {d}P {l}S\n"
            f"Gol: {gf} fatti, {ga} subiti (media: {avg_gf:.1f}/{avg_ga:.1f} a partita)\n"
            f"Casa: {home_w}V {home_d}P {home_l}S | Trasferta: {away_w}V {away_d}P {away_l}S\n"
            f"Ultime 5: {' '.join(last5)}"
        )


ALL_TOOLS = [get_matches, get_lineup, get_prediction, get_transfers,
             get_injuries, get_standings, get_team_stats]


class FootballChatbot:
    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.llm = build_llm()
        self.chat_history = []
        prompt = ChatPromptTemplate.from_messages([
            ("system", get_system_prompt()),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        agent = create_tool_calling_agent(self.llm, ALL_TOOLS, prompt)
        self.executor = AgentExecutor(
            agent=agent, tools=ALL_TOOLS,
            verbose=False, max_iterations=5, handle_parsing_errors=True,
        )

    @staticmethod
    def _clean(text: str) -> str:
        text = re.sub(r"function=\w+[^\n]*", "", text)
        text = re.sub(r"<[/]?function[^>]*>", "", text)
        text = re.sub(r"\n\n\n+", "\n\n", text)
        return text.strip()

    async def chat_async(self, message: str) -> str:
        import asyncio
        last_error = None
        for attempt in range(3):
            try:
                if attempt > 0:
                    await asyncio.sleep(2 * attempt)
                result = await self.executor.ainvoke({
                    "input": message,
                    "chat_history": self.chat_history[-16:],
                })
                output = self._clean(result.get("output", "Non ho capito la domanda."))
                self.chat_history.append(HumanMessage(content=message))
                self.chat_history.append(AIMessage(content=output))
                return output
            except Exception as e:
                last_error = e
                logger.warning(f"Chat tentativo {attempt+1} fallito [{self.session_id}]: {e}")
                if "rate_limit" in str(e).lower():
                    await asyncio.sleep(10)
        logger.error(f"Chat fallita [{self.session_id}]: {last_error}")
        return "Mi dispiace, servizio temporaneamente non disponibile. Riprova tra qualche secondo."

    async def stream_response(self, message: str):
        """
        Versione streaming per l'endpoint SSE: genera prima la risposta
        completa (con retry/cleaning di chat_async) e la restituisce
        a piccoli blocchi, per un effetto "a comparsa" lato client.
        """
        import asyncio
        response = await self.chat_async(message)
        for i in range(0, len(response), 4):
            yield response[i:i + 4]
            await asyncio.sleep(0.02)

    def clear_history(self):
        self.chat_history = []


class ChatSessionManager:
    def __init__(self, max_sessions: int = 50):
        self._sessions: dict[str, FootballChatbot] = {}
        self._max = max_sessions

    def get_or_create(self, session_id: str) -> FootballChatbot:
        if session_id not in self._sessions:
            if len(self._sessions) >= self._max:
                del self._sessions[next(iter(self._sessions))]
            self._sessions[session_id] = FootballChatbot(session_id)
        return self._sessions[session_id]

    def delete(self, session_id: str):
        self._sessions.pop(session_id, None)

    @property
    def active_sessions(self) -> int:
        return len(self._sessions)


chat_manager = ChatSessionManager()
