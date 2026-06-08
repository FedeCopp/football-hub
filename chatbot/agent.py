"""
chatbot/agent.py
Chatbot AI — LangChain 0.3.x + Groq (gratis)
"""
import logging
import re
from datetime import datetime
from typing import Optional

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

from config import settings
from db.database import get_db_session
from db.models import Match, Team, Lineup, LineupPlayer, Transfer, Prediction, Injury, Competition

logger = logging.getLogger(__name__)


# ── LLM Factory ──────────────────────────────────────────────

def build_llm():
    provider = settings.llm_provider
    logger.info(f"LLM provider: {provider}")
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=settings.GROQ_MODEL, temperature=0.2, api_key=settings.GROQ_API_KEY)
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=settings.OPENAI_API_KEY)
    from langchain_community.chat_models import ChatOllama
    return ChatOllama(model=settings.OLLAMA_MODEL, base_url=settings.OLLAMA_BASE_URL, temperature=0.2)


def get_system_prompt() -> str:
    now = datetime.utcnow().strftime("%d/%m/%Y %H:%M UTC")
    return (
        "Sei FootballHub AI, assistente esperto di calcio italiano.\n\n"
        "REGOLE FONDAMENTALI:\n"
        "- Usa SEMPRE i tools per rispondere. Non usare mai la tua memoria per dati su partite, risultati, classifiche o mercato.\n"
        "- NON mostrare mai nel testo tag come function= o XML. Esegui i tool silenziosamente e mostra solo la risposta finale.\n"
        "- Rispondi sempre in italiano pulito e conciso. Usa emoji.\n"
        "- Il database contiene Serie A, Champions League e Premier League stagioni 2023/24, 2024/25 e 2025/26.\n"
        f"- Data e ora attuale: {now}\n"
    )


# ── Tools ─────────────────────────────────────────────────────

@tool
def get_today_matches(league: str = "") -> str:
    """Partite di oggi. Se non ce ne sono mostra le ultime giocate o le prossime."""
    from datetime import timedelta
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    tomorrow = today + timedelta(days=1)

    def fmt(m):
        home = m.home_team.short_name or m.home_team.name
        away = m.away_team.short_name or m.away_team.name
        d = m.kickoff.strftime("%d/%m %H:%M") if m.kickoff else "?"
        if m.status == "finished":
            return f"FINE {home} {m.home_score}-{m.away_score} {away} ({d})"
        elif m.status in ("live", "in_play"):
            return f"LIVE {home} {m.home_score or 0}-{m.away_score or 0} {away}"
        return f"ORE {home} vs {away} ({d})"

    with get_db_session() as db:
        matches = db.query(Match).filter(
            Match.kickoff >= today, Match.kickoff < tomorrow
        ).order_by(Match.kickoff).all()
        if matches:
            return "Partite di oggi:\n" + "\n".join(fmt(m) for m in matches)

        recent = db.query(Match).filter(
            Match.kickoff >= today - timedelta(days=7),
            Match.kickoff < today,
            Match.status == "finished"
        ).order_by(Match.kickoff.desc()).limit(10).all()
        if recent:
            return "Nessuna partita oggi. Ultime partite:\n" + "\n".join(fmt(m) for m in recent)

        nxt = db.query(Match).filter(Match.kickoff > tomorrow).order_by(Match.kickoff).limit(10).all()
        if nxt:
            return "Nessuna partita oggi. Prossime in programma:\n" + "\n".join(fmt(m) for m in nxt)

        return "Nessuna partita trovata nel database."


@tool
def get_lineup(team_name: str) -> str:
    """Formazione probabile o ufficiale di una squadra."""
    from sqlalchemy import func, or_
    from datetime import timedelta
    with get_db_session() as db:
        team = db.query(Team).filter(func.lower(Team.name).contains(team_name.lower())).first()
        if not team:
            return f"Squadra '{team_name}' non trovata."
        now = datetime.utcnow()
        match = db.query(Match).filter(
            or_(Match.home_team_id == team.id, Match.away_team_id == team.id),
            Match.kickoff >= now - timedelta(hours=2),
            Match.kickoff <= now + timedelta(days=7),
        ).order_by(Match.kickoff).first()
        if not match:
            return f"Nessuna partita per {team.name} nei prossimi 7 giorni."
        lineup = db.query(Lineup).filter_by(match_id=match.id, team_id=team.id).order_by(Lineup.is_official.desc()).first()
        if not lineup:
            return f"Formazione non ancora disponibile per {team.name}."
        starters = db.query(LineupPlayer).filter_by(lineup_id=lineup.id, role="starter").all()
        tipo = "UFFICIALE" if lineup.is_official else "Probabile"
        lines = [f"{tipo} - {team.name} ({lineup.formation or '?'})", ""]
        for i, lp in enumerate(starters, 1):
            name = lp.player.name if lp.player else "N/D"
            unc = " (dubbio)" if lp.is_uncertain else ""
            lines.append(f"{i:2}. {name}{unc}")
        return "\n".join(lines)


@tool
def get_prediction(home_team: str, away_team: str) -> str:
    """Previsione ML per una partita. Esempio: get_prediction('Milan', 'Inter')"""
    from sqlalchemy import func
    from datetime import timedelta
    with get_db_session() as db:
        home = db.query(Team).filter(func.lower(Team.name).contains(home_team.lower())).first()
        away = db.query(Team).filter(func.lower(Team.name).contains(away_team.lower())).first()
        if not home or not away:
            return "Squadra non trovata."
        now = datetime.utcnow()
        match = db.query(Match).filter(
            Match.home_team_id == home.id, Match.away_team_id == away.id,
            Match.kickoff >= now - timedelta(days=1),
            Match.kickoff <= now + timedelta(days=60),
        ).order_by(Match.kickoff).first()
        if not match:
            return f"Nessuna partita imminente tra {home.name} e {away.name}."
        pred = db.query(Prediction).filter_by(match_id=match.id).first()
        if not pred:
            return "Previsione non disponibile. Il modello ML deve essere addestrato prima."
        lines = [
            f"Previsione: {home.name} vs {away.name}",
            f"1 {home.short_name or home.name}: {pred.prob_home:.0f}%",
            f"X Pareggio: {pred.prob_draw:.0f}%",
            f"2 {away.short_name or away.name}: {pred.prob_away:.0f}%",
        ]
        if pred.btts_prob:
            lines.append(f"BTTS: {pred.btts_prob:.0f}%  Over 2.5: {pred.over25_prob:.0f}%")
        if pred.scorer_probs:
            top = sorted(pred.scorer_probs, key=lambda x: -x.get("prob", 0))[:3]
            lines.append("Marcatori: " + ", ".join(f"{s['name']} {s['prob']:.0f}%" for s in top))
        return "\n".join(lines)


@tool
def get_transfers(query: str) -> str:
    """Rumors calciomercato per giocatore o squadra."""
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
            return f"Nessun rumor trovato per '{query}'."
        lines = []
        for t in transfers:
            hwg = " HERE WE GO!" if t.here_we_go else ""
            lines.append(f"{t.player_name}{hwg}: {t.from_team or '?'} -> {t.to_team or '?'} ({t.probability:.0f}%)")
            if t.detail:
                lines.append(f"  {t.detail[:150]}")
        return "\n".join(lines)


@tool
def get_injuries(team_name: str) -> str:
    """Infortuni e squalifiche attive di una squadra."""
    from sqlalchemy import func
    with get_db_session() as db:
        team = db.query(Team).filter(func.lower(Team.name).contains(team_name.lower())).first()
        if not team:
            return f"Squadra '{team_name}' non trovata."
        from db.models import Player
        injuries = db.query(Injury).join(Player).filter(
            Player.team_id == team.id, Injury.is_active == True
        ).all()
        if not injuries:
            return f"{team.name}: nessun infortunio attivo registrato."
        return "\n".join(f"- {i.player.name}: {i.type} ({i.reason})" for i in injuries if i.player)


@tool
def get_standings(competition: str) -> str:
    """Classifica di una competizione. Esempio: get_standings('Serie A')"""
    from sqlalchemy import func
    with get_db_session() as db:
        comp = db.query(Competition).filter(func.lower(Competition.name).contains(competition.lower())).first()
        if not comp:
            return f"Competizione '{competition}' non trovata nel database."
        teams = db.query(Team).join(
            Match, (Match.home_team_id == Team.id) | (Match.away_team_id == Team.id)
        ).filter(Match.competition_id == comp.id, Match.status == "finished").distinct().all()
        standings = []
        for team in teams:
            matches = db.query(Match).filter(
                Match.competition_id == comp.id, Match.status == "finished",
                (Match.home_team_id == team.id) | (Match.away_team_id == team.id),
            ).all()
            pts, gf, ga = 0, 0, 0
            for m in matches:
                gf_m = (m.home_score or 0) if m.home_team_id == team.id else (m.away_score or 0)
                ga_m = (m.away_score or 0) if m.home_team_id == team.id else (m.home_score or 0)
                gf += gf_m
                ga += ga_m
                if gf_m > ga_m:
                    pts += 3
                elif gf_m == ga_m:
                    pts += 1
            if matches:
                standings.append({"team": team.short_name or team.name, "pts": pts, "gd": gf - ga, "played": len(matches)})
        if not standings:
            return f"Nessun dato di classifica per {comp.name}."
        standings.sort(key=lambda x: (-x["pts"], -x["gd"]))
        return f"Classifica {comp.name}:\n" + "\n".join(
            f"{i:2}. {s['team']:<20} {s['pts']}pt  {s['played']}G  ({s['gd']:+d})"
            for i, s in enumerate(standings[:20], 1)
        )


# ── Agent ─────────────────────────────────────────────────────

ALL_TOOLS = [get_today_matches, get_lineup, get_prediction, get_transfers, get_injuries, get_standings]


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
            verbose=False, max_iterations=4, handle_parsing_errors=True,
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
        for attempt in range(3):  # max 3 tentativi
            try:
                if attempt > 0:
                    await asyncio.sleep(2 * attempt)  # attendi prima di riprovare
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
                    await asyncio.sleep(10)  # attendi di più per rate limit
        logger.error(f"Chat fallita dopo 3 tentativi [{self.session_id}]: {last_error}")
        return "Mi dispiace, il servizio AI è momentaneamente sovraccarico. Riprova tra qualche secondo."

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
