"""
chatbot/agent.py
Chatbot AI — compatibile con LangChain 0.3.x
"""
import logging
from datetime import datetime
from typing import Optional

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain.memory import ConversationBufferWindowMemory

from config import settings
from db.database import get_db_session
from db.models import (
    Match, Team, Player, Lineup, LineupPlayer,
    Transfer, Prediction, Injury, PlayerStats,
    Competition, Odds
)

logger = logging.getLogger(__name__)


# ─── LLM Factory ─────────────────────────────────────────────

def build_llm():
    provider = settings.llm_provider
    logger.info(f"LLM provider: {provider}")

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=settings.GROQ_MODEL,
            temperature=0.2,
            api_key=settings.GROQ_API_KEY,
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            api_key=settings.OPENAI_API_KEY,
        )
    from langchain_community.chat_models import ChatOllama
    return ChatOllama(
        model=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0.2,
    )


# ─── Tools ───────────────────────────────────────────────────

@tool
def get_today_matches(league: str = "") -> str:
    """Partite di oggi con orari e punteggi. Parametro opzionale: league."""
    from datetime import timedelta
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    tomorrow = today + timedelta(days=1)
    with get_db_session() as db:
        query = db.query(Match).filter(Match.kickoff >= today, Match.kickoff < tomorrow)
        matches = query.order_by(Match.kickoff).all()
        if not matches:
            return "Nessuna partita oggi."
        result = []
        for m in matches:
            home = m.home_team.short_name or m.home_team.name
            away = m.away_team.short_name or m.away_team.name
            t = m.kickoff.strftime("%H:%M") if m.kickoff else "?"
            if m.status == "finished":
                result.append(f"✅ {home} {m.home_score}-{m.away_score} {away} (FT)")
            elif m.status in ("live", "in_play"):
                result.append(f"🔴 {home} {m.home_score or 0}-{m.away_score or 0} {away} (LIVE)")
            else:
                result.append(f"⏰ {home} vs {away} ore {t}")
        return "\n".join(result)


@tool
def get_lineup(team_name: str) -> str:
    """Formazione probabile o ufficiale di una squadra. Es: get_lineup('Milan')"""
    from sqlalchemy import func, or_
    from datetime import timedelta
    with get_db_session() as db:
        team = db.query(Team).filter(
            func.lower(Team.name).contains(team_name.lower())
        ).first()
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
        lineup = db.query(Lineup).filter_by(
            match_id=match.id, team_id=team.id
        ).order_by(Lineup.is_official.desc()).first()
        if not lineup:
            return f"Formazione non ancora disponibile per {team.name}."
        starters = db.query(LineupPlayer).filter_by(lineup_id=lineup.id, role="starter").all()
        lines = [f"{'🔴 UFFICIALE' if lineup.is_official else '📋 Probabile'} — {team.name} ({lineup.formation or '?'})", ""]
        for i, lp in enumerate(starters, 1):
            name = lp.player.name if lp.player else "N/D"
            unc = " ⚠️" if lp.is_uncertain else ""
            lines.append(f"{i:2}. {name}{unc}")
        return "\n".join(lines)


@tool
def get_prediction(home_team: str, away_team: str) -> str:
    """Previsione ML per una partita. Es: get_prediction('Milan', 'Inter')"""
    from sqlalchemy import func
    from datetime import timedelta
    with get_db_session() as db:
        home = db.query(Team).filter(func.lower(Team.name).contains(home_team.lower())).first()
        away = db.query(Team).filter(func.lower(Team.name).contains(away_team.lower())).first()
        if not home or not away:
            return f"Squadra non trovata."
        now = datetime.utcnow()
        match = db.query(Match).filter(
            Match.home_team_id == home.id, Match.away_team_id == away.id,
            Match.kickoff >= now - timedelta(days=1),
            Match.kickoff <= now + timedelta(days=30),
        ).order_by(Match.kickoff).first()
        if not match:
            return f"Nessuna partita trovata tra {home.name} e {away.name}."
        pred = db.query(Prediction).filter_by(match_id=match.id).first()
        if not pred:
            return f"Previsione non ancora disponibile. Assicurati che il training ML sia stato eseguito."
        lines = [
            f"🎯 {home.name} vs {away.name}",
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
    """Rumors calciomercato. Es: get_transfers('Juventus') o get_transfers('Osimhen')"""
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
            hwg = " 🟢 HERE WE GO!" if t.here_we_go else ""
            lines.append(f"{'🔥' if t.probability > 70 else '💬'} {t.player_name}{hwg}: {t.from_team or '?'} → {t.to_team or '?'} ({t.probability:.0f}%)")
            if t.detail:
                lines.append(f"   {t.detail[:150]}")
        return "\n".join(lines)


@tool
def get_injuries(team_name: str) -> str:
    """Infortuni attivi di una squadra. Es: get_injuries('Napoli')"""
    from sqlalchemy import func
    with get_db_session() as db:
        team = db.query(Team).filter(func.lower(Team.name).contains(team_name.lower())).first()
        if not team:
            return f"Squadra '{team_name}' non trovata."
        injuries = db.query(Injury).join(Player).filter(
            Player.team_id == team.id, Injury.is_active == True
        ).all()
        if not injuries:
            return f"✅ {team.name}: nessun infortunio attivo."
        return "\n".join(f"• {i.player.name}: {i.type} ({i.reason})" for i in injuries if i.player)


@tool
def get_standings(competition: str) -> str:
    """Classifica di una competizione. Es: get_standings('Serie A')"""
    from sqlalchemy import func
    with get_db_session() as db:
        comp = db.query(Competition).filter(
            func.lower(Competition.name).contains(competition.lower())
        ).first()
        if not comp:
            return f"Competizione '{competition}' non trovata."
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
                gf += gf_m; ga += ga_m
                if gf_m > ga_m: pts += 3
                elif gf_m == ga_m: pts += 1
            if matches:
                standings.append({"team": team.short_name or team.name, "pts": pts, "gd": gf-ga, "played": len(matches)})
        if not standings:
            return f"Nessun dato per {comp.name}."
        standings.sort(key=lambda x: (-x["pts"], -x["gd"]))
        return f"🏆 {comp.name}\n" + "\n".join(
            f"{i:2}. {s['team']:<18} {s['pts']}pt ({s['played']}G)"
            for i, s in enumerate(standings[:20], 1)
        )


# ─── Agent ───────────────────────────────────────────────────

def get_system_prompt():
    from datetime import datetime
    now = datetime.utcnow().strftime("%d/%m/%Y %H:%M UTC")
    return f"""Sei FootballHub AI, assistente esperto di calcio italiano.
Usa SEMPRE i tools per rispondere — non inventare dati su partite, formazioni o mercato.
Rispondi in italiano. Sii conciso e usa emoji per leggibilità.
Data attuale: {now}"""

SYSTEM_PROMPT = get_system_prompt()

ALL_TOOLS = [get_today_matches, get_lineup, get_prediction, get_transfers, get_injuries, get_standings]


class FootballChatbot:
    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.llm = build_llm()
        self.memory = ConversationBufferWindowMemory(
            k=8, memory_key="chat_history", return_messages=True
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", get_system_prompt()),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        agent = create_tool_calling_agent(self.llm, ALL_TOOLS, prompt)
        self.executor = AgentExecutor(
            agent=agent, tools=ALL_TOOLS, memory=self.memory,
            verbose=False, max_iterations=4, handle_parsing_errors=True,
        )

    async def chat_async(self, message: str) -> str:
        try:
            result = await self.executor.ainvoke({
                "input": message,
            })
            return result.get("output", "Non ho capito la domanda.")
        except Exception as e:
            logger.error(f"Chat error [{self.session_id}]: {e}")
            return f"Errore temporaneo: {str(e)[:150]}"

    def clear_history(self):
        self.memory.clear()


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
