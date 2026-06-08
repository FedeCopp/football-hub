"""
db/models.py
Definizione di tutte le tabelle del database con SQLAlchemy ORM.
"""
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Boolean,
    DateTime, Text, ForeignKey, Index, JSON
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


# ─────────────────────────────────────────────────────────────
# COMPETIZIONI & SQUADRE
# ─────────────────────────────────────────────────────────────

class Competition(Base):
    __tablename__ = "competitions"

    id          = Column(Integer, primary_key=True)
    ext_id      = Column(String(50), unique=True)   # id da API esterna
    name        = Column(String(100), nullable=False)
    country     = Column(String(50))
    season      = Column(String(10))                 # es. "2023/24"
    created_at  = Column(DateTime, default=datetime.utcnow)

    matches = relationship("Match", back_populates="competition")


class Team(Base):
    __tablename__ = "teams"

    id          = Column(Integer, primary_key=True)
    ext_id      = Column(String(50), unique=True)
    name        = Column(String(100), nullable=False)
    short_name  = Column(String(20))
    country     = Column(String(50))
    logo_url    = Column(String(255))
    created_at  = Column(DateTime, default=datetime.utcnow)
    updated_at  = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    home_matches = relationship("Match", foreign_keys="Match.home_team_id", back_populates="home_team")
    away_matches = relationship("Match", foreign_keys="Match.away_team_id", back_populates="away_team")
    players      = relationship("Player", back_populates="team")


class Player(Base):
    __tablename__ = "players"

    id              = Column(Integer, primary_key=True)
    ext_id          = Column(String(50), unique=True)
    team_id         = Column(Integer, ForeignKey("teams.id"))
    name            = Column(String(100), nullable=False)
    position        = Column(String(20))            # GK, CB, LB, RB, DM, CM, AM, LW, RW, CF
    nationality     = Column(String(50))
    birth_date      = Column(DateTime)
    shirt_number    = Column(Integer)
    market_value    = Column(Float)                 # in milioni €
    created_at      = Column(DateTime, default=datetime.utcnow)
    updated_at      = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    team        = relationship("Team", back_populates="players")
    stats       = relationship("PlayerStats", back_populates="player")
    lineups     = relationship("LineupPlayer", back_populates="player")
    injuries    = relationship("Injury", back_populates="player")


# ─────────────────────────────────────────────────────────────
# PARTITE
# ─────────────────────────────────────────────────────────────

class Match(Base):
    __tablename__ = "matches"

    id              = Column(Integer, primary_key=True)
    ext_id          = Column(String(50), unique=True)
    competition_id  = Column(Integer, ForeignKey("competitions.id"))
    home_team_id    = Column(Integer, ForeignKey("teams.id"))
    away_team_id    = Column(Integer, ForeignKey("teams.id"))

    matchday        = Column(Integer)
    kickoff         = Column(DateTime)
    status          = Column(String(20), default="scheduled")
    # scheduled | live | finished | postponed

    # Risultato
    home_score      = Column(Integer)
    away_score      = Column(Integer)
    home_ht         = Column(Integer)               # half-time
    away_ht         = Column(Integer)

    # Statistiche partita (popolate a fine gara)
    home_xg         = Column(Float)
    away_xg         = Column(Float)
    home_possession = Column(Float)
    away_possession = Column(Float)
    home_shots      = Column(Integer)
    away_shots      = Column(Integer)
    home_shots_ot   = Column(Integer)               # shots on target
    away_shots_ot   = Column(Integer)
    home_corners    = Column(Integer)
    away_corners    = Column(Integer)
    home_fouls      = Column(Integer)
    away_fouls      = Column(Integer)
    home_yellow     = Column(Integer)
    away_yellow     = Column(Integer)
    home_red        = Column(Integer)
    away_red        = Column(Integer)

    created_at      = Column(DateTime, default=datetime.utcnow)
    updated_at      = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    competition = relationship("Competition", back_populates="matches")
    home_team   = relationship("Team", foreign_keys=[home_team_id], back_populates="home_matches")
    away_team   = relationship("Team", foreign_keys=[away_team_id], back_populates="away_matches")
    lineups     = relationship("Lineup", back_populates="match")
    events      = relationship("MatchEvent", back_populates="match")
    odds        = relationship("Odds", back_populates="match")
    prediction  = relationship("Prediction", back_populates="match", uselist=False)

    __table_args__ = (
        Index("ix_matches_kickoff", "kickoff"),
        Index("ix_matches_status", "status"),
    )


class MatchEvent(Base):
    """Gol, cartellini, sostituzioni."""
    __tablename__ = "match_events"

    id          = Column(Integer, primary_key=True)
    match_id    = Column(Integer, ForeignKey("matches.id"))
    team_id     = Column(Integer, ForeignKey("teams.id"))
    player_id   = Column(Integer, ForeignKey("players.id"), nullable=True)
    event_type  = Column(String(20))
    # goal | yellow | red | yellow_red | substitution | penalty_missed
    minute      = Column(Integer)
    extra_time  = Column(Integer, default=0)
    detail      = Column(String(100))               # es. "Normal Goal", "Own Goal"

    match = relationship("Match", back_populates="events")


# ─────────────────────────────────────────────────────────────
# FORMAZIONI
# ─────────────────────────────────────────────────────────────

class Lineup(Base):
    __tablename__ = "lineups"

    id              = Column(Integer, primary_key=True)
    match_id        = Column(Integer, ForeignKey("matches.id"))
    team_id         = Column(Integer, ForeignKey("teams.id"))
    formation       = Column(String(20))            # es. "4-3-3"
    is_official     = Column(Boolean, default=False)
    source          = Column(String(50))            # "api_football"|"gazzetta"|"fantacalcio"
    fetched_at      = Column(DateTime, default=datetime.utcnow)

    match   = relationship("Match", back_populates="lineups")
    players = relationship("LineupPlayer", back_populates="lineup")


class LineupPlayer(Base):
    __tablename__ = "lineup_players"

    id          = Column(Integer, primary_key=True)
    lineup_id   = Column(Integer, ForeignKey("lineups.id"))
    player_id   = Column(Integer, ForeignKey("players.id"))
    role        = Column(String(10))                # starter | substitute
    position    = Column(String(20))
    shirt_num   = Column(Integer)
    is_uncertain= Column(Boolean, default=False)    # dubbio per infortunio ecc.

    lineup  = relationship("Lineup", back_populates="players")
    player  = relationship("Player", back_populates="lineups")


# ─────────────────────────────────────────────────────────────
# STATISTICHE GIOCATORI (per stagione)
# ─────────────────────────────────────────────────────────────

class PlayerStats(Base):
    __tablename__ = "player_stats"

    id              = Column(Integer, primary_key=True)
    player_id       = Column(Integer, ForeignKey("players.id"))
    competition_id  = Column(Integer, ForeignKey("competitions.id"))
    season          = Column(String(10))

    appearances     = Column(Integer, default=0)
    minutes         = Column(Integer, default=0)
    goals           = Column(Integer, default=0)
    assists         = Column(Integer, default=0)
    yellow_cards    = Column(Integer, default=0)
    red_cards       = Column(Integer, default=0)

    # Stats avanzate (da FBref/soccerdata)
    xg              = Column(Float)
    xa              = Column(Float)
    shots           = Column(Integer)
    shots_on_target = Column(Integer)
    key_passes      = Column(Integer)
    dribbles_succ   = Column(Integer)
    tackles         = Column(Integer)
    interceptions   = Column(Integer)

    # Rendimento casa/trasferta
    goals_home      = Column(Integer, default=0)
    goals_away      = Column(Integer, default=0)
    yellow_home     = Column(Integer, default=0)
    yellow_away     = Column(Integer, default=0)

    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    player = relationship("Player", back_populates="stats")

    __table_args__ = (
        Index("ix_playerstats_player_season", "player_id", "season"),
    )


# ─────────────────────────────────────────────────────────────
# INFORTUNI / SQUALIFICHE
# ─────────────────────────────────────────────────────────────

class Injury(Base):
    __tablename__ = "injuries"

    id          = Column(Integer, primary_key=True)
    player_id   = Column(Integer, ForeignKey("players.id"))
    type        = Column(String(50))                # "Muscle" | "Knee" | "Suspended" ecc.
    reason      = Column(String(200))
    start_date  = Column(DateTime)
    end_date    = Column(DateTime, nullable=True)   # None = ancora in corso
    is_active   = Column(Boolean, default=True)
    source      = Column(String(50))
    fetched_at  = Column(DateTime, default=datetime.utcnow)

    player = relationship("Player", back_populates="injuries")


# ─────────────────────────────────────────────────────────────
# QUOTE
# ─────────────────────────────────────────────────────────────

class Odds(Base):
    __tablename__ = "odds"

    id              = Column(Integer, primary_key=True)
    match_id        = Column(Integer, ForeignKey("matches.id"))
    bookmaker       = Column(String(50))
    market          = Column(String(30))
    # 1x2 | btts | over_under | correct_score | first_scorer

    # 1X2
    home_win        = Column(Float)
    draw            = Column(Float)
    away_win        = Column(Float)

    # BTTS
    btts_yes        = Column(Float)
    btts_no         = Column(Float)

    # Over/Under 2.5
    over_25         = Column(Float)
    under_25        = Column(Float)

    # Probabilità implicite normalizzate (rimozione margine bookmaker)
    impl_home       = Column(Float)
    impl_draw       = Column(Float)
    impl_away       = Column(Float)

    fetched_at      = Column(DateTime, default=datetime.utcnow)

    match = relationship("Match", back_populates="odds")

    __table_args__ = (
        Index("ix_odds_match_bookmaker", "match_id", "bookmaker"),
    )


# ─────────────────────────────────────────────────────────────
# TRASFERIMENTI
# ─────────────────────────────────────────────────────────────

class Transfer(Base):
    __tablename__ = "transfers"

    id              = Column(Integer, primary_key=True)
    player_id       = Column(Integer, ForeignKey("players.id"), nullable=True)
    player_name     = Column(String(100))           # fallback se player non in DB
    from_team       = Column(String(100))
    to_team         = Column(String(100), nullable=True)
    fee_estimate    = Column(String(50))            # es. "€70M" o "Free"
    probability     = Column(Float)                 # 0-100
    status          = Column(String(20), default="rumor")
    # rumor | advanced | confirmed | completed | denied

    detail          = Column(Text)                  # spiegazione trattativa
    sources         = Column(JSON)                  # lista fonti con peso
    here_we_go      = Column(Boolean, default=False)

    first_seen      = Column(DateTime, default=datetime.utcnow)
    updated_at      = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("ix_transfers_player", "player_name"),
        Index("ix_transfers_status", "status"),
    )


# ─────────────────────────────────────────────────────────────
# PREVISIONI ML
# ─────────────────────────────────────────────────────────────

class Prediction(Base):
    __tablename__ = "predictions"

    id              = Column(Integer, primary_key=True)
    match_id        = Column(Integer, ForeignKey("matches.id"), unique=True)

    # Esito 1X2
    prob_home       = Column(Float)
    prob_draw       = Column(Float)
    prob_away       = Column(Float)

    # Risultati esatti (JSON: {"1-0": 11.2, "1-1": 14.1, ...})
    score_probs     = Column(JSON)

    # Marcatori (JSON: [{"player_id": 1, "name": "Giroud", "prob": 28.4}, ...])
    scorer_probs    = Column(JSON)

    # Ammoniti (JSON: [{"player_id": 5, "name": "Calhanoglu", "prob": 44.1}, ...])
    booked_probs    = Column(JSON)

    # Altri mercati
    btts_prob       = Column(Float)
    over25_prob     = Column(Float)

    model_version   = Column(String(20))
    confidence      = Column(Float)                 # 0-1 confidence score
    created_at      = Column(DateTime, default=datetime.utcnow)

    match = relationship("Match", back_populates="prediction")


# ─────────────────────────────────────────────────────────────
# NEWS / TRANSFER RUMORS (raw, pre-NLP)
# ─────────────────────────────────────────────────────────────

class NewsItem(Base):
    __tablename__ = "news_items"

    id          = Column(Integer, primary_key=True)
    source      = Column(String(50))                # "sky_sport"|"romano_twitter" ecc.
    title       = Column(String(500))
    body        = Column(Text)
    url         = Column(String(500))
    published   = Column(DateTime)
    processed   = Column(Boolean, default=False)    # già passato per NLP?
    fetched_at  = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_news_source_published", "source", "published"),
        Index("ix_news_processed", "processed"),
    )
