"""
ml/predictor.py
Modello predittivo per le partite di calcio.

Architettura ensemble:
  - XGBoost      → probabilità esito 1/X/2
  - Poisson      → distribuzione gol (risultati esatti)
  - Random Forest → probabilità marcatori e ammoniti

Feature usate:
  - Forma ultima 10 partite (casa/trasferta separati)
  - Head-to-head ultimi 5 anni
  - xG stagionali (expected goals)
  - Statistiche individuali giocatori in formazione
  - Quote bookmaker (probabilità implicite normalizzate)
  - Infortuni / assenze
  - Giorni di riposo dall'ultima partita
"""
import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from db.database import get_db_session
from db.models import Match, Team, Player, Lineup, LineupPlayer, PlayerStats, Odds, Prediction, Injury

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


class FootballPredictor:
    """
    Predittore completo per partite di calcio.
    Prima esegui `train()` con i dati storici, poi usa `predict_match()`.
    """

    def __init__(self):
        self.outcome_model  = None   # XGBoost → 1/X/2
        self.goals_model    = None   # Poisson params
        self.scorer_model   = None   # RandomForest
        self.booked_model   = None   # RandomForest
        self._load_models()

    # ─────────────────────────────────────────────────────────
    # FEATURE ENGINEERING
    # ─────────────────────────────────────────────────────────

    def build_match_features(self, match_id: int) -> Optional[pd.Series]:
        """
        Costruisce il vettore di feature per un match.
        Questo è il cuore del sistema — più feature = più accuratezza.
        """
        with get_db_session() as db:
            match = db.query(Match).filter_by(id=match_id).first()
            if not match or not match.kickoff:
                return None

            home_id = match.home_team_id
            away_id = match.away_team_id

            f = {}

            # ── Forma recente ────────────────────────────────
            home_form_h = self._get_form(db, home_id, match.kickoff, venue="home", n=10)
            home_form_a = self._get_form(db, home_id, match.kickoff, venue="away", n=10)
            away_form_h = self._get_form(db, away_id, match.kickoff, venue="home", n=10)
            away_form_a = self._get_form(db, away_id, match.kickoff, venue="away", n=10)

            f["home_pts_home"]    = home_form_h["points"]
            f["home_gf_home"]     = home_form_h["goals_for"]
            f["home_ga_home"]     = home_form_h["goals_against"]
            f["home_xg_home"]     = home_form_h["xg"]
            f["home_pts_away"]    = home_form_a["points"]

            f["away_pts_away"]    = away_form_a["points"]
            f["away_gf_away"]     = away_form_a["goals_for"]
            f["away_ga_away"]     = away_form_a["goals_against"]
            f["away_xg_away"]     = away_form_a["xg"]
            f["away_pts_home"]    = away_form_h["points"]

            # Forma combinata ultimi 5 (qualsiasi sede)
            home_form5 = self._get_form(db, home_id, match.kickoff, venue="all", n=5)
            away_form5 = self._get_form(db, away_id, match.kickoff, venue="all", n=5)
            f["home_form5_pts"]   = home_form5["points"]
            f["away_form5_pts"]   = away_form5["points"]
            f["form_diff"]        = home_form5["points"] - away_form5["points"]

            # ── Head-to-head ─────────────────────────────────
            h2h = self._get_h2h(db, home_id, away_id, match.kickoff, n=10)
            f["h2h_home_wins"]    = h2h["home_wins"]
            f["h2h_draws"]        = h2h["draws"]
            f["h2h_away_wins"]    = h2h["away_wins"]
            f["h2h_home_goals"]   = h2h["home_goals_avg"]
            f["h2h_away_goals"]   = h2h["away_goals_avg"]
            f["h2h_btts_rate"]    = h2h["btts_rate"]

            # ── Quote (probabilità implicite) ─────────────────
            odds = db.query(Odds).filter_by(
                match_id=match_id, market="1x2", bookmaker="average"
            ).first()
            if odds and odds.impl_home:
                f["odds_impl_home"]  = odds.impl_home
                f["odds_impl_draw"]  = odds.impl_draw
                f["odds_impl_away"]  = odds.impl_away
                f["odds_home_raw"]   = odds.home_win or 0
                f["odds_away_raw"]   = odds.away_win or 0
            else:
                f["odds_impl_home"]  = 33.3
                f["odds_impl_draw"]  = 33.3
                f["odds_impl_away"]  = 33.3
                f["odds_home_raw"]   = 0.0
                f["odds_away_raw"]   = 0.0

            # ── xG stagionali ─────────────────────────────────
            home_season = self._get_season_stats(db, home_id, match.kickoff)
            away_season = self._get_season_stats(db, away_id, match.kickoff)
            f["home_xg_season"]   = home_season.get("xg_per_match", 1.2)
            f["home_xga_season"]  = home_season.get("xga_per_match", 1.2)
            f["away_xg_season"]   = away_season.get("xg_per_match", 1.2)
            f["away_xga_season"]  = away_season.get("xga_per_match", 1.2)
            f["xg_diff"]          = f["home_xg_season"] - f["away_xg_season"]

            # ── Riposo / stanchezza ───────────────────────────
            f["home_days_rest"]   = self._days_since_last_match(db, home_id, match.kickoff)
            f["away_days_rest"]   = self._days_since_last_match(db, away_id, match.kickoff)
            f["rest_advantage"]   = f["home_days_rest"] - f["away_days_rest"]

            # ── Forza formazione (da lineup) ──────────────────
            home_lineup_str = self._lineup_strength(db, match_id, home_id)
            away_lineup_str = self._lineup_strength(db, match_id, away_id)
            f["home_lineup_str"]  = home_lineup_str
            f["away_lineup_str"]  = away_lineup_str
            f["lineup_diff"]      = home_lineup_str - away_lineup_str

            # ── Infortuni ─────────────────────────────────────
            home_injuries = self._count_key_injuries(db, home_id)
            away_injuries = self._count_key_injuries(db, away_id)
            f["home_injuries"]    = home_injuries
            f["away_injuries"]    = away_injuries

            return pd.Series(f)

    # ─────────────────────────────────────────────────────────
    # PREDIZIONE PARTITA
    # ─────────────────────────────────────────────────────────

    def predict_match(self, match_id: int) -> Optional[Prediction]:
        """
        Genera la previsione completa per una partita e la salva nel DB.
        """
        features = self.build_match_features(match_id)
        if features is None:
            logger.warning(f"Feature non disponibili per match {match_id}")
            return None

        X = features.values.reshape(1, -1)

        # ── Esito 1/X/2 ──────────────────────────────────────
        if self.outcome_model:
            probs = self.outcome_model.predict_proba(X)[0]
            prob_home = round(float(probs[1]) * 100, 1)  # classe 1 = home win
            prob_draw = round(float(probs[0]) * 100, 1)  # classe 0 = draw
            prob_away = round(float(probs[2]) * 100, 1)  # classe 2 = away win
        else:
            # Fallback: usa probabilità implicite quote
            prob_home = float(features.get("odds_impl_home", 33.3))
            prob_draw = float(features.get("odds_impl_draw", 33.3))
            prob_away = float(features.get("odds_impl_away", 33.3))

        # ── Risultati esatti (Poisson bivariato) ─────────────
        home_lambda, away_lambda = self._estimate_goal_rates(features)
        score_probs = self._poisson_score_probs(home_lambda, away_lambda)

        # ── Marcatori probabili ───────────────────────────────
        scorer_probs = self._predict_scorers(match_id, home_lambda, away_lambda)

        # ── Ammoniti probabili ────────────────────────────────
        booked_probs = self._predict_bookings(match_id)

        # ── Altri mercati ─────────────────────────────────────
        btts_prob  = round(self._calc_btts(home_lambda, away_lambda) * 100, 1)
        over25_prob = round(self._calc_over_n5(home_lambda, away_lambda, 2.5) * 100, 1)

        # ── Confidence score ──────────────────────────────────
        # Alta se abbiamo dati storici abbondanti e quote disponibili
        has_odds    = float(features.get("odds_impl_home", 0)) > 0
        has_h2h     = float(features.get("h2h_home_wins", 0)) + float(features.get("h2h_draws", 0)) > 0
        confidence  = 0.5 + (0.25 if has_odds else 0) + (0.15 if has_h2h else 0)

        # ── Salva nel DB ──────────────────────────────────────
        with get_db_session() as db:
            pred = db.query(Prediction).filter_by(match_id=match_id).first()
            if not pred:
                pred = Prediction(match_id=match_id)
                db.add(pred)

            pred.prob_home     = prob_home
            pred.prob_draw     = prob_draw
            pred.prob_away     = prob_away
            pred.score_probs   = score_probs
            pred.scorer_probs  = scorer_probs
            pred.booked_probs  = booked_probs
            pred.btts_prob     = btts_prob
            pred.over25_prob   = over25_prob
            pred.model_version = "v1.0-ensemble"
            pred.confidence    = round(confidence, 2)
            pred.created_at    = datetime.utcnow()

        logger.info(
            f"Previsione match {match_id}: "
            f"1={prob_home}% X={prob_draw}% 2={prob_away}% "
            f"[confidence={confidence:.2f}]"
        )
        return pred

    # ─────────────────────────────────────────────────────────
    # TRAINING
    # ─────────────────────────────────────────────────────────

    def train(self, min_matches: int = 200) -> dict:
        """
        Addestra i modelli su tutti i match storici nel DB.
        Da eseguire dopo l'import iniziale dei dati.

        min_matches: numero minimo di partite richieste per il training.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        import xgboost as xgb

        logger.info("Inizio training modelli ML...")

        # Raccoglie feature per tutte le partite finite
        X_rows, y_outcome, y_home_goals, y_away_goals = [], [], [], []

        with get_db_session() as db:
            finished = db.query(Match).filter(
                Match.status == "finished",
                Match.home_score.isnot(None),
            ).all()
            match_ids = [m.id for m in finished]

        logger.info(f"Partite disponibili per training: {len(match_ids)}")

        if len(match_ids) < min_matches:
            logger.warning(
                f"Solo {len(match_ids)} partite disponibili. "
                f"Minimo richiesto: {min_matches}. "
                f"Esegui prima l'import storico."
            )
            return {"error": "insufficient_data", "available": len(match_ids)}

        for mid in match_ids:
            try:
                feats = self.build_match_features(mid)
                if feats is None:
                    continue

                with get_db_session() as db:
                    match = db.query(Match).filter_by(id=mid).first()
                    hg    = match.home_score or 0
                    ag    = match.away_score or 0

                # Esito: 1 = home, 0 = draw, 2 = away
                if hg > ag:
                    outcome = 1
                elif hg == ag:
                    outcome = 0
                else:
                    outcome = 2

                X_rows.append(feats.values)
                y_outcome.append(outcome)
                y_home_goals.append(hg)
                y_away_goals.append(ag)

            except Exception as e:
                logger.debug(f"Skip match {mid}: {e}")

        if not X_rows:
            return {"error": "no_features_built"}

        X = np.array(X_rows)
        y = np.array(y_outcome)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # ── XGBoost per esito ─────────────────────────────────
        logger.info("Training XGBoost (esito)...")
        self.outcome_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
        )
        self.outcome_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        acc = accuracy_score(y_test, self.outcome_model.predict(X_test))
        logger.info(f"XGBoost accuracy: {acc:.3f}")

        # ── Stima Poisson (media gol per team) ────────────────
        # Semplice modello Dixon-Coles: stima attack/defense per ogni squadra
        self.goals_model = self._fit_poisson_model(
            X_rows, y_home_goals, y_away_goals
        )

        # ── RandomForest per ammoniti ─────────────────────────
        # Target: numero totale cartellini nella partita (0,1,2,3,4+)
        logger.info("Training RandomForest (ammoniti)...")
        with get_db_session() as db:
            yellow_data = []
            for mid in match_ids[:len(X_rows)]:
                m = db.query(Match).filter_by(id=mid).first()
                if m:
                    tot = (m.home_yellow or 0) + (m.away_yellow or 0)
                    yellow_data.append(min(tot, 6))   # cap a 6

        if len(yellow_data) == len(X_rows):
            self.booked_model = RandomForestClassifier(
                n_estimators=200, max_depth=6, random_state=42
            )
            self.booked_model.fit(X_train, np.array(yellow_data)[:len(y_train)])

        # Salva modelli
        self._save_models()

        result = {
            "outcome_accuracy": round(acc, 3),
            "training_samples": len(X_rows),
            "test_samples": len(X_test),
            "model_version": "v1.0-ensemble",
        }
        logger.info(f"Training completato: {result}")
        return result

    # ─────────────────────────────────────────────────────────
    # HELPER: STATISTICHE STORICHE
    # ─────────────────────────────────────────────────────────

    def _get_form(
        self, db, team_id: int, before: datetime,
        venue: str = "all", n: int = 10
    ) -> dict:
        """Statistiche forma per le ultime N partite."""
        query = db.query(Match).filter(
            Match.status == "finished",
            Match.kickoff < before,
            Match.home_score.isnot(None),
        )
        if venue == "home":
            query = query.filter(Match.home_team_id == team_id)
        elif venue == "away":
            query = query.filter(Match.away_team_id == team_id)
        else:
            query = query.filter(
                (Match.home_team_id == team_id) | (Match.away_team_id == team_id)
            )

        matches = query.order_by(Match.kickoff.desc()).limit(n).all()

        if not matches:
            return {"points": 0, "goals_for": 0, "goals_against": 0, "xg": 0}

        points, gf, ga, xg_sum = 0, 0, 0, 0.0
        for m in matches:
            if m.home_team_id == team_id:
                g_for  = m.home_score or 0
                g_ag   = m.away_score or 0
                xg_sum += m.home_xg or g_for
                if g_for > g_ag:
                    points += 3
                elif g_for == g_ag:
                    points += 1
            else:
                g_for  = m.away_score or 0
                g_ag   = m.home_score or 0
                xg_sum += m.away_xg or g_for
                if g_for > g_ag:
                    points += 3
                elif g_for == g_ag:
                    points += 1
            gf += g_for
            ga += g_ag

        n_actual = len(matches)
        return {
            "points":         round(points / n_actual, 2),
            "goals_for":      round(gf / n_actual, 2),
            "goals_against":  round(ga / n_actual, 2),
            "xg":             round(xg_sum / n_actual, 2),
        }

    def _get_h2h(
        self, db, home_id: int, away_id: int,
        before: datetime, n: int = 10
    ) -> dict:
        """Head-to-head tra le due squadre."""
        matches = db.query(Match).filter(
            Match.status == "finished",
            Match.kickoff < before,
            (
                ((Match.home_team_id == home_id) & (Match.away_team_id == away_id))
                | ((Match.home_team_id == away_id) & (Match.away_team_id == home_id))
            ),
        ).order_by(Match.kickoff.desc()).limit(n).all()

        if not matches:
            return {
                "home_wins": 0, "draws": 0, "away_wins": 0,
                "home_goals_avg": 1.2, "away_goals_avg": 1.0, "btts_rate": 0.5,
            }

        hw, draws, aw = 0, 0, 0
        hg_sum, ag_sum, btts = 0, 0, 0

        for m in matches:
            hg = (m.home_score or 0) if m.home_team_id == home_id else (m.away_score or 0)
            ag = (m.away_score or 0) if m.away_team_id == away_id else (m.home_score or 0)
            hg_sum += hg
            ag_sum += ag
            if hg > 0 and ag > 0:
                btts += 1
            if hg > ag:
                hw += 1
            elif hg == ag:
                draws += 1
            else:
                aw += 1

        n_actual = len(matches)
        return {
            "home_wins":      hw / n_actual,
            "draws":          draws / n_actual,
            "away_wins":      aw / n_actual,
            "home_goals_avg": hg_sum / n_actual,
            "away_goals_avg": ag_sum / n_actual,
            "btts_rate":      btts / n_actual,
        }

    def _get_season_stats(self, db, team_id: int, before: datetime) -> dict:
        """xG medio stagionale di una squadra."""
        season_start = before.replace(month=8, day=1)
        if before.month < 8:
            season_start = season_start.replace(year=before.year - 1)

        matches = db.query(Match).filter(
            Match.status == "finished",
            Match.kickoff >= season_start,
            Match.kickoff < before,
            (Match.home_team_id == team_id) | (Match.away_team_id == team_id),
        ).all()

        if not matches:
            return {"xg_per_match": 1.2, "xga_per_match": 1.2}

        xg_sum, xga_sum = 0.0, 0.0
        for m in matches:
            if m.home_team_id == team_id:
                xg_sum  += m.home_xg or m.home_score or 1.2
                xga_sum += m.away_xg or m.away_score or 1.2
            else:
                xg_sum  += m.away_xg or m.away_score or 1.2
                xga_sum += m.home_xg or m.home_score or 1.2

        n = len(matches)
        return {
            "xg_per_match":  round(xg_sum / n, 2),
            "xga_per_match": round(xga_sum / n, 2),
        }

    def _days_since_last_match(self, db, team_id: int, before: datetime) -> float:
        last = db.query(Match).filter(
            Match.status == "finished",
            Match.kickoff < before,
            (Match.home_team_id == team_id) | (Match.away_team_id == team_id),
        ).order_by(Match.kickoff.desc()).first()

        if not last or not last.kickoff:
            return 7.0
        return min((before - last.kickoff).days, 30)

    def _lineup_strength(self, db, match_id: int, team_id: int) -> float:
        """
        Stima la forza della formazione basandosi sui minuti totali giocati
        dagli 11 titolari (proxy per esperienza/forma).
        """
        from sqlalchemy import func

        lineup = db.query(Lineup).filter_by(
            match_id=match_id,
            team_id=team_id,
        ).order_by(Lineup.is_official.desc()).first()

        if not lineup:
            return 0.5

        starters = db.query(LineupPlayer).filter_by(
            lineup_id=lineup.id,
            role="starter",
        ).all()

        if not starters:
            return 0.5

        total_minutes = 0
        for lp in starters:
            if lp.player_id:
                stats = db.query(PlayerStats).filter_by(
                    player_id=lp.player_id
                ).order_by(PlayerStats.season.desc()).first()
                if stats:
                    total_minutes += stats.minutes or 0

        # Normalizza: ~900 min per giocatore = full season ~ 9000 totale
        return min(total_minutes / 9000.0, 1.0)

    def _count_key_injuries(self, db, team_id: int) -> int:
        """Conta giocatori chiave infortunati (per questa squadra)."""
        injured = db.query(Injury).join(Player).filter(
            Player.team_id == team_id,
            Injury.is_active == True,
        ).count()
        return min(injured, 5)

    # ─────────────────────────────────────────────────────────
    # POISSON — RISULTATI ESATTI
    # ─────────────────────────────────────────────────────────

    def _estimate_goal_rates(self, features: pd.Series) -> tuple[float, float]:
        """
        Stima il lambda (gol attesi) per home e away usando il modello Poisson
        o un calcolo diretto dalle feature se il modello non è addestrato.
        """
        if self.goals_model:
            return self.goals_model.predict(features)

        # Fallback diretto: media xG ponderata
        home_xg  = float(features.get("home_xg_season", 1.3))
        away_xga = float(features.get("away_xga_season", 1.2))
        away_xg  = float(features.get("away_xg_season", 1.1))
        home_xga = float(features.get("home_xga_season", 1.1))

        home_lambda = (home_xg * 0.6 + away_xga * 0.4)
        away_lambda = (away_xg * 0.6 + home_xga * 0.4)

        # Aggiusta con form
        form_diff = float(features.get("form_diff", 0))
        home_lambda *= 1 + form_diff * 0.02
        away_lambda *= 1 - form_diff * 0.02

        return max(0.3, home_lambda), max(0.3, away_lambda)

    @staticmethod
    def _poisson_score_probs(
        home_lambda: float, away_lambda: float, max_goals: int = 6
    ) -> dict:
        """
        Probabilità di ogni risultato esatto usando distribuzione di Poisson bivariata.
        Ritorna dict {"0-0": prob, "1-0": prob, ...} ordinato per prob decrescente.
        """
        from scipy.stats import poisson

        probs = {}
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                p = poisson.pmf(h, home_lambda) * poisson.pmf(a, away_lambda)
                probs[f"{h}-{a}"] = round(float(p) * 100, 2)

        # Ordina e prendi top 10
        sorted_scores = sorted(probs.items(), key=lambda x: -x[1])
        return {s: p for s, p in sorted_scores[:10]}

    @staticmethod
    def _calc_btts(home_lambda: float, away_lambda: float) -> float:
        """P(home scores ≥ 1) × P(away scores ≥ 1)."""
        from scipy.stats import poisson
        p_home_scores = 1 - poisson.pmf(0, home_lambda)
        p_away_scores = 1 - poisson.pmf(0, away_lambda)
        return float(p_home_scores * p_away_scores)

    @staticmethod
    def _calc_over_n5(home_lambda: float, away_lambda: float, threshold: float) -> float:
        """P(total goals > threshold) via Poisson."""
        from scipy.stats import poisson
        prob_under = 0.0
        for total in range(int(threshold) + 1):
            for h in range(total + 1):
                a = total - h
                prob_under += poisson.pmf(h, home_lambda) * poisson.pmf(a, away_lambda)
        return 1.0 - float(prob_under)

    # ─────────────────────────────────────────────────────────
    # MARCATORI E AMMONITI
    # ─────────────────────────────────────────────────────────

    def _predict_scorers(
        self, match_id: int, home_lambda: float, away_lambda: float
    ) -> list[dict]:
        """
        Calcola la probabilità che ogni giocatore in formazione segni.
        Formula: P(segna) = (xG_individuale / xG_squadra) × P(squadra segna ≥ 1)
        """
        from scipy.stats import poisson

        result = []
        with get_db_session() as db:
            for is_home, lambda_val in [(True, home_lambda), (False, away_lambda)]:
                team_id = db.query(Match).filter_by(id=match_id).first()
                if not team_id:
                    continue
                team_id = team_id.home_team_id if is_home else team_id.away_team_id

                lineup = db.query(Lineup).filter_by(
                    match_id=match_id,
                    team_id=team_id,
                ).order_by(Lineup.is_official.desc()).first()

                if not lineup:
                    continue

                starters = db.query(LineupPlayer).filter_by(
                    lineup_id=lineup.id,
                    role="starter",
                ).all()

                p_team_scores = 1 - poisson.pmf(0, lambda_val)

                for lp in starters:
                    if not lp.player or not lp.player_id:
                        continue

                    player = lp.player
                    # Ignora portieri
                    pos = (lp.position or player.position or "").upper()
                    if "GK" in pos or "G" == pos:
                        continue

                    # Gol per partita dallo storico
                    stats = db.query(PlayerStats).filter_by(
                        player_id=lp.player_id
                    ).order_by(PlayerStats.season.desc()).first()

                    if stats and stats.appearances and stats.appearances > 0:
                        gpm = (stats.goals or 0) / stats.appearances
                        xg_pm = (stats.xg or gpm) / max(stats.appearances, 1)
                    else:
                        # Default per ruolo
                        defaults = {"CF": 0.35, "LW": 0.20, "RW": 0.20,
                                    "AM": 0.15, "CM": 0.08, "DM": 0.04,
                                    "CB": 0.03, "LB": 0.04, "RB": 0.04}
                        gpm  = defaults.get(pos[:2], 0.10)
                        xg_pm = gpm

                    # Probabilità che segni in questa partita
                    p_score = min(0.95, xg_pm * p_team_scores * 1.2)
                    if p_score < 0.03:
                        continue

                    result.append({
                        "player_id": lp.player_id,
                        "name":      player.name,
                        "team_id":   team_id,
                        "position":  pos,
                        "prob":      round(p_score * 100, 1),
                    })

        # Ordina per probabilità
        result.sort(key=lambda x: -x["prob"])
        return result[:10]

    def _predict_bookings(self, match_id: int) -> list[dict]:
        """
        Calcola la probabilità che ogni giocatore prenda un cartellino giallo.
        Basato su: yellow_cards / appearances nello storico.
        """
        result = []
        with get_db_session() as db:
            match = db.query(Match).filter_by(id=match_id).first()
            if not match:
                return []

            for team_id in [match.home_team_id, match.away_team_id]:
                lineup = db.query(Lineup).filter_by(
                    match_id=match_id,
                    team_id=team_id,
                ).order_by(Lineup.is_official.desc()).first()

                if not lineup:
                    continue

                starters = db.query(LineupPlayer).filter_by(
                    lineup_id=lineup.id,
                    role="starter",
                ).all()

                for lp in starters:
                    if not lp.player_id:
                        continue

                    player = lp.player
                    stats = db.query(PlayerStats).filter_by(
                        player_id=lp.player_id
                    ).order_by(PlayerStats.season.desc()).first()

                    if stats and stats.appearances and stats.appearances > 0:
                        ypm = (stats.yellow_cards or 0) / stats.appearances
                    else:
                        pos = (lp.position or "").upper()
                        # Centrocampisti e difensori prendono più gialli
                        ypm_defaults = {
                            "DM": 0.22, "CM": 0.18, "CB": 0.16,
                            "LB": 0.14, "RB": 0.14, "AM": 0.14,
                            "LW": 0.10, "RW": 0.10, "CF": 0.10,
                        }
                        ypm = ypm_defaults.get(pos[:2], 0.12)

                    if ypm < 0.05:
                        continue

                    result.append({
                        "player_id": lp.player_id,
                        "name":      player.name if player else "Unknown",
                        "team_id":   team_id,
                        "prob":      round(min(ypm * 100, 65), 1),
                    })

        result.sort(key=lambda x: -x["prob"])
        return result[:8]

    # ─────────────────────────────────────────────────────────
    # POISSON MODEL (fit semplice)
    # ─────────────────────────────────────────────────────────

    def _fit_poisson_model(self, X_rows, home_goals, away_goals):
        """
        Modello semplice di Poisson per stimare i gol attesi.
        In una versione avanzata si può usare Dixon-Coles o SciPy minimize.
        Per ora: regressione di Poisson via scikit-learn.
        """
        try:
            from sklearn.linear_model import PoissonRegressor
            from sklearn.preprocessing import StandardScaler

            X = np.array(X_rows)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            home_reg = PoissonRegressor(alpha=0.1, max_iter=300)
            away_reg = PoissonRegressor(alpha=0.1, max_iter=300)

            home_reg.fit(X_scaled, home_goals)
            away_reg.fit(X_scaled, away_goals)

            class PoissonModel:
                def __init__(self, h_reg, a_reg, scaler):
                    self.home_reg = h_reg
                    self.away_reg = a_reg
                    self.scaler   = scaler

                def predict(self, features: pd.Series):
                    X = features.values.reshape(1, -1)
                    X_s = self.scaler.transform(X)
                    h = float(self.home_reg.predict(X_s)[0])
                    a = float(self.away_reg.predict(X_s)[0])
                    return max(0.3, h), max(0.2, a)

            return PoissonModel(home_reg, away_reg, scaler)

        except Exception as e:
            logger.warning(f"Poisson model fit fallito: {e}")
            return None

    # ─────────────────────────────────────────────────────────
    # SAVE / LOAD MODELLI
    # ─────────────────────────────────────────────────────────

    def _save_models(self):
        try:
            if self.outcome_model:
                with open(MODEL_DIR / "outcome_model.pkl", "wb") as f:
                    pickle.dump(self.outcome_model, f)
            if self.goals_model:
                with open(MODEL_DIR / "goals_model.pkl", "wb") as f:
                    pickle.dump(self.goals_model, f)
            if self.booked_model:
                with open(MODEL_DIR / "booked_model.pkl", "wb") as f:
                    pickle.dump(self.booked_model, f)
            logger.info(f"Modelli salvati in {MODEL_DIR}")
        except Exception as e:
            logger.error(f"Errore salvataggio modelli: {e}")

    def _load_models(self):
        for attr, filename in [
            ("outcome_model", "outcome_model.pkl"),
            ("goals_model",   "goals_model.pkl"),
            ("booked_model",  "booked_model.pkl"),
        ]:
            path = MODEL_DIR / filename
            if path.exists():
                try:
                    with open(path, "rb") as f:
                        setattr(self, attr, pickle.load(f))
                    logger.info(f"Modello {filename} caricato")
                except Exception as e:
                    logger.warning(f"Errore caricamento {filename}: {e}")


predictor = FootballPredictor()
