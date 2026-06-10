"""
ml/predictor.py
Modello predittivo migliorato.

Quando il modello ML non è addestrato usa un sistema basato su regole
che considera: scontri diretti, forma recente, posizione in classifica,
differenza reti, rendimento casa/trasferta, infortuni.
"""
import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from db.database import get_db_session
from db.models import Match, Team, Lineup, LineupPlayer, PlayerStats, Player, Prediction, Injury, Competition

logger = logging.getLogger(__name__)
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


class FootballPredictor:

    def __init__(self):
        self.outcome_model = None
        self.goals_model = None
        self._load_models()

    # ─────────────────────────────────────────────────────────
    # PREDIZIONE PRINCIPALE
    # ─────────────────────────────────────────────────────────

    def predict_match(self, match_id: int) -> Optional[Prediction]:
        with get_db_session() as db:
            match = db.query(Match).filter_by(id=match_id).first()
            if not match:
                return None
            home_id = match.home_team_id
            away_id = match.away_team_id
            comp_id = match.competition_id
            kickoff = match.kickoff

        # Prova ML prima
        if self.outcome_model:
            try:
                features = self.build_match_features(match_id)
                if features is not None:
                    return self._predict_with_ml(match_id, features)
            except Exception as e:
                logger.warning(f"ML fallback to rules for match {match_id}: {e}")

        # Fallback: sistema basato su regole
        return self._predict_with_rules(match_id, home_id, away_id, comp_id, kickoff)

    # ─────────────────────────────────────────────────────────
    # SISTEMA BASATO SU REGOLE (funziona senza ML)
    # ─────────────────────────────────────────────────────────

    def _predict_with_rules(self, match_id, home_id, away_id, comp_id, kickoff) -> Optional[Prediction]:
        """
        Calcola probabilità basandosi esclusivamente sullo storico:
        1. Scontri diretti storici
        2. Forma recente (ultimi 8 match)
        3. Posizione in classifica
        4. Rendimento casa/trasferta
        5. Differenza reti
        6. Infortuni chiave

        Nota: nessuna dipendenza da quote bookmaker — le probabilità
        derivano solo dal confronto tra partite precedenti e statistiche.
        """
        with get_db_session() as db:
            now = kickoff or datetime.utcnow()

            # ── 1. Scontri diretti ──────────────────────────
            h2h = self._get_h2h(db, home_id, away_id, now, n=10)
            h2h_home_rate = h2h["home_wins"] / max(h2h["total"], 1)
            h2h_draw_rate = h2h["draws"] / max(h2h["total"], 1)
            h2h_away_rate = h2h["away_wins"] / max(h2h["total"], 1)

            # ── 2. Forma recente ────────────────────────────
            home_form = self._get_form(db, home_id, now, n=8)
            away_form = self._get_form(db, away_id, now, n=8)

            # ── 3. Forma casa/trasferta specifica ───────────
            home_home_form = self._get_form(db, home_id, now, n=6, venue="home")
            away_away_form = self._get_form(db, away_id, now, n=6, venue="away")

            # ── 4. Classifica della stagione ────────────────
            home_standing = self._get_standing(db, home_id, comp_id, now)
            away_standing = self._get_standing(db, away_id, comp_id, now)

            # ── 5. Infortuni ────────────────────────────────
            home_injuries = self._count_injuries(db, home_id)
            away_injuries = self._count_injuries(db, away_id)

            # ── Calcolo score ───────────────────────────────
            # Base: vantaggio casa (in media 45% vittorie home nel calcio)
            home_score = 1.45
            away_score = 1.00
            draw_score = 0.85

            # Aggiusta per forma recente (punti per partita, max 3)
            if home_form["played"] >= 3:
                home_score += (home_form["ppm"] - 1.5) * 0.4
            if away_form["played"] >= 3:
                away_score += (away_form["ppm"] - 1.5) * 0.4

            # Aggiusta per forma casa/trasferta specifica
            if home_home_form["played"] >= 3:
                home_score += (home_home_form["ppm"] - 1.5) * 0.3
            if away_away_form["played"] >= 3:
                away_score += (away_away_form["ppm"] - 1.5) * 0.3

            # Aggiusta per classifica (differenza posizioni normalizzata)
            if home_standing["position"] > 0 and away_standing["position"] > 0:
                pos_diff = (away_standing["position"] - home_standing["position"]) / 20.0
                home_score += pos_diff * 0.3
                away_score -= pos_diff * 0.3

            # Aggiusta per differenza reti media
            if home_form["played"] >= 3:
                home_score += home_form["gd_per_match"] * 0.15
            if away_form["played"] >= 3:
                away_score += away_form["gd_per_match"] * 0.15

            # Aggiusta per scontri diretti (peso 20%)
            if h2h["total"] >= 3:
                home_score = home_score * 0.80 + h2h_home_rate * 3.0 * 0.20
                away_score = away_score * 0.80 + h2h_away_rate * 3.0 * 0.20
                draw_score = draw_score * 0.80 + h2h_draw_rate * 3.0 * 0.20

            # Penalizza per infortuni
            home_score -= home_injuries * 0.05
            away_score -= away_injuries * 0.05

            # Converti score in probabilità
            total = max(home_score + draw_score + away_score, 0.1)
            prob_home = (home_score / total) * 100
            prob_draw = (draw_score / total) * 100
            prob_away = (away_score / total) * 100

            # Normalizza
            total_prob = prob_home + prob_draw + prob_away
            prob_home = round((prob_home / total_prob) * 100, 1)
            prob_draw = round((prob_draw / total_prob) * 100, 1)
            prob_away = round(100 - prob_home - prob_draw, 1)

            # Gol attesi (Poisson)
            home_goals_avg = home_home_form.get("gf_per_match", 1.3) if home_home_form["played"] >= 3 else 1.3
            away_goals_avg = away_away_form.get("gf_per_match", 1.0) if away_away_form["played"] >= 3 else 1.0
            home_lambda = (home_goals_avg + away_form.get("ga_per_match", 1.2)) / 2
            away_lambda = (away_goals_avg + home_form.get("ga_per_match", 1.1)) / 2
            home_lambda = max(0.3, home_lambda)
            away_lambda = max(0.2, away_lambda)

            score_probs = self._poisson_score_probs(home_lambda, away_lambda)
            btts = round(self._calc_btts(home_lambda, away_lambda) * 100, 1)
            over25 = round(self._calc_over(home_lambda, away_lambda, 2.5) * 100, 1)

            # Marcatori e ammoniti
            scorer_probs = self._predict_scorers(db, match_id, home_lambda, away_lambda)
            booked_probs = self._predict_bookings(db, match_id)

            # Confidence: alta se abbiamo molti dati storici
            data_points = home_form["played"] + away_form["played"] + h2h["total"]
            has_standings = home_standing["position"] > 0 and away_standing["position"] > 0
            confidence = min(0.95, 0.3 + data_points * 0.02 + (0.1 if has_standings else 0))

        # Salva
        with get_db_session() as db:
            pred = db.query(Prediction).filter_by(match_id=match_id).first()
            if not pred:
                pred = Prediction(match_id=match_id)
                db.add(pred)
            pred.prob_home = prob_home
            pred.prob_draw = prob_draw
            pred.prob_away = prob_away
            pred.score_probs = score_probs
            pred.scorer_probs = scorer_probs
            pred.booked_probs = booked_probs
            pred.btts_prob = btts
            pred.over25_prob = over25
            pred.model_version = "v2.0-rules+ml"
            pred.confidence = round(confidence, 2)
            pred.created_at = datetime.utcnow()

        logger.info(f"Previsione match {match_id}: 1={prob_home}% X={prob_draw}% 2={prob_away}% [conf={confidence:.2f}]")
        return pred

    # ─────────────────────────────────────────────────────────
    # HELPER: STATISTICHE
    # ─────────────────────────────────────────────────────────

    def _get_form(self, db, team_id, before, n=8, venue="all") -> dict:
        from sqlalchemy import or_
        q = db.query(Match).filter(
            Match.status == "finished",
            Match.kickoff < before,
            Match.home_score.isnot(None),
        )
        if venue == "home":
            q = q.filter(Match.home_team_id == team_id)
        elif venue == "away":
            q = q.filter(Match.away_team_id == team_id)
        else:
            q = q.filter(or_(Match.home_team_id == team_id, Match.away_team_id == team_id))

        matches = q.order_by(Match.kickoff.desc()).limit(n).all()
        if not matches:
            return {"played": 0, "ppm": 1.2, "gf_per_match": 1.2, "ga_per_match": 1.2, "gd_per_match": 0}

        pts, gf, ga = 0, 0, 0
        for m in matches:
            is_home = m.home_team_id == team_id
            g_for = (m.home_score or 0) if is_home else (m.away_score or 0)
            g_ag = (m.away_score or 0) if is_home else (m.home_score or 0)
            gf += g_for; ga += g_ag
            if g_for > g_ag: pts += 3
            elif g_for == g_ag: pts += 1

        n_real = len(matches)
        return {
            "played": n_real,
            "ppm": pts / n_real,
            "gf_per_match": gf / n_real,
            "ga_per_match": ga / n_real,
            "gd_per_match": (gf - ga) / n_real,
        }

    def _get_h2h(self, db, home_id, away_id, before, n=10) -> dict:
        from sqlalchemy import or_, and_
        matches = db.query(Match).filter(
            Match.status == "finished",
            Match.kickoff < before,
            Match.home_score.isnot(None),
            or_(
                and_(Match.home_team_id == home_id, Match.away_team_id == away_id),
                and_(Match.home_team_id == away_id, Match.away_team_id == home_id),
            )
        ).order_by(Match.kickoff.desc()).limit(n).all()

        if not matches:
            return {"total": 0, "home_wins": 0, "draws": 0, "away_wins": 0,
                    "home_goals_avg": 1.2, "away_goals_avg": 1.0, "btts_rate": 0.5}

        hw, draws, aw, hg, ag = 0, 0, 0, 0, 0
        for m in matches:
            if m.home_team_id == home_id:
                g_h, g_a = m.home_score or 0, m.away_score or 0
            else:
                g_h, g_a = m.away_score or 0, m.home_score or 0
            hg += g_h; ag += g_a
            if g_h > g_a: hw += 1
            elif g_h == g_a: draws += 1
            else: aw += 1

        n_real = len(matches)
        return {
            "total": n_real,
            "home_wins": hw, "draws": draws, "away_wins": aw,
            "home_goals_avg": hg / n_real,
            "away_goals_avg": ag / n_real,
            "btts_rate": sum(1 for m in matches if (m.home_score or 0) > 0 and (m.away_score or 0) > 0) / n_real,
        }

    def _get_standing(self, db, team_id, comp_id, before) -> dict:
        """Posizione in classifica nella stagione corrente."""
        if not comp_id:
            return {"position": 0, "pts": 0}

        if before.month >= 7:
            season_start = datetime(before.year, 7, 1)
        else:
            season_start = datetime(before.year - 1, 7, 1)

        from sqlalchemy import or_
        all_teams_matches = {}
        matches = db.query(Match).filter(
            Match.competition_id == comp_id,
            Match.status == "finished",
            Match.kickoff >= season_start,
            Match.kickoff < before,
            Match.home_score.isnot(None),
        ).all()

        for m in matches:
            for tid in [m.home_team_id, m.away_team_id]:
                if tid not in all_teams_matches:
                    all_teams_matches[tid] = 0
                is_home = m.home_team_id == tid
                g_for = (m.home_score or 0) if is_home else (m.away_score or 0)
                g_ag = (m.away_score or 0) if is_home else (m.home_score or 0)
                if g_for > g_ag: all_teams_matches[tid] += 3
                elif g_for == g_ag: all_teams_matches[tid] += 1

        if not all_teams_matches:
            return {"position": 0, "pts": 0}

        sorted_teams = sorted(all_teams_matches.items(), key=lambda x: -x[1])
        for i, (tid, pts) in enumerate(sorted_teams, 1):
            if tid == team_id:
                return {"position": i, "pts": pts}

        return {"position": len(sorted_teams) + 1, "pts": 0}

    def _count_injuries(self, db, team_id) -> int:
        from db.models import Player
        return min(db.query(Injury).join(Player).filter(
            Player.team_id == team_id,
            Injury.is_active == True
        ).count(), 5)

    # ─────────────────────────────────────────────────────────
    # POISSON
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _poisson_score_probs(hl: float, al: float, max_g: int = 5) -> dict:
        from scipy.stats import poisson
        probs = {}
        for h in range(max_g + 1):
            for a in range(max_g + 1):
                p = poisson.pmf(h, hl) * poisson.pmf(a, al)
                probs[f"{h}-{a}"] = round(float(p) * 100, 2)
        return dict(sorted(probs.items(), key=lambda x: -x[1])[:8])

    @staticmethod
    def _calc_btts(hl: float, al: float) -> float:
        from scipy.stats import poisson
        return float((1 - poisson.pmf(0, hl)) * (1 - poisson.pmf(0, al)))

    @staticmethod
    def _calc_over(hl: float, al: float, threshold: float) -> float:
        from scipy.stats import poisson
        under = sum(
            poisson.pmf(h, hl) * poisson.pmf(t - h, al)
            for t in range(int(threshold) + 1)
            for h in range(t + 1)
        )
        return 1.0 - float(under)

    # ─────────────────────────────────────────────────────────
    # MARCATORI E AMMONITI
    # ─────────────────────────────────────────────────────────

    def _get_squad_candidates(self, db, match_id, team_id) -> list:
        """
        Restituisce i giocatori da considerare per marcatori/ammoniti come
        lista di tuple (player_id, name, position, ultime PlayerStats|None).

        Usa la formazione (ufficiale o probabile) se già disponibile,
        altrimenti ripiega sulla rosa ordinata per presenze nello storico,
        cosi' le previsioni esistono anche prima che le formazioni vengano
        scaricate.
        """
        lineup = db.query(Lineup).filter_by(
            match_id=match_id, team_id=team_id
        ).order_by(Lineup.is_official.desc()).first()

        if lineup:
            starters = db.query(LineupPlayer).filter_by(
                lineup_id=lineup.id, role="starter"
            ).all()
            candidates = []
            for lp in starters:
                if not lp.player_id:
                    continue
                player = lp.player
                pos = (lp.position or (player.position if player else "") or "").upper()
                stats = db.query(PlayerStats).filter_by(
                    player_id=lp.player_id
                ).order_by(PlayerStats.season.desc()).first()
                candidates.append((lp.player_id, player.name if player else "N/D", pos, stats))
            if candidates:
                return candidates

        # Fallback: rosa della squadra ordinata per presenze nello storico
        candidates = []
        for player in db.query(Player).filter_by(team_id=team_id).all():
            stats = db.query(PlayerStats).filter_by(
                player_id=player.id
            ).order_by(PlayerStats.season.desc()).first()
            candidates.append((player.id, player.name, (player.position or "").upper(), stats))

        candidates.sort(key=lambda c: (c[3].appearances or 0) if c[3] else 0, reverse=True)
        return candidates[:11]

    def _predict_scorers(self, db, match_id, home_lambda, away_lambda) -> list:
        from scipy.stats import poisson

        result = []
        match = db.query(Match).filter_by(id=match_id).first()
        if not match:
            return []

        for lam, team_id in [
            (home_lambda, match.home_team_id),
            (away_lambda, match.away_team_id)
        ]:
            p_team = float(1 - poisson.pmf(0, lam))

            for player_id, name, pos, stats in self._get_squad_candidates(db, match_id, team_id):
                if pos in ("GK", "G"):
                    continue

                # Gol per partita dallo storico
                if stats and stats.appearances and stats.appearances > 0:
                    gpm = (stats.goals or 0) / stats.appearances
                    xg = (stats.xg or gpm) / max(stats.appearances, 1)
                else:
                    defaults = {"CF": 0.35, "LW": 0.20, "RW": 0.20, "AM": 0.15,
                                "CM": 0.08, "DM": 0.04, "CB": 0.03, "LB": 0.04, "RB": 0.04}
                    gpm = defaults.get(pos[:2], 0.10)
                    xg = gpm

                p_score = min(0.95, xg * p_team * 1.3)
                if p_score < 0.04:
                    continue

                result.append({
                    "player_id": player_id,
                    "name": name,
                    "team_id": team_id,
                    "prob": round(p_score * 100, 1),
                })

        result.sort(key=lambda x: -x["prob"])
        return result[:10]

    def _predict_bookings(self, db, match_id) -> list:
        result = []
        match = db.query(Match).filter_by(id=match_id).first()
        if not match:
            return []

        for team_id in [match.home_team_id, match.away_team_id]:
            for player_id, name, pos, stats in self._get_squad_candidates(db, match_id, team_id):
                if stats and stats.appearances and stats.appearances > 0:
                    ypm = (stats.yellow_cards or 0) / stats.appearances
                else:
                    defaults = {"DM": 0.22, "CM": 0.18, "CB": 0.16, "LB": 0.14, "RB": 0.14}
                    ypm = defaults.get(pos[:2], 0.12)

                if ypm < 0.05:
                    continue

                result.append({
                    "player_id": player_id,
                    "name": name,
                    "team_id": team_id,
                    "prob": round(min(ypm * 100, 60), 1),
                })

        result.sort(key=lambda x: -x["prob"])
        return result[:8]

    # ─────────────────────────────────────────────────────────
    # ML (quando addestrato)
    # ─────────────────────────────────────────────────────────

    def build_match_features(self, match_id: int) -> Optional[pd.Series]:
        with get_db_session() as db:
            match = db.query(Match).filter_by(id=match_id).first()
            if not match or not match.kickoff:
                return None
            home_id = match.home_team_id
            away_id = match.away_team_id
            now = match.kickoff

            h2h = self._get_h2h(db, home_id, away_id, now)
            hf = self._get_form(db, home_id, now, n=10)
            af = self._get_form(db, away_id, now, n=10)
            hhf = self._get_form(db, home_id, now, n=6, venue="home")
            aaf = self._get_form(db, away_id, now, n=6, venue="away")
            hs = self._get_standing(db, home_id, match.competition_id, now)
            as_ = self._get_standing(db, away_id, match.competition_id, now)
            hi = self._count_injuries(db, home_id)
            ai = self._count_injuries(db, away_id)

            f = {
                "home_ppm": hf["ppm"], "away_ppm": af["ppm"],
                "home_gf": hf["gf_per_match"], "away_gf": af["gf_per_match"],
                "home_ga": hf["ga_per_match"], "away_ga": af["ga_per_match"],
                "home_home_ppm": hhf["ppm"], "away_away_ppm": aaf["ppm"],
                "home_pos": hs["position"], "away_pos": as_["position"],
                "pos_diff": as_["position"] - hs["position"],
                "h2h_home_rate": h2h["home_wins"] / max(h2h["total"], 1),
                "h2h_draw_rate": h2h["draws"] / max(h2h["total"], 1),
                "h2h_away_rate": h2h["away_wins"] / max(h2h["total"], 1),
                "h2h_total": h2h["total"],
                "h2h_home_goals_avg": h2h["home_goals_avg"],
                "h2h_away_goals_avg": h2h["away_goals_avg"],
                "h2h_btts_rate": h2h["btts_rate"],
                "home_injuries": hi, "away_injuries": ai,
            }
            return pd.Series(f)

    def _predict_with_ml(self, match_id, features) -> Optional[Prediction]:
        X = features.values.reshape(1, -1)
        probs = self.outcome_model.predict_proba(X)[0]
        classes = self.outcome_model.classes_

        prob_map = {c: p for c, p in zip(classes, probs)}
        prob_home = round(float(prob_map.get(1, 0.33)) * 100, 1)
        prob_draw = round(float(prob_map.get(0, 0.33)) * 100, 1)
        prob_away = round(float(prob_map.get(2, 0.34)) * 100, 1)

        # Normalizza
        total = prob_home + prob_draw + prob_away
        prob_home = round(prob_home / total * 100, 1)
        prob_draw = round(prob_draw / total * 100, 1)
        prob_away = round(100 - prob_home - prob_draw, 1)

        # Usa regole per i dettagli (gol, marcatori ecc)
        with get_db_session() as db:
            match = db.query(Match).filter_by(id=match_id).first()
            if not match:
                return None
            home_id, away_id = match.home_team_id, match.away_team_id
            now = match.kickoff or datetime.utcnow()
            hf = self._get_form(db, home_id, now, n=6, venue="home")
            af = self._get_form(db, away_id, now, n=6, venue="away")
            hl = max(0.3, (hf["gf_per_match"] + af["ga_per_match"]) / 2)
            al = max(0.2, (af["gf_per_match"] + hf["ga_per_match"]) / 2)
            score_probs = self._poisson_score_probs(hl, al)
            btts = round(self._calc_btts(hl, al) * 100, 1)
            over25 = round(self._calc_over(hl, al, 2.5) * 100, 1)
            scorer_probs = self._predict_scorers(db, match_id, hl, al)
            booked_probs = self._predict_bookings(db, match_id)

        with get_db_session() as db:
            pred = db.query(Prediction).filter_by(match_id=match_id).first()
            if not pred:
                pred = Prediction(match_id=match_id)
                db.add(pred)
            pred.prob_home = prob_home
            pred.prob_draw = prob_draw
            pred.prob_away = prob_away
            pred.score_probs = score_probs
            pred.scorer_probs = scorer_probs
            pred.booked_probs = booked_probs
            pred.btts_prob = btts
            pred.over25_prob = over25
            pred.model_version = "v2.0-xgboost"
            pred.confidence = 0.85
            pred.created_at = datetime.utcnow()

        return pred

    def train(self, min_matches: int = 50) -> dict:
        try:
            import xgboost as xgb
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score
        except ImportError as e:
            return {"error": f"Missing dependency: {e}"}

        X_rows, y = [], []
        with get_db_session() as db:
            finished = db.query(Match).filter(
                Match.status == "finished",
                Match.home_score.isnot(None),
            ).all()

        logger.info(f"Partite per training: {len(finished)}")
        if len(finished) < min_matches:
            return {"error": "insufficient_data", "available": len(finished)}

        for m in finished:
            try:
                f = self.build_match_features(m.id)
                if f is None:
                    continue
                hg = m.home_score or 0
                ag = m.away_score or 0
                if hg > ag: outcome = 1
                elif hg == ag: outcome = 0
                else: outcome = 2
                X_rows.append(f.values)
                y.append(outcome)
            except Exception:
                continue

        if len(X_rows) < min_matches:
            return {"error": "insufficient_features", "available": len(X_rows)}

        X = np.array(X_rows)
        y_arr = np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y_arr, test_size=0.2, random_state=42, stratify=y_arr)

        self.outcome_model = xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss", random_state=42,
        )
        self.outcome_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        acc = accuracy_score(y_test, self.outcome_model.predict(X_test))
        self._save_models()

        return {"outcome_accuracy": round(acc, 3), "training_samples": len(X_rows), "model_version": "v2.0-xgboost"}

    def _save_models(self):
        if self.outcome_model:
            with open(MODEL_DIR / "outcome_model.pkl", "wb") as f:
                pickle.dump(self.outcome_model, f)

    def _load_models(self):
        path = MODEL_DIR / "outcome_model.pkl"
        if path.exists():
            try:
                with open(path, "rb") as f:
                    self.outcome_model = pickle.load(f)
                logger.info("Modello ML caricato")
            except Exception as e:
                logger.warning(f"Errore caricamento modello: {e}")


predictor = FootballPredictor()
