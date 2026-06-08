"""
ml/train.py
Script standalone per addestrare i modelli.

Uso:
    python ml/train.py
    python ml/train.py --min-matches 100   # abbassa soglia per test
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


def main():
    parser = argparse.ArgumentParser(description="Addestra i modelli ML di FootballHub")
    parser.add_argument("--min-matches", type=int, default=200,
                        help="Numero minimo di partite storiche richieste (default: 200)")
    parser.add_argument("--no-save", action="store_true",
                        help="Non salvare i modelli su disco")
    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("FootballHub — Training ML")
    logger.info("=" * 50)

    # Verifica DB
    from db.database import health_check
    if not health_check():
        logger.error("Database non raggiungibile. Verifica la configurazione in .env")
        sys.exit(1)

    logger.info("DB connesso ✓")

    # Conta partite disponibili
    from db.database import get_db_session
    from db.models import Match
    with get_db_session() as db:
        n_finished = db.query(Match).filter(
            Match.status == "finished",
            Match.home_score.isnot(None),
        ).count()

    logger.info(f"Partite storiche disponibili: {n_finished}")

    if n_finished < 50:
        logger.error(
            f"Troppo poche partite ({n_finished}). "
            f"Esegui prima l'import storico:\n"
            f"  celery -A tasks call tasks.initial_data_import"
        )
        sys.exit(1)

    # Training
    from ml.predictor import predictor
    logger.info(f"Avvio training con soglia {args.min_matches} partite...")

    result = predictor.train(min_matches=args.min_matches)

    if "error" in result:
        logger.error(f"Training fallito: {result}")
        sys.exit(1)

    logger.info("=" * 50)
    logger.info("Training completato!")
    logger.info(f"  Accuracy esito (1/X/2): {result.get('outcome_accuracy', '?')}")
    logger.info(f"  Campioni training:       {result.get('training_samples', '?')}")
    logger.info(f"  Campioni test:           {result.get('test_samples', '?')}")
    logger.info(f"  Versione modello:        {result.get('model_version', '?')}")
    logger.info("=" * 50)

    # Test su una partita futura
    from db.models import Match as MatchModel
    from datetime import datetime
    with get_db_session() as db:
        test_match = db.query(MatchModel).filter(
            MatchModel.status == "scheduled",
            MatchModel.kickoff >= datetime.utcnow(),
        ).first()

    if test_match:
        logger.info(f"Test previsione su: {test_match.home_team.name} vs {test_match.away_team.name}")
        pred = predictor.predict_match(test_match.id)
        if pred:
            logger.info(
                f"  1: {pred.prob_home}%  X: {pred.prob_draw}%  2: {pred.prob_away}%"
            )
            logger.info(f"  BTTS: {pred.btts_prob}%  Over 2.5: {pred.over25_prob}%")


if __name__ == "__main__":
    main()
