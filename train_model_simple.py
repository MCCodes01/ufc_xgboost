"""
UFC Fight Prediction - Simplified XGBoost Training Pipeline
Uses basic aggregated stats without complex temporal feature engineering
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import psycopg2
import pickle
import json
import logging
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleUFCPredictor:
    """Simplified XGBoost-based UFC fight outcome predictor"""

    def __init__(self, db_config: dict):
        self.db_config = db_config
        self.model = None
        self.feature_names = None
        self.model_version = None

    def extract_training_data(self) -> pd.DataFrame:
        """Extract training data with simple aggregated features"""
        logger.info("Connecting to database...")
        conn = psycopg2.connect(**self.db_config)

        # Get fights with fighter IDs from fight_stats
        query = """
        SELECT DISTINCT
            f.fight_id,
            f.outcome,
            e.date as fight_date,
            fs1.fighter_id as fighter1_id,
            fs2.fighter_id as fighter2_id
        FROM fights f
        JOIN events e ON f.event_id = e.event_id
        JOIN fight_stats fs1 ON f.fight_id = fs1.fight_id
        JOIN fight_stats fs2 ON f.fight_id = fs2.fight_id
        WHERE f.outcome IN ('W/L', 'L/W')
        AND fs1.fighter_id < fs2.fighter_id
        ORDER BY e.date ASC
        """

        logger.info("Fetching historical fights...")
        fights_df = pd.read_sql_query(query, conn)
        logger.info(f"Found {len(fights_df)} historical fights")

        # Get fighter aggregate stats
        logger.info("Calculating fighter statistics...")
        fighter_stats = self._get_fighter_stats(conn)

        # Get fighter physical attributes
        logger.info("Getting fighter physical attributes...")
        fighter_physical = self._get_fighter_physical(conn)

        conn.close()

        # Build training dataset
        logger.info("Building training dataset...")
        training_data = []

        for idx, row in fights_df.iterrows():
            if (idx + 1) % 500 == 0:
                logger.info(f"Processing fight {idx + 1}/{len(fights_df)}")

            try:
                f1_id = int(row['fighter1_id'])
                f2_id = int(row['fighter2_id'])

                # Get stats for both fighters
                f1_stats = fighter_stats.get(f1_id, {})
                f2_stats = fighter_stats.get(f2_id, {})
                f1_phys = fighter_physical.get(f1_id, {})
                f2_phys = fighter_physical.get(f2_id, {})

                # Create feature vector
                features = {
                    # Fighter 1 stats
                    'f1_total_fights': f1_stats.get('total_fights', 0),
                    'f1_wins': f1_stats.get('wins', 0),
                    'f1_losses': f1_stats.get('losses', 0),
                    'f1_win_rate': f1_stats.get('win_rate', 0),
                    'f1_avg_sig_strikes': f1_stats.get('avg_sig_strikes', 0),
                    'f1_avg_sig_strikes_attempted': f1_stats.get('avg_sig_strikes_attempted', 0),
                    'f1_striking_accuracy': f1_stats.get('striking_accuracy', 0),
                    'f1_avg_takedowns': f1_stats.get('avg_takedowns', 0),
                    'f1_avg_takedowns_attempted': f1_stats.get('avg_takedowns_attempted', 0),
                    'f1_takedown_accuracy': f1_stats.get('takedown_accuracy', 0),
                    'f1_avg_knockdowns': f1_stats.get('avg_knockdowns', 0),
                    'f1_avg_sub_attempts': f1_stats.get('avg_sub_attempts', 0),
                    'f1_height': f1_phys.get('height', 0),
                    'f1_weight': f1_phys.get('weight', 0),
                    'f1_reach': f1_phys.get('reach', 0),
                    'f1_age': f1_phys.get('age', 0),

                    # Fighter 2 stats
                    'f2_total_fights': f2_stats.get('total_fights', 0),
                    'f2_wins': f2_stats.get('wins', 0),
                    'f2_losses': f2_stats.get('losses', 0),
                    'f2_win_rate': f2_stats.get('win_rate', 0),
                    'f2_avg_sig_strikes': f2_stats.get('avg_sig_strikes', 0),
                    'f2_avg_sig_strikes_attempted': f2_stats.get('avg_sig_strikes_attempted', 0),
                    'f2_striking_accuracy': f2_stats.get('striking_accuracy', 0),
                    'f2_avg_takedowns': f2_stats.get('avg_takedowns', 0),
                    'f2_avg_takedowns_attempted': f2_stats.get('avg_takedowns_attempted', 0),
                    'f2_takedown_accuracy': f2_stats.get('takedown_accuracy', 0),
                    'f2_avg_knockdowns': f2_stats.get('avg_knockdowns', 0),
                    'f2_avg_sub_attempts': f2_stats.get('avg_sub_attempts', 0),
                    'f2_height': f2_phys.get('height', 0),
                    'f2_weight': f2_phys.get('weight', 0),
                    'f2_reach': f2_phys.get('reach', 0),
                    'f2_age': f2_phys.get('age', 0),

                    # Differentials
                    'win_rate_diff': f1_stats.get('win_rate', 0) - f2_stats.get('win_rate', 0),
                    'experience_diff': f1_stats.get('total_fights', 0) - f2_stats.get('total_fights', 0),
                    'striking_acc_diff': f1_stats.get('striking_accuracy', 0) - f2_stats.get('striking_accuracy', 0),
                    'takedown_acc_diff': f1_stats.get('takedown_accuracy', 0) - f2_stats.get('takedown_accuracy', 0),
                    'height_diff': f1_phys.get('height', 0) - f2_phys.get('height', 0),
                    'reach_diff': f1_phys.get('reach', 0) - f2_phys.get('reach', 0),
                    'age_diff': f1_phys.get('age', 0) - f2_phys.get('age', 0),
                }

                # Determine label
                label = self._get_winner_label(conn, row['fight_id'], f1_id, f2_id, row['outcome'])
                features['label'] = label

                training_data.append(features)

            except Exception as e:
                logger.warning(f"Error processing fight {row['fight_id']}: {e}")
                continue

        logger.info(f"Successfully processed {len(training_data)} fights")
        return pd.DataFrame(training_data)

    def _get_fighter_stats(self, conn) -> dict:
        """Get aggregated statistics for all fighters"""
        query = """
        WITH fighter_outcomes AS (
            SELECT
                fs.fighter_id,
                f.outcome,
                f.bout,
                CASE
                    WHEN (f.bout LIKE CONCAT(fi.first_name, ' ', fi.last_name, '%') AND f.outcome = 'W/L')
                        OR (f.bout NOT LIKE CONCAT(fi.first_name, ' ', fi.last_name, '%') AND f.outcome = 'L/W')
                    THEN 1
                    ELSE 0
                END as won
            FROM fight_stats fs
            JOIN fights f ON fs.fight_id = f.fight_id
            JOIN fighters fi ON fs.fighter_id = fi.fighter_id
            WHERE f.outcome IN ('W/L', 'L/W')
        ),
        fight_aggregates AS (
            SELECT
                fighter_id,
                COUNT(*) as total_fights,
                SUM(won) as wins,
                COUNT(*) - SUM(won) as losses,
                CAST(SUM(won) AS FLOAT) / NULLIF(COUNT(*), 0) as win_rate,
                AVG(COALESCE(sig_strikes, 0)) as avg_sig_strikes,
                AVG(COALESCE(sig_strikes_attempted, 0)) as avg_sig_strikes_attempted,
                AVG(COALESCE(CAST(sig_strikes AS FLOAT) / NULLIF(sig_strikes_attempted, 0), 0)) as striking_accuracy,
                AVG(COALESCE(takedowns, 0)) as avg_takedowns,
                AVG(COALESCE(takedowns_attempted, 0)) as avg_takedowns_attempted,
                AVG(COALESCE(CAST(takedowns AS FLOAT) / NULLIF(takedowns_attempted, 0), 0)) as takedown_accuracy,
                AVG(COALESCE(knockdowns, 0)) as avg_knockdowns,
                AVG(COALESCE(submission_attempts, 0)) as avg_sub_attempts
            FROM fight_stats fs
            JOIN fighter_outcomes fo USING (fighter_id)
            GROUP BY fighter_id
        )
        SELECT * FROM fight_aggregates
        """

        df = pd.read_sql_query(query, conn)
        return {
            row['fighter_id']: {
                'total_fights': int(row['total_fights']),
                'wins': int(row['wins']),
                'losses': int(row['losses']),
                'win_rate': float(row['win_rate'] or 0),
                'avg_sig_strikes': float(row['avg_sig_strikes'] or 0),
                'avg_sig_strikes_attempted': float(row['avg_sig_strikes_attempted'] or 0),
                'striking_accuracy': float(row['striking_accuracy'] or 0),
                'avg_takedowns': float(row['avg_takedowns'] or 0),
                'avg_takedowns_attempted': float(row['avg_takedowns_attempted'] or 0),
                'takedown_accuracy': float(row['takedown_accuracy'] or 0),
                'avg_knockdowns': float(row['avg_knockdowns'] or 0),
                'avg_sub_attempts': float(row['avg_sub_attempts'] or 0),
            }
            for _, row in df.iterrows()
        }

    def _get_fighter_physical(self, conn) -> dict:
        """Get physical attributes for all fighters"""
        query = """
        SELECT
            fighter_id,
            height,
            weight,
            reach,
            dob
        FROM fighters
        """

        df = pd.read_sql_query(query, conn)

        def parse_height(h):
            if not h or pd.isna(h): return 0
            try:
                parts = str(h).replace('"', '').replace("'", ' ').split()
                feet = int(parts[0]) if len(parts) > 0 else 0
                inches = int(parts[1]) if len(parts) > 1 else 0
                return float(feet * 12 + inches)
            except: return 0

        def parse_weight(w):
            if not w or pd.isna(w): return 0
            try: return float(str(w).split()[0])
            except: return 0

        def parse_reach(r):
            if not r or pd.isna(r): return 0
            try: return float(str(r).replace('"', '').strip())
            except: return 0

        def calc_age(dob):
            if not dob or pd.isna(dob): return 0
            try:
                if isinstance(dob, str):
                    dob = datetime.strptime(dob, '%Y-%m-%d')
                return float((datetime.now() - dob).days / 365.25)
            except: return 0

        return {
            row['fighter_id']: {
                'height': parse_height(row['height']),
                'weight': parse_weight(row['weight']),
                'reach': parse_reach(row['reach']),
                'age': calc_age(row['dob']),
            }
            for _, row in df.iterrows()
        }

    def _get_winner_label(self, conn, fight_id, fighter1_id, fighter2_id, outcome):
        """Determine winner label"""
        cursor = conn.cursor()
        cursor.execute("SELECT bout FROM fights WHERE fight_id = %s", [fight_id])
        bout_str = cursor.fetchone()[0]

        cursor.execute("""
            SELECT fighter_id, first_name, last_name, nickname
            FROM fighters WHERE fighter_id IN (%s, %s)
        """, [fighter1_id, fighter2_id])

        fighters = {r[0]: {'first': r[1], 'last': r[2], 'nick': r[3]} for r in cursor.fetchall()}

        f1_name = f"{fighters[fighter1_id]['first']} {fighters[fighter1_id]['last']}"
        f2_name = f"{fighters[fighter2_id]['first']} {fighters[fighter2_id]['last']}"

        f1_pos = bout_str.find(f1_name)
        f2_pos = bout_str.find(f2_name)

        if f1_pos == -1 and fighters[fighter1_id]['nick']:
            f1_pos = bout_str.find(fighters[fighter1_id]['nick'])
        if f2_pos == -1 and fighters[fighter2_id]['nick']:
            f2_pos = bout_str.find(fighters[fighter2_id]['nick'])

        if f1_pos < f2_pos:
            return 1 if outcome == 'W/L' else 0
        else:
            return 0 if outcome == 'W/L' else 1

    def train(self, df: pd.DataFrame) -> dict:
        """Train XGBoost model"""
        logger.info("Preparing training data...")

        X = df.drop(['label'], axis=1).fillna(0)
        y = df['label']

        self.feature_names = list(X.columns)
        logger.info(f"Training with {len(self.feature_names)} features on {len(X)} samples")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train model
        logger.info("Training XGBoost model...")
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            max_depth=6,
            learning_rate=0.1,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False
        )

        self.model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        # Evaluate
        logger.info("Evaluating model...")
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'num_features': len(self.feature_names)
        }

        logger.info(f"Model Performance:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1_score']:.4f}")
        logger.info(f"  ROC AUC:   {metrics['roc_auc']:.4f}")

        self.model_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return metrics

    def save_model(self, output_dir='/models'):
        """Save model"""
        os.makedirs(output_dir, exist_ok=True)

        model_path = os.path.join(output_dir, f'ufc_predictor_{self.model_version}.pkl')

        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'version': self.model_version
            }, f)

        logger.info(f"Model saved to {model_path}")

    def save_model_to_db(self, metrics: dict):
        """Save model metadata to database"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO model_versions (
                version_name, num_training_samples, num_features,
                accuracy_score, precision_score, recall_score, f1_score,
                model_params
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, [
            self.model_version,
            metrics['train_samples'],
            metrics['num_features'],
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1_score'],
            json.dumps(self.model.get_params())
        ])

        conn.commit()
        conn.close()
        logger.info("Model metadata saved to database")


def main():
    """Main training pipeline"""
    db_config = {
        'host': os.getenv('DB_HOST', 'postgres-dev.xgboost-dev.svc.cluster.local'),
        'port': int(os.getenv('DB_PORT', 5432)),
        'database': os.getenv('DB_NAME', 'octagona'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'devpassword')
    }

    predictor = SimpleUFCPredictor(db_config)

    # Extract training data
    logger.info("Starting simplified training pipeline...")
    df = predictor.extract_training_data()

    if len(df) == 0:
        logger.error("No training data extracted!")
        return

    # Train model
    metrics = predictor.train(df)

    # Save model
    predictor.save_model()
    predictor.save_model_to_db(metrics)

    logger.info("Training pipeline complete!")


if __name__ == '__main__':
    main()
