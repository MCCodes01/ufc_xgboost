"""
UFC Fight Prediction - XGBoost Training Pipeline
Trains a binary classifier to predict fight outcomes
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import psycopg2
import pickle
import json
import logging
from datetime import datetime
from typing import Tuple, Dict
import os

from feature_engineering import UFCFeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UFCFightPredictor:
    """XGBoost-based UFC fight outcome predictor"""

    def __init__(self, db_config: Dict[str, str]):
        """
        Initialize predictor with database configuration

        Args:
            db_config: Dictionary with DB connection parameters
        """
        self.db_config = db_config
        self.feature_engineer = UFCFeatureEngineer(db_config)
        self.model = None
        self.feature_names = None
        self.model_version = None

    def extract_training_data(self, min_date: str = None, max_date: str = None) -> pd.DataFrame:
        """
        Extract and prepare training data from historical fights

        Args:
            min_date: Minimum fight date to include
            max_date: Maximum fight date to include

        Returns:
            DataFrame with features and labels
        """
        logger.info("Connecting to database...")
        self.feature_engineer.connect()

        # Get all historical fights with fighter IDs using fight_stats table
        query = """
        SELECT DISTINCT
            f.fight_id,
            f.outcome,
            e.date as fight_date,
            f.weight_class,
            fs1.fighter_id as fighter1_id,
            fs2.fighter_id as fighter2_id
        FROM fights f
        JOIN events e ON f.event_id = e.event_id
        JOIN fight_stats fs1 ON f.fight_id = fs1.fight_id
        JOIN fight_stats fs2 ON f.fight_id = fs2.fight_id
        WHERE f.outcome IN ('W/L', 'L/W')
        AND fs1.fighter_id < fs2.fighter_id
        """

        params = []
        if min_date:
            query += " AND e.date >= %s"
            params.append(min_date)
        if max_date:
            query += " AND e.date <= %s"
            params.append(max_date)

        query += " ORDER BY e.date ASC"

        logger.info("Fetching historical fights with fighter IDs...")
        conn = psycopg2.connect(**self.db_config)
        fights_df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        logger.info(f"Found {len(fights_df)} historical fights")

        # Process each fight
        training_data = []

        for idx, row in fights_df.iterrows():
            if (idx + 1) % 100 == 0:
                logger.info(f"Processing fight {idx + 1}/{len(fights_df)}")

            try:
                fighter1_id = int(row['fighter1_id'])
                fighter2_id = int(row['fighter2_id'])

                # Engineer features
                features = self.feature_engineer.engineer_fight_features(
                    fighter1_id, fighter2_id, row['fight_date']
                )

                # Determine label based on outcome
                # W/L means first fighter in bout won
                # We need to figure out which fighter in our pair won
                # Since we don't have position info, we'll determine winner from fight_stats
                label = self._get_fight_winner(row['fight_id'], fighter1_id, fighter2_id, row['outcome'])

                # Add metadata
                features['fight_id'] = row['fight_id']
                features['fight_date'] = row['fight_date']
                features['label'] = label

                training_data.append(features)

            except Exception as e:
                logger.warning(f"Error processing fight {row['fight_id']}: {e}")
                continue

        self.feature_engineer.disconnect()

        logger.info(f"Successfully processed {len(training_data)} fights")
        return pd.DataFrame(training_data)

    def _get_fight_winner(self, fight_id: int, fighter1_id: int, fighter2_id: int, outcome: str) -> int:
        """
        Determine which of the two fighters won the fight

        Args:
            fight_id: Fight ID
            fighter1_id: First fighter's ID
            fighter2_id: Second fighter's ID
            outcome: Fight outcome string (W/L or L/W)

        Returns:
            1 if fighter1 won, 0 if fighter2 won
        """
        # Get the bout string to determine fighter order
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT bout FROM fights WHERE fight_id = %s
        """, [fight_id])

        result = cursor.fetchone()
        if not result:
            conn.close()
            raise ValueError(f"Fight {fight_id} not found")

        bout_str = result[0]

        # Get fighter names
        cursor.execute("""
            SELECT fighter_id, first_name, last_name, nickname
            FROM fighters
            WHERE fighter_id IN (%s, %s)
        """, [fighter1_id, fighter2_id])

        fighters = {row[0]: {'first': row[1], 'last': row[2], 'nick': row[3]} for row in cursor.fetchall()}
        conn.close()

        # Check which fighter appears first in the bout string
        f1_name = f"{fighters[fighter1_id]['first']} {fighters[fighter1_id]['last']}"
        f2_name = f"{fighters[fighter2_id]['first']} {fighters[fighter2_id]['last']}"

        f1_pos = bout_str.find(f1_name)
        f2_pos = bout_str.find(f2_name)

        # If names not found, try nicknames
        if f1_pos == -1 and fighters[fighter1_id]['nick']:
            f1_pos = bout_str.find(fighters[fighter1_id]['nick'])
        if f2_pos == -1 and fighters[fighter2_id]['nick']:
            f2_pos = bout_str.find(fighters[fighter2_id]['nick'])

        # Determine winner based on position and outcome
        # W/L means first fighter in bout won
        # L/W means second fighter in bout won
        if f1_pos < f2_pos:
            # fighter1 is first in bout
            return 1 if outcome == 'W/L' else 0
        else:
            # fighter2 is first in bout
            return 0 if outcome == 'W/L' else 1

    def train(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Dict:
        """
        Train XGBoost model on prepared data

        Args:
            df: DataFrame with features and labels
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility

        Returns:
            Dictionary with training metrics
        """
        logger.info("Preparing training data...")

        # Separate features and labels
        meta_cols = ['fight_id', 'fight_date', 'label']
        feature_cols = [col for col in df.columns if col not in meta_cols]

        X = df[feature_cols].fillna(0)  # Fill NaN with 0
        y = df['label']

        self.feature_names = feature_cols

        logger.info(f"Training with {len(feature_cols)} features on {len(X)} samples")

        # Split data (chronological split would be better, but random for now)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Train XGBoost model
        logger.info("Training XGBoost model...")

        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            max_depth=6,
            learning_rate=0.1,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            use_label_encoder=False
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

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
            'num_features': len(feature_cols)
        }

        logger.info(f"Model Performance:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1_score']:.4f}")
        logger.info(f"  ROC AUC:   {metrics['roc_auc']:.4f}")

        # Generate model version name
        self.model_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return metrics

    def save_model(self, output_dir: str = '/models'):
        """
        Save trained model and metadata

        Args:
            output_dir: Directory to save model files
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")

        os.makedirs(output_dir, exist_ok=True)

        model_path = os.path.join(output_dir, f'ufc_predictor_{self.model_version}.pkl')
        metadata_path = os.path.join(output_dir, f'model_metadata_{self.model_version}.json')

        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'version': self.model_version
            }, f)

        logger.info(f"Model saved to {model_path}")

        # Save metadata
        metadata = {
            'version': self.model_version,
            'created_at': datetime.now().isoformat(),
            'num_features': len(self.feature_names),
            'feature_names': self.feature_names
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Metadata saved to {metadata_path}")

    def load_model(self, model_path: str):
        """Load a trained model from file"""
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.feature_names = data['feature_names']
            self.model_version = data['version']

        logger.info(f"Loaded model version {self.model_version}")

    def predict_fight(self, fighter1_id: int, fighter2_id: int) -> Dict:
        """
        Predict outcome of a fight between two fighters

        Args:
            fighter1_id: First fighter's ID
            fighter2_id: Second fighter's ID

        Returns:
            Dictionary with prediction probabilities
        """
        if self.model is None:
            raise ValueError("No model loaded. Train or load a model first.")

        # Engineer features
        self.feature_engineer.connect()
        features = self.feature_engineer.engineer_fight_features(fighter1_id, fighter2_id)
        self.feature_engineer.disconnect()

        # Convert to DataFrame with correct feature order
        X = pd.DataFrame([features])[self.feature_names].fillna(0)

        # Predict
        proba = self.model.predict_proba(X)[0]

        return {
            'fighter1_win_probability': float(proba[1] * 100),  # Convert to percentage
            'fighter2_win_probability': float(proba[0] * 100),
            'model_version': self.model_version
        }

    def save_model_to_db(self, metrics: Dict):
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

        logger.info(f"Model metadata saved to database")


def main():
    """Main training pipeline"""
    # Database configuration
    db_config = {
        'host': os.getenv('DB_HOST', 'postgres-dev.xgboost-dev.svc.cluster.local'),
        'port': int(os.getenv('DB_PORT', 5432)),
        'database': os.getenv('DB_NAME', 'octagona'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'devpassword')
    }

    # Initialize predictor
    predictor = UFCFightPredictor(db_config)

    # Extract training data
    logger.info("Starting training pipeline...")
    df = predictor.extract_training_data()

    # Train model
    metrics = predictor.train(df)

    # Save model
    predictor.save_model()

    # Save to database
    predictor.save_model_to_db(metrics)

    logger.info("Training pipeline complete!")


if __name__ == '__main__':
    main()
