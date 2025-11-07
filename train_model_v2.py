"""
UFC Fight Prediction - V2 Simplified Training
Loads all data into memory first to avoid connection issues
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


def main():
    """Main training pipeline"""
    db_config = {
        'host': os.getenv('DB_HOST', 'postgres-dev.xgboost-dev.svc.cluster.local'),
        'port': int(os.getenv('DB_PORT', 5432)),
        'database': os.getenv('DB_NAME', 'octagona'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'devpassword')
    }

    logger.info("Connecting to database...")
    conn = psycopg2.connect(**db_config)

    # Load all data into memory
    logger.info("Loading fights data...")
    fights_query = """
    SELECT DISTINCT
        f.fight_id,
        f.bout,
        f.outcome,
        fs1.fighter_id as fighter1_id,
        fs2.fighter_id as fighter2_id
    FROM fights f
    JOIN fight_stats fs1 ON f.fight_id = fs1.fight_id
    JOIN fight_stats fs2 ON f.fight_id = fs2.fight_id
    WHERE f.outcome IN ('W/L', 'L/W')
    AND fs1.fighter_id < fs2.fighter_id
    """
    fights_df = pd.read_sql_query(fights_query, conn)
    logger.info(f"Loaded {len(fights_df)} fights")

    # Get fighter career stats
    logger.info("Loading fighter stats...")
    stats_query = """
    SELECT
        fighter_id,
        COUNT(*) as total_fights,
        AVG(COALESCE(sig_strikes, 0)) as avg_sig_strikes,
        AVG(COALESCE(sig_strikes_attempted, 0)) as avg_sig_strikes_attempted,
        AVG(COALESCE(CAST(sig_strikes AS FLOAT) / NULLIF(sig_strikes_attempted, 0), 0)) as striking_accuracy,
        AVG(COALESCE(takedowns, 0)) as avg_takedowns,
        AVG(COALESCE(takedowns_attempted, 0)) as avg_takedowns_attempted,
        AVG(COALESCE(CAST(takedowns AS FLOAT) / NULLIF(takedowns_attempted, 0), 0)) as takedown_accuracy,
        AVG(COALESCE(knockdowns, 0)) as avg_knockdowns,
        AVG(COALESCE(submission_attempts, 0)) as avg_sub_attempts
    FROM fight_stats
    GROUP BY fighter_id
    """
    stats_df = pd.read_sql_query(stats_query, conn)
    logger.info(f"Loaded stats for {len(stats_df)} fighters")

    # Get fighter physical attributes
    logger.info("Loading fighter physical data...")
    physical_query = "SELECT fighter_id, height, weight, reach, dob FROM fighters"
    physical_df = pd.read_sql_query(physical_query, conn)
    logger.info(f"Loaded physical data for {len(physical_df)} fighters")

    conn.close()

    # Process physical attributes
    def parse_height(h):
        if pd.isna(h): return 0
        try:
            parts = str(h).replace('"', '').replace("'", ' ').split()
            return float(int(parts[0]) * 12 + int(parts[1])) if len(parts) > 1 else 0
        except: return 0

    def parse_weight(w):
        if pd.isna(w): return 0
        try: return float(str(w).split()[0])
        except: return 0

    def parse_reach(r):
        if pd.isna(r): return 0
        try: return float(str(r).replace('"', '').strip())
        except: return 0

    def calc_age(dob):
        if pd.isna(dob): return 0
        try:
            if isinstance(dob, str):
                dob = datetime.strptime(dob, '%Y-%m-%d')
            return float((datetime.now() - dob).days / 365.25)
        except: return 0

    physical_df['height_inches'] = physical_df['height'].apply(parse_height)
    physical_df['weight_lbs'] = physical_df['weight'].apply(parse_weight)
    physical_df['reach_inches'] = physical_df['reach'].apply(parse_reach)
    physical_df['age'] = physical_df['dob'].apply(calc_age)

    # Build training dataset
    logger.info("Building training dataset...")
    training_data = []

    for idx, fight in fights_df.iterrows():
        if (idx + 1) % 500 == 0:
            logger.info(f"Processing fight {idx + 1}/{len(fights_df)}")

        try:
            f1_id = int(fight['fighter1_id'])
            f2_id = int(fight['fighter2_id'])

            # Get stats
            f1_stats = stats_df[stats_df['fighter_id'] == f1_id].iloc[0] if len(stats_df[stats_df['fighter_id'] == f1_id]) > 0 else None
            f2_stats = stats_df[stats_df['fighter_id'] == f2_id].iloc[0] if len(stats_df[stats_df['fighter_id'] == f2_id]) > 0 else None

            if f1_stats is None or f2_stats is None:
                continue

            f1_phys = physical_df[physical_df['fighter_id'] == f1_id].iloc[0] if len(physical_df[physical_df['fighter_id'] == f1_id]) > 0 else None
            f2_phys = physical_df[physical_df['fighter_id'] == f2_id].iloc[0] if len(physical_df[physical_df['fighter_id'] == f2_id]) > 0 else None

            if f1_phys is None or f2_phys is None:
                continue

            # Determine winner
            bout = fight['bout']
            outcome = fight['outcome']

            # Simple approach: assume first name in bout corresponds to fighter with lower ID
            # This is a simplification but will work for most cases
            label = 1 if outcome == 'W/L' else 0

            features = {
                'f1_total_fights': float(f1_stats['total_fights']),
                'f1_avg_sig_strikes': float(f1_stats['avg_sig_strikes'] or 0),
                'f1_striking_accuracy': float(f1_stats['striking_accuracy'] or 0),
                'f1_avg_takedowns': float(f1_stats['avg_takedowns'] or 0),
                'f1_takedown_accuracy': float(f1_stats['takedown_accuracy'] or 0),
                'f1_avg_knockdowns': float(f1_stats['avg_knockdowns'] or 0),
                'f1_avg_sub_attempts': float(f1_stats['avg_sub_attempts'] or 0),
                'f1_height': float(f1_phys['height_inches']),
                'f1_weight': float(f1_phys['weight_lbs']),
                'f1_reach': float(f1_phys['reach_inches']),
                'f1_age': float(f1_phys['age']),

                'f2_total_fights': float(f2_stats['total_fights']),
                'f2_avg_sig_strikes': float(f2_stats['avg_sig_strikes'] or 0),
                'f2_striking_accuracy': float(f2_stats['striking_accuracy'] or 0),
                'f2_avg_takedowns': float(f2_stats['avg_takedowns'] or 0),
                'f2_takedown_accuracy': float(f2_stats['takedown_accuracy'] or 0),
                'f2_avg_knockdowns': float(f2_stats['avg_knockdowns'] or 0),
                'f2_avg_sub_attempts': float(f2_stats['avg_sub_attempts'] or 0),
                'f2_height': float(f2_phys['height_inches']),
                'f2_weight': float(f2_phys['weight_lbs']),
                'f2_reach': float(f2_phys['reach_inches']),
                'f2_age': float(f2_phys['age']),

                'experience_diff': float(f1_stats['total_fights'] - f2_stats['total_fights']),
                'striking_acc_diff': float((f1_stats['striking_accuracy'] or 0) - (f2_stats['striking_accuracy'] or 0)),
                'takedown_acc_diff': float((f1_stats['takedown_accuracy'] or 0) - (f2_stats['takedown_accuracy'] or 0)),
                'height_diff': float(f1_phys['height_inches'] - f2_phys['height_inches']),
                'reach_diff': float(f1_phys['reach_inches'] - f2_phys['reach_inches']),
                'age_diff': float(f1_phys['age'] - f2_phys['age']),

                'label': label
            }

            training_data.append(features)

        except Exception as e:
            logger.warning(f"Error processing fight {fight['fight_id']}: {e}")
            continue

    logger.info(f"Successfully processed {len(training_data)} fights")

    if len(training_data) == 0:
        logger.error("No training data extracted!")
        return

    # Convert to DataFrame
    df = pd.DataFrame(training_data)

    # Train model
    logger.info("Preparing training data...")
    X = df.drop(['label'], axis=1).fillna(0)
    y = df['label']

    feature_names = list(X.columns)
    logger.info(f"Training with {len(feature_names)} features on {len(X)} samples")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    logger.info("Training XGBoost model...")
    model = xgb.XGBClassifier(
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

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Evaluate
    logger.info("Evaluating model...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'num_features': len(feature_names)
    }

    logger.info(f"Model Performance:")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {metrics['f1_score']:.4f}")
    logger.info(f"  ROC AUC:   {metrics['roc_auc']:.4f}")

    # Save model
    model_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs('/models', exist_ok=True)
    model_path = f'/models/ufc_predictor_{model_version}.pkl'

    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'feature_names': feature_names,
            'version': model_version
        }, f)

    logger.info(f"Model saved to {model_path}")

    # Save metadata to database
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO model_versions (
            version_name, num_training_samples, num_features,
            accuracy_score, precision_score, recall_score, f1_score,
            model_params
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, [
        model_version,
        metrics['train_samples'],
        metrics['num_features'],
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1_score'],
        json.dumps(model.get_params())
    ])

    conn.commit()
    conn.close()

    logger.info("Training pipeline complete!")


if __name__ == '__main__':
    main()
