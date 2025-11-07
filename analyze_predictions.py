#!/usr/bin/env python3
"""
Analyze XGBoost model predictions - right vs wrong
"""
import pickle
import psycopg2
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import os

# Database configuration - use environment variables if available
db_config = {
    'host': os.environ.get('DB_HOST', 'localhost'),
    'port': int(os.environ.get('DB_PORT', '40997')),
    'database': os.environ.get('DB_NAME', 'octagona'),
    'user': os.environ.get('DB_USER', 'postgres'),
    'password': os.environ.get('DB_PASSWORD', 'postgres')
}

def load_model():
    """Load the trained model"""
    model_dir = '/models' if os.path.exists('/models') else '/home/models/xgboost-dev'
    model_files = [f for f in os.listdir(model_dir) if f.startswith('ufc_predictor_') and f.endswith('.pkl')]

    if not model_files:
        raise Exception(f"No model files found in {model_dir}")

    # Get the most recent model
    model_file = sorted(model_files)[-1]
    model_path = os.path.join(model_dir, model_file)

    print(f"Loading model: {model_path}")
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    # Handle both dict format (with metadata) and direct model format
    if isinstance(model_data, dict):
        model = model_data['model']
        print(f"Loaded model with metadata: {model_data.get('version', 'unknown')}")
    else:
        model = model_data
        print("Loaded model (no metadata)")

    return model, model_file

def get_all_fights_data():
    """Load all fights data like the training script does"""
    conn = psycopg2.connect(**db_config)

    # Get fights with outcomes
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
    ORDER BY f.fight_id
    """

    # Get fighter stats
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

    # Get physical attributes
    physical_query = """
    SELECT fighter_id, height, weight, reach, dob FROM fighters
    """

    print("Loading data from database...")
    fights_df = pd.read_sql_query(fights_query, conn)
    stats_df = pd.read_sql_query(stats_query, conn)
    physical_df = pd.read_sql_query(physical_query, conn)
    conn.close()

    print(f"Loaded {len(fights_df)} fights")

    # Parse physical attributes like the training script does
    from datetime import datetime

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

    return fights_df, stats_df, physical_df

def extract_features(fight, stats_df, physical_df):
    """Extract features for a single fight"""
    fighter1_id = fight['fighter1_id']
    fighter2_id = fight['fighter2_id']

    # Get stats
    f1_stats = stats_df[stats_df['fighter_id'] == fighter1_id]
    f2_stats = stats_df[stats_df['fighter_id'] == fighter2_id]

    if len(f1_stats) == 0 or len(f2_stats) == 0:
        return None

    f1_stats = f1_stats.iloc[0]
    f2_stats = f2_stats.iloc[0]

    # Get physical attributes
    f1_physical = physical_df[physical_df['fighter_id'] == fighter1_id]
    f2_physical = physical_df[physical_df['fighter_id'] == fighter2_id]

    if len(f1_physical) == 0 or len(f2_physical) == 0:
        return None

    f1_physical = f1_physical.iloc[0]
    f2_physical = f2_physical.iloc[0]

    # Determine winner (1 = fighter1 wins, 0 = fighter2 wins)
    outcome = fight['outcome']
    if outcome == 'W/L':
        label = 1  # Fighter 1 wins
    else:  # L/W
        label = 0  # Fighter 2 wins

    # Build features dictionary (exactly as in training script)
    features = {
        # Fighter 1 features
        'f1_total_fights': float(f1_stats['total_fights']),
        'f1_avg_sig_strikes': float(f1_stats['avg_sig_strikes'] or 0),
        'f1_striking_accuracy': float(f1_stats['striking_accuracy'] or 0),
        'f1_avg_takedowns': float(f1_stats['avg_takedowns'] or 0),
        'f1_takedown_accuracy': float(f1_stats['takedown_accuracy'] or 0),
        'f1_avg_knockdowns': float(f1_stats['avg_knockdowns'] or 0),
        'f1_avg_sub_attempts': float(f1_stats['avg_sub_attempts'] or 0),
        'f1_height': float(f1_physical['height_inches'] or 0),
        'f1_weight': float(f1_physical['weight_lbs'] or 0),
        'f1_reach': float(f1_physical['reach_inches'] or 0),
        'f1_age': float(f1_physical['age'] or 0),

        # Fighter 2 features
        'f2_total_fights': float(f2_stats['total_fights']),
        'f2_avg_sig_strikes': float(f2_stats['avg_sig_strikes'] or 0),
        'f2_striking_accuracy': float(f2_stats['striking_accuracy'] or 0),
        'f2_avg_takedowns': float(f2_stats['avg_takedowns'] or 0),
        'f2_takedown_accuracy': float(f2_stats['takedown_accuracy'] or 0),
        'f2_avg_knockdowns': float(f2_stats['avg_knockdowns'] or 0),
        'f2_avg_sub_attempts': float(f2_stats['avg_sub_attempts'] or 0),
        'f2_height': float(f2_physical['height_inches'] or 0),
        'f2_weight': float(f2_physical['weight_lbs'] or 0),
        'f2_reach': float(f2_physical['reach_inches'] or 0),
        'f2_age': float(f2_physical['age'] or 0),

        # Differential features
        'experience_diff': float(f1_stats['total_fights'] - f2_stats['total_fights']),
        'striking_acc_diff': float((f1_stats['striking_accuracy'] or 0) - (f2_stats['striking_accuracy'] or 0)),
        'takedown_acc_diff': float((f1_stats['takedown_accuracy'] or 0) - (f2_stats['takedown_accuracy'] or 0)),
        'height_diff': float((f1_physical['height_inches'] or 0) - (f2_physical['height_inches'] or 0)),
        'reach_diff': float((f1_physical['reach_inches'] or 0) - (f2_physical['reach_inches'] or 0)),
        'age_diff': float((f1_physical['age'] or 0) - (f2_physical['age'] or 0)),

        'label': label
    }

    return features

def analyze_predictions():
    """Analyze model predictions"""
    # Load model
    model, model_file = load_model()

    # Get feature names
    feature_names = model.get_booster().feature_names
    print(f"\nModel uses {len(feature_names)} features")

    # Load all fights data
    fights_df, stats_df, physical_df = get_all_fights_data()

    # Extract features for all fights
    print("\nExtracting features for all fights...")
    all_features = []
    fight_details = []

    for idx, fight in fights_df.iterrows():
        features = extract_features(fight, stats_df, physical_df)
        if features:
            all_features.append(features)
            fight_details.append({
                'fight_id': fight['fight_id'],
                'bout': fight['bout'],
                'outcome': fight['outcome'],
                'fighter1_id': fight['fighter1_id'],
                'fighter2_id': fight['fighter2_id']
            })

    print(f"Extracted features for {len(all_features)} fights")

    # Convert to DataFrame
    df = pd.DataFrame(all_features)

    # Separate features and labels
    y_true = df['label'].values
    X = df[feature_names].fillna(0)

    # Make predictions
    print("\nMaking predictions...")
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)

    # Calculate accuracy
    correct = (y_pred == y_true).sum()
    total = len(y_true)
    accuracy = correct / total * 100

    print("\n" + "="*70)
    print("PREDICTION ANALYSIS - RIGHT VS WRONG")
    print("="*70)
    print(f"\nTotal Fights Analyzed: {total}")
    print(f"Correct Predictions: {correct}")
    print(f"Incorrect Predictions: {total - correct}")
    print(f"Accuracy: {accuracy:.2f}%")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\n" + "-"*70)
    print("CONFUSION MATRIX")
    print("-"*70)
    print(f"                    Predicted Fighter 2 Win | Predicted Fighter 1 Win")
    print(f"Actual Fighter 2 Win:        {cm[0][0]:4d}          |        {cm[0][1]:4d}")
    print(f"Actual Fighter 1 Win:        {cm[1][0]:4d}          |        {cm[1][1]:4d}")

    # Classification report
    print("\n" + "-"*70)
    print("DETAILED METRICS")
    print("-"*70)
    print(classification_report(y_true, y_pred,
                              target_names=['Fighter 2 Wins', 'Fighter 1 Wins'],
                              digits=4))

    # Analyze confidence levels
    print("\n" + "-"*70)
    print("CONFIDENCE ANALYSIS")
    print("-"*70)

    # Get confidence (max probability) for each prediction
    confidence = y_pred_proba.max(axis=1)

    # Separate correct and incorrect by confidence
    correct_mask = (y_pred == y_true)

    high_conf_correct = ((confidence > 0.7) & correct_mask).sum()
    med_conf_correct = ((confidence > 0.5) & (confidence <= 0.7) & correct_mask).sum()
    low_conf_correct = ((confidence <= 0.5) & correct_mask).sum()

    high_conf_incorrect = ((confidence > 0.7) & ~correct_mask).sum()
    med_conf_incorrect = ((confidence > 0.5) & (confidence <= 0.7) & ~correct_mask).sum()
    low_conf_incorrect = ((confidence <= 0.5) & ~correct_mask).sum()

    print(f"\nHigh Confidence (>70%):")
    print(f"  Correct: {high_conf_correct}")
    print(f"  Incorrect: {high_conf_incorrect}")

    print(f"\nMedium Confidence (50-70%):")
    print(f"  Correct: {med_conf_correct}")
    print(f"  Incorrect: {med_conf_incorrect}")

    print(f"\nLow Confidence (<50%):")
    print(f"  Correct: {low_conf_correct}")
    print(f"  Incorrect: {low_conf_incorrect}")

    # Find some interesting examples
    print("\n" + "="*70)
    print("EXAMPLE PREDICTIONS")
    print("="*70)

    # Most confident correct predictions
    print("\nMOST CONFIDENT CORRECT PREDICTIONS:")
    print("-" * 70)
    correct_indices = np.where(correct_mask)[0]
    if len(correct_indices) > 0:
        most_confident_correct = correct_indices[confidence[correct_indices].argsort()[-5:][::-1]]
        for idx in most_confident_correct:
            bout = fight_details[idx]['bout']
            outcome = fight_details[idx]['outcome']
            conf = confidence[idx]
            pred_f1_prob = y_pred_proba[idx][1] * 100
            pred_f2_prob = y_pred_proba[idx][0] * 100
            winner = "Fighter 1" if y_true[idx] == 1 else "Fighter 2"
            print(f"\n{bout}")
            print(f"  Actual Winner: {winner} ({outcome})")
            print(f"  Predicted: F1={pred_f1_prob:.1f}% | F2={pred_f2_prob:.1f}%")
            print(f"  Confidence: {conf*100:.1f}% ✅")

    # Most confident incorrect predictions
    print("\n\nMOST CONFIDENT INCORRECT PREDICTIONS:")
    print("-" * 70)
    incorrect_indices = np.where(~correct_mask)[0]
    if len(incorrect_indices) > 0:
        most_confident_incorrect = incorrect_indices[confidence[incorrect_indices].argsort()[-5:][::-1]]
        for idx in most_confident_incorrect:
            bout = fight_details[idx]['bout']
            outcome = fight_details[idx]['outcome']
            conf = confidence[idx]
            pred_f1_prob = y_pred_proba[idx][1] * 100
            pred_f2_prob = y_pred_proba[idx][0] * 100
            winner = "Fighter 1" if y_true[idx] == 1 else "Fighter 2"
            predicted = "Fighter 1" if y_pred[idx] == 1 else "Fighter 2"
            print(f"\n{bout}")
            print(f"  Actual Winner: {winner} ({outcome})")
            print(f"  Predicted: F1={pred_f1_prob:.1f}% | F2={pred_f2_prob:.1f}%")
            print(f"  Model thought: {predicted} (but was WRONG)")
            print(f"  Confidence: {conf*100:.1f}% ❌")

    print("\n" + "="*70)
    print(f"\nModel Version: {model_file}")
    print(f"Final Accuracy: {accuracy:.2f}%")
    print("="*70)

if __name__ == '__main__':
    analyze_predictions()
