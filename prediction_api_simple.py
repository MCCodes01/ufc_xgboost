"""
UFC Fight Prediction API - Simplified Version
Compatible with the trained v2 model
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import psycopg2
import pandas as pd
import pickle
import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables
model = None
feature_names = None
model_version = None

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'postgres-dev.xgboost-dev.svc.cluster.local'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_NAME', 'octagona'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'devpassword')
}


def load_latest_model():
    """Load the most recent trained model"""
    global model, feature_names, model_version

    model_dir = os.getenv('MODEL_DIR', '/models')
    model_files = [f for f in os.listdir(model_dir) if f.startswith('ufc_predictor_') and f.endswith('.pkl')]

    if not model_files:
        raise FileNotFoundError("No trained models found")

    latest_model = sorted(model_files)[-1]
    model_path = os.path.join(model_dir, latest_model)

    logger.info(f"Loading model from {model_path}")

    with open(model_path, 'rb') as f:
        data = pickle.load(f)
        model = data['model']
        feature_names = data['feature_names']
        model_version = data['version']

    logger.info(f"Loaded model version {model_version} with {len(feature_names)} features")


def get_db_connection():
    """Create database connection"""
    return psycopg2.connect(**DB_CONFIG)


def get_fighter_features(fighter_id):
    """Get features for a fighter"""
    conn = get_db_connection()

    # Get fighter stats
    stats_query = """
    SELECT
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
    WHERE fighter_id = %s
    """

    stats_df = pd.read_sql_query(stats_query, conn, params=[fighter_id])

    # Get physical attributes
    physical_query = """
    SELECT height, weight, reach, dob
    FROM fighters
    WHERE fighter_id = %s
    """

    physical_df = pd.read_sql_query(physical_query, conn, params=[fighter_id])
    conn.close()

    if len(stats_df) == 0 or len(physical_df) == 0:
        return None

    # Parse physical attributes
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

    phys = physical_df.iloc[0]
    stats = stats_df.iloc[0]

    return {
        'total_fights': float(stats['total_fights']),
        'avg_sig_strikes': float(stats['avg_sig_strikes'] or 0),
        'striking_accuracy': float(stats['striking_accuracy'] or 0),
        'avg_takedowns': float(stats['avg_takedowns'] or 0),
        'takedown_accuracy': float(stats['takedown_accuracy'] or 0),
        'avg_knockdowns': float(stats['avg_knockdowns'] or 0),
        'avg_sub_attempts': float(stats['avg_sub_attempts'] or 0),
        'height': parse_height(phys['height']),
        'weight': parse_weight(phys['weight']),
        'reach': parse_reach(phys['reach']),
        'age': calc_age(phys['dob'])
    }


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_version': model_version
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict fight outcome"""
    try:
        data = request.json
        fighter1_id = data.get('fighter1_id')
        fighter2_id = data.get('fighter2_id')

        if not fighter1_id or not fighter2_id:
            return jsonify({'error': 'fighter1_id and fighter2_id are required'}), 400

        # Get features for both fighters
        f1_features = get_fighter_features(fighter1_id)
        f2_features = get_fighter_features(fighter2_id)

        if not f1_features or not f2_features:
            return jsonify({'error': 'Fighter data not found'}), 404

        # Build feature vector
        features = {
            'f1_total_fights': f1_features['total_fights'],
            'f1_avg_sig_strikes': f1_features['avg_sig_strikes'],
            'f1_striking_accuracy': f1_features['striking_accuracy'],
            'f1_avg_takedowns': f1_features['avg_takedowns'],
            'f1_takedown_accuracy': f1_features['takedown_accuracy'],
            'f1_avg_knockdowns': f1_features['avg_knockdowns'],
            'f1_avg_sub_attempts': f1_features['avg_sub_attempts'],
            'f1_height': f1_features['height'],
            'f1_weight': f1_features['weight'],
            'f1_reach': f1_features['reach'],
            'f1_age': f1_features['age'],

            'f2_total_fights': f2_features['total_fights'],
            'f2_avg_sig_strikes': f2_features['avg_sig_strikes'],
            'f2_striking_accuracy': f2_features['striking_accuracy'],
            'f2_avg_takedowns': f2_features['avg_takedowns'],
            'f2_takedown_accuracy': f2_features['takedown_accuracy'],
            'f2_avg_knockdowns': f2_features['avg_knockdowns'],
            'f2_avg_sub_attempts': f2_features['avg_sub_attempts'],
            'f2_height': f2_features['height'],
            'f2_weight': f2_features['weight'],
            'f2_reach': f2_features['reach'],
            'f2_age': f2_features['age'],

            'experience_diff': f1_features['total_fights'] - f2_features['total_fights'],
            'striking_acc_diff': f1_features['striking_accuracy'] - f2_features['striking_accuracy'],
            'takedown_acc_diff': f1_features['takedown_accuracy'] - f2_features['takedown_accuracy'],
            'height_diff': f1_features['height'] - f2_features['height'],
            'reach_diff': f1_features['reach'] - f2_features['reach'],
            'age_diff': f1_features['age'] - f2_features['age'],
        }

        # Prepare for prediction
        X = pd.DataFrame([features])[feature_names].fillna(0)

        # Predict
        proba = model.predict_proba(X)[0]

        return jsonify({
            'fighter1_win_probability': round(float(proba[1] * 100), 2),
            'fighter2_win_probability': round(float(proba[0] * 100), 2),
            'model_version': model_version
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/fighters/search', methods=['GET'])
def search_fighters():
    """Search for fighters by name"""
    try:
        query = request.args.get('q', '').strip()
        if len(query) < 2:
            return jsonify({'fighters': []})

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT fighter_id, first_name, last_name, nickname
            FROM fighters
            WHERE
                first_name ILIKE %s OR
                last_name ILIKE %s OR
                nickname ILIKE %s
            LIMIT 20
        """, [f'%{query}%', f'%{query}%', f'%{query}%'])

        fighters = []
        for row in cursor.fetchall():
            fighters.append({
                'fighter_id': row[0],
                'name': f"{row[1]} {row[2]}",
                'nickname': row[3]
            })

        conn.close()

        return jsonify({'fighters': fighters})

    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/model/info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    return jsonify({
        'version': model_version,
        'num_features': len(feature_names),
        'features': feature_names
    })


if __name__ == '__main__':
    # Load model
    try:
        load_latest_model()
    except FileNotFoundError as e:
        logger.warning(f"No model loaded: {e}")
        logger.warning("API will start but predictions will fail until model is trained")

    # Start server
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
