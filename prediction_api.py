"""
UFC Fight Prediction API
Flask-based REST API for serving XGBoost predictions
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import psycopg2
import pickle
import logging
import os
from datetime import datetime
from typing import Dict, List

from feature_engineering import UFCFeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables
model = None
feature_names = None
model_version = None
feature_engineer = None

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

    # Sort by filename (which includes timestamp) and get latest
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
    """
    Predict fight outcome

    Request body:
    {
        "fighter1_id": 123,
        "fighter2_id": 456
    }
    """
    try:
        data = request.json
        fighter1_id = data.get('fighter1_id')
        fighter2_id = data.get('fighter2_id')

        if not fighter1_id or not fighter2_id:
            return jsonify({'error': 'fighter1_id and fighter2_id are required'}), 400

        # Engineer features
        feature_engineer.connect()
        features = feature_engineer.engineer_fight_features(fighter1_id, fighter2_id)
        feature_engineer.disconnect()

        # Prepare features for prediction
        import pandas as pd
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


@app.route('/api/upcoming-cards', methods=['GET'])
def get_upcoming_cards():
    """Get all upcoming fight cards"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT card_id, event_name, event_date, location, is_processed
            FROM upcoming_fight_cards
            ORDER BY event_date ASC
        """)

        cards = []
        for row in cursor.fetchall():
            cards.append({
                'card_id': row[0],
                'event_name': row[1],
                'event_date': str(row[2]),
                'location': row[3],
                'is_processed': row[4]
            })

        conn.close()

        return jsonify({'cards': cards})

    except Exception as e:
        logger.error(f"Error fetching cards: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/upcoming-cards/<int:card_id>/predictions', methods=['GET'])
def get_card_predictions(card_id: int):
    """
    Get all predictions for a specific card

    Response:
    {
        "card_info": {...},
        "fights": [...]
    }
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get card info
        cursor.execute("""
            SELECT event_name, event_date, location
            FROM upcoming_fight_cards
            WHERE card_id = %s
        """, [card_id])

        card_row = cursor.fetchone()
        if not card_row:
            return jsonify({'error': 'Card not found'}), 404

        card_info = {
            'card_id': card_id,
            'event_name': card_row[0],
            'event_date': str(card_row[1]),
            'location': card_row[2]
        }

        # Get fights with predictions
        cursor.execute("""
            SELECT
                uf.upcoming_fight_id,
                uf.weight_class,
                uf.bout_type,
                uf.fight_order,
                f1.first_name || ' ' || f1.last_name AS fighter1_name,
                f1.nickname AS fighter1_nickname,
                f2.first_name || ' ' || f2.last_name AS fighter2_name,
                f2.nickname AS fighter2_nickname,
                p.fighter1_win_probability,
                p.fighter2_win_probability,
                p.prediction_summary,
                p.model_version
            FROM upcoming_fights uf
            JOIN fighters f1 ON uf.fighter1_id = f1.fighter_id
            JOIN fighters f2 ON uf.fighter2_id = f2.fighter_id
            LEFT JOIN predictions p ON uf.upcoming_fight_id = p.upcoming_fight_id
            WHERE uf.card_id = %s
            ORDER BY uf.fight_order ASC
        """, [card_id])

        fights = []
        for row in cursor.fetchall():
            fights.append({
                'upcoming_fight_id': row[0],
                'weight_class': row[1],
                'bout_type': row[2],
                'fight_order': row[3],
                'fighter1': {
                    'name': row[4],
                    'nickname': row[5],
                    'win_probability': float(row[8]) if row[8] else None
                },
                'fighter2': {
                    'name': row[6],
                    'nickname': row[7],
                    'win_probability': float(row[9]) if row[9] else None
                },
                'summary': row[10],
                'model_version': row[11]
            })

        conn.close()

        return jsonify({
            'card_info': card_info,
            'fights': fights
        })

    except Exception as e:
        logger.error(f"Error fetching predictions: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/fighters/search', methods=['GET'])
def search_fighters():
    """
    Search for fighters by name
    Query param: q (search query)
    """
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
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT version_name, training_date, num_training_samples, num_features,
               accuracy_score, precision_score, recall_score, f1_score
        FROM model_versions
        ORDER BY training_date DESC
        LIMIT 1
    """)

    row = cursor.fetchone()
    conn.close()

    if row:
        return jsonify({
            'version': row[0],
            'training_date': str(row[1]),
            'num_training_samples': row[2],
            'num_features': row[3],
            'accuracy': float(row[4]) if row[4] else None,
            'precision': float(row[5]) if row[5] else None,
            'recall': float(row[6]) if row[6] else None,
            'f1_score': float(row[7]) if row[7] else None
        })
    else:
        return jsonify({'error': 'No model metadata found'}), 404


if __name__ == '__main__':
    # Initialize feature engineer
    feature_engineer = UFCFeatureEngineer(DB_CONFIG)

    # Load model
    try:
        load_latest_model()
    except FileNotFoundError as e:
        logger.warning(f"No model loaded: {e}")
        logger.warning("API will start but predictions will fail until model is trained")

    # Start server
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
