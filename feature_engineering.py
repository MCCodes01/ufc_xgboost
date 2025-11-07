"""
UFC Fight Prediction - Feature Engineering Module
Extracts and engineers features from historical fight data for XGBoost training
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import psycopg2
from typing import Dict, Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UFCFeatureEngineer:
    """Handles feature extraction and engineering for UFC fight predictions"""

    def __init__(self, db_config: Dict[str, str]):
        """
        Initialize feature engineer with database configuration

        Args:
            db_config: Dictionary with keys: host, port, database, user, password
        """
        self.db_config = db_config
        self.conn = None

    def connect(self):
        """Establish database connection"""
        self.conn = psycopg2.connect(**self.db_config)
        logger.info("Connected to database")

    def disconnect(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Disconnected from database")

    def get_fighter_career_stats(self, fighter_id: int, up_to_date: str = None) -> Dict:
        """
        Calculate fighter's career statistics up to a specific date

        Args:
            fighter_id: Fighter's ID
            up_to_date: Calculate stats only up to this date (for temporal consistency)

        Returns:
            Dictionary of career statistics
        """
        query = """
        WITH fighter_fights AS (
            SELECT
                f.fight_id,
                f.event_id,
                e.date as fight_date,
                f.bout,
                f.outcome,
                f.method,
                f.round,
                f.weight_class,
                CASE
                    WHEN f.bout LIKE %s THEN 1
                    WHEN f.bout LIKE %s THEN 2
                    ELSE 0
                END as fighter_position,
                CASE
                    WHEN f.bout LIKE %s AND f.outcome IN ('W/L', 'D/D') THEN 'W'
                    WHEN f.bout LIKE %s AND f.outcome IN ('L/W', 'D/D') THEN 'L'
                    WHEN f.bout LIKE %s AND f.outcome IN ('W/L') THEN 'L'
                    WHEN f.bout LIKE %s AND f.outcome IN ('L/W') THEN 'W'
                    ELSE 'NC'
                END as result
            FROM fights f
            JOIN events e ON f.event_id = e.event_id
            JOIN fighters fi ON (
                (f.bout LIKE '%' || fi.first_name || ' ' || fi.last_name || '%' AND fi.fighter_id = %s)
                OR (f.bout LIKE '%' || fi.nickname || '%' AND fi.fighter_id = %s AND fi.nickname IS NOT NULL)
            )
            WHERE fi.fighter_id = %s
        """

        params = [
            f'%', f'% vs %', f'%', f'% vs %', f'% vs %', f'% vs %',
            fighter_id, fighter_id, fighter_id
        ]

        if up_to_date:
            query += " AND e.date < %s"
            params.append(up_to_date)

        query += " ORDER BY e.date ASC"

        df = pd.read_sql_query(query, self.conn, params=params)

        if len(df) == 0:
            return self._empty_career_stats()

        # Calculate career statistics
        total_fights = len(df)
        wins = len(df[df['result'] == 'W'])
        losses = len(df[df['result'] == 'L'])

        # Win rate
        win_rate = wins / total_fights if total_fights > 0 else 0.0

        # Finish rates
        ko_wins = len(df[(df['result'] == 'W') & (df['method'].str.contains('KO|TKO', case=False, na=False))])
        sub_wins = len(df[(df['result'] == 'W') & (df['method'].str.contains('Submission', case=False, na=False))])
        dec_wins = len(df[(df['result'] == 'W') & (df['method'].str.contains('Decision', case=False, na=False))])

        ko_rate = ko_wins / wins if wins > 0 else 0.0
        sub_rate = sub_wins / wins if wins > 0 else 0.0
        dec_rate = dec_wins / wins if wins > 0 else 0.0

        # Recent form (last 5 fights)
        recent_fights = df.tail(5)
        recent_wins = len(recent_fights[recent_fights['result'] == 'W'])
        recent_win_rate = recent_wins / len(recent_fights) if len(recent_fights) > 0 else 0.0

        # Win/loss streak
        current_streak = 0
        streak_type = None
        for result in reversed(df['result'].tolist()):
            if result == 'NC':
                continue
            if streak_type is None:
                streak_type = result
                current_streak = 1
            elif result == streak_type:
                current_streak += 1
            else:
                break

        # Average fight duration (in rounds)
        avg_fight_length = df['round'].mean() if len(df) > 0 else 0.0

        return {
            'total_fights': total_fights,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'ko_rate': ko_rate,
            'submission_rate': sub_rate,
            'decision_rate': dec_rate,
            'recent_win_rate': recent_win_rate,
            'current_streak': current_streak if streak_type == 'W' else -current_streak,
            'avg_fight_length': avg_fight_length
        }

    def get_fighter_stats_average(self, fighter_id: int, up_to_date: str = None, last_n_fights: int = 5) -> Dict:
        """
        Calculate average fight statistics for a fighter

        Args:
            fighter_id: Fighter's ID
            up_to_date: Calculate stats only up to this date
            last_n_fights: Number of recent fights to average

        Returns:
            Dictionary of averaged fight statistics
        """
        query = """
        WITH fighter_fight_stats AS (
            SELECT
                fs.*,
                e.date as fight_date
            FROM fight_stats fs
            JOIN fights f ON fs.fight_id = f.fight_id
            JOIN events e ON f.event_id = e.event_id
            WHERE fs.fighter_id = %s
        """

        params = [fighter_id]

        if up_to_date:
            query += " AND e.date < %s"
            params.append(up_to_date)

        query += f" ORDER BY e.date DESC LIMIT {last_n_fights})"
        query += """
        SELECT
            AVG(COALESCE(knockdowns, 0)) as avg_knockdowns,
            AVG(COALESCE(submission_attempts, 0)) as avg_sub_attempts,
            AVG(COALESCE(CAST(takedowns AS FLOAT) / NULLIF(takedowns_attempted, 0), 0)) as takedown_accuracy,
            AVG(COALESCE(CAST(sig_strikes AS FLOAT) / NULLIF(sig_strikes_attempted, 0), 0)) as striking_accuracy,
            AVG(COALESCE(total_strikes, 0)) as avg_total_strikes,
            AVG(COALESCE(sig_strikes, 0)) as avg_sig_strikes,
            AVG(COALESCE(takedowns, 0)) as avg_takedowns,
            AVG(COALESCE(takedowns_attempted, 0)) as avg_takedown_attempts
        FROM fighter_fight_stats
        """

        df = pd.read_sql_query(query, self.conn, params=params)

        if df.empty or df.iloc[0].isnull().all():
            return self._empty_stats_average()

        return {
            'avg_knockdowns': float(df.iloc[0]['avg_knockdowns'] or 0),
            'avg_sub_attempts': float(df.iloc[0]['avg_sub_attempts'] or 0),
            'takedown_accuracy': float(df.iloc[0]['takedown_accuracy'] or 0),
            'striking_accuracy': float(df.iloc[0]['striking_accuracy'] or 0),
            'avg_total_strikes': float(df.iloc[0]['avg_total_strikes'] or 0),
            'avg_sig_strikes': float(df.iloc[0]['avg_sig_strikes'] or 0),
            'avg_takedowns': float(df.iloc[0]['avg_takedowns'] or 0),
            'avg_takedown_attempts': float(df.iloc[0]['avg_takedown_attempts'] or 0)
        }

    def get_fighter_physical_stats(self, fighter_id: int) -> Dict:
        """Get fighter's physical attributes"""
        query = """
        SELECT
            height,
            weight,
            reach,
            stance,
            dob
        FROM fighters
        WHERE fighter_id = %s
        """

        df = pd.read_sql_query(query, self.conn, params=[fighter_id])

        if df.empty:
            return self._empty_physical_stats()

        # Parse height to inches
        height_str = df.iloc[0]['height']
        height_inches = self._parse_height(height_str)

        # Parse weight to pounds
        weight_str = df.iloc[0]['weight']
        weight_lbs = self._parse_weight(weight_str)

        # Parse reach to inches
        reach_str = df.iloc[0]['reach']
        reach_inches = self._parse_reach(reach_str)

        # Calculate age
        dob = df.iloc[0]['dob']
        age = self._calculate_age(dob)

        # Encode stance
        stance = df.iloc[0]['stance']
        stance_encoded = self._encode_stance(stance)

        return {
            'height_inches': height_inches,
            'weight_lbs': weight_lbs,
            'reach_inches': reach_inches,
            'age': age,
            'stance_orthodox': stance_encoded['orthodox'],
            'stance_southpaw': stance_encoded['southpaw'],
            'stance_switch': stance_encoded['switch']
        }

    def engineer_fight_features(self, fighter1_id: int, fighter2_id: int,
                                fight_date: str = None) -> Dict:
        """
        Engineer all features for a fight matchup

        Args:
            fighter1_id: First fighter's ID
            fighter2_id: Second fighter's ID
            fight_date: Date of fight (for temporal consistency)

        Returns:
            Dictionary of engineered features
        """
        # Get stats for both fighters
        f1_career = self.get_fighter_career_stats(fighter1_id, fight_date)
        f2_career = self.get_fighter_career_stats(fighter2_id, fight_date)

        f1_stats = self.get_fighter_stats_average(fighter1_id, fight_date)
        f2_stats = self.get_fighter_stats_average(fighter2_id, fight_date)

        f1_physical = self.get_fighter_physical_stats(fighter1_id)
        f2_physical = self.get_fighter_physical_stats(fighter2_id)

        # Create differential features
        features = {}

        # Career stat differentials
        features['win_rate_diff'] = f1_career['win_rate'] - f2_career['win_rate']
        features['total_fights_diff'] = f1_career['total_fights'] - f2_career['total_fights']
        features['ko_rate_diff'] = f1_career['ko_rate'] - f2_career['ko_rate']
        features['sub_rate_diff'] = f1_career['submission_rate'] - f2_career['submission_rate']
        features['recent_form_diff'] = f1_career['recent_win_rate'] - f2_career['recent_win_rate']
        features['streak_diff'] = f1_career['current_streak'] - f2_career['current_streak']

        # Fight stats differentials
        features['striking_acc_diff'] = f1_stats['striking_accuracy'] - f2_stats['striking_accuracy']
        features['takedown_acc_diff'] = f1_stats['takedown_accuracy'] - f2_stats['takedown_accuracy']
        features['avg_strikes_diff'] = f1_stats['avg_sig_strikes'] - f2_stats['avg_sig_strikes']
        features['avg_takedowns_diff'] = f1_stats['avg_takedowns'] - f2_stats['avg_takedowns']

        # Physical differentials
        features['height_diff'] = f1_physical['height_inches'] - f2_physical['height_inches']
        features['weight_diff'] = f1_physical['weight_lbs'] - f2_physical['weight_lbs']
        features['reach_diff'] = f1_physical['reach_inches'] - f2_physical['reach_inches']
        features['age_diff'] = f1_physical['age'] - f2_physical['age']

        # Stance matchup
        features['stance_advantage'] = self._calculate_stance_advantage(
            f1_physical['stance_orthodox'], f1_physical['stance_southpaw'],
            f2_physical['stance_orthodox'], f2_physical['stance_southpaw']
        )

        # Individual fighter features (prefixed)
        for key, value in f1_career.items():
            features[f'f1_{key}'] = value
        for key, value in f1_stats.items():
            features[f'f1_{key}'] = value
        for key, value in f1_physical.items():
            features[f'f1_{key}'] = value

        for key, value in f2_career.items():
            features[f'f2_{key}'] = value
        for key, value in f2_stats.items():
            features[f'f2_{key}'] = value
        for key, value in f2_physical.items():
            features[f'f2_{key}'] = value

        return features

    # Helper methods
    def _empty_career_stats(self) -> Dict:
        return {
            'total_fights': 0, 'wins': 0, 'losses': 0, 'win_rate': 0.0,
            'ko_rate': 0.0, 'submission_rate': 0.0, 'decision_rate': 0.0,
            'recent_win_rate': 0.0, 'current_streak': 0, 'avg_fight_length': 0.0
        }

    def _empty_stats_average(self) -> Dict:
        return {
            'avg_knockdowns': 0.0, 'avg_sub_attempts': 0.0, 'takedown_accuracy': 0.0,
            'striking_accuracy': 0.0, 'avg_total_strikes': 0.0, 'avg_sig_strikes': 0.0,
            'avg_takedowns': 0.0, 'avg_takedown_attempts': 0.0
        }

    def _empty_physical_stats(self) -> Dict:
        return {
            'height_inches': 0.0, 'weight_lbs': 0.0, 'reach_inches': 0.0,
            'age': 0.0, 'stance_orthodox': 0, 'stance_southpaw': 0, 'stance_switch': 0
        }

    def _parse_height(self, height_str: str) -> float:
        """Convert height string like 5'11" to inches"""
        if not height_str or pd.isna(height_str):
            return 0.0
        try:
            parts = height_str.replace('"', '').replace("'", ' ').split()
            feet = int(parts[0]) if len(parts) > 0 else 0
            inches = int(parts[1]) if len(parts) > 1 else 0
            return float(feet * 12 + inches)
        except:
            return 0.0

    def _parse_weight(self, weight_str: str) -> float:
        """Convert weight string like '185 lbs' to pounds"""
        if not weight_str or pd.isna(weight_str):
            return 0.0
        try:
            return float(weight_str.split()[0])
        except:
            return 0.0

    def _parse_reach(self, reach_str: str) -> float:
        """Convert reach string to inches"""
        if not reach_str or pd.isna(reach_str):
            return 0.0
        try:
            return float(reach_str.replace('"', '').strip())
        except:
            return 0.0

    def _calculate_age(self, dob) -> float:
        """Calculate age from date of birth"""
        if not dob or pd.isna(dob):
            return 0.0
        try:
            if isinstance(dob, str):
                dob = datetime.strptime(dob, '%Y-%m-%d')
            age = (datetime.now() - dob).days / 365.25
            return float(age)
        except:
            return 0.0

    def _encode_stance(self, stance: str) -> Dict:
        """One-hot encode fighter stance"""
        if not stance or pd.isna(stance):
            return {'orthodox': 0, 'southpaw': 0, 'switch': 0}

        stance_lower = stance.lower()
        return {
            'orthodox': 1 if 'orthodox' in stance_lower else 0,
            'southpaw': 1 if 'southpaw' in stance_lower else 0,
            'switch': 1 if 'switch' in stance_lower else 0
        }

    def _calculate_stance_advantage(self, f1_orth: int, f1_south: int,
                                    f2_orth: int, f2_south: int) -> int:
        """
        Calculate stance matchup advantage
        Orthodox vs Southpaw typically favors southpaw slightly
        """
        if f1_south and f2_orth:
            return 1  # Fighter 1 has slight advantage
        elif f1_orth and f2_south:
            return -1  # Fighter 2 has slight advantage
        else:
            return 0  # No stance advantage
