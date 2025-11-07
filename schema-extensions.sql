-- Schema extensions for XGBoost UFC Prediction System
-- Created: 2025-11-06

-- Table for upcoming UFC fight cards
CREATE TABLE IF NOT EXISTS upcoming_fight_cards (
    card_id SERIAL PRIMARY KEY,
    event_name VARCHAR(128) NOT NULL,
    event_date DATE NOT NULL,
    location VARCHAR(64),
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_processed BOOLEAN DEFAULT FALSE,
    UNIQUE(event_name, event_date)
);

-- Table for individual fights on upcoming cards
CREATE TABLE IF NOT EXISTS upcoming_fights (
    upcoming_fight_id SERIAL PRIMARY KEY,
    card_id INTEGER NOT NULL REFERENCES upcoming_fight_cards(card_id) ON DELETE CASCADE,
    fighter1_id INTEGER NOT NULL REFERENCES fighters(fighter_id),
    fighter2_id INTEGER NOT NULL REFERENCES fighters(fighter_id),
    weight_class VARCHAR(128),
    fight_order INTEGER,
    bout_type VARCHAR(64),
    CHECK (fighter1_id != fighter2_id),
    UNIQUE(card_id, fighter1_id, fighter2_id)
);

-- Table for ML predictions
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id SERIAL PRIMARY KEY,
    upcoming_fight_id INTEGER NOT NULL REFERENCES upcoming_fights(upcoming_fight_id) ON DELETE CASCADE,
    fighter1_id INTEGER NOT NULL REFERENCES fighters(fighter_id),
    fighter2_id INTEGER NOT NULL REFERENCES fighters(fighter_id),
    fighter1_win_probability DECIMAL(5,2) NOT NULL CHECK (fighter1_win_probability >= 0 AND fighter1_win_probability <= 100),
    fighter2_win_probability DECIMAL(5,2) NOT NULL CHECK (fighter2_win_probability >= 0 AND fighter2_win_probability <= 100),
    model_version VARCHAR(32),
    prediction_summary TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CHECK (ABS((fighter1_win_probability + fighter2_win_probability) - 100.0) < 0.01),
    UNIQUE(upcoming_fight_id, model_version)
);

-- Table for tracking model performance
CREATE TABLE IF NOT EXISTS model_performance (
    performance_id SERIAL PRIMARY KEY,
    prediction_id INTEGER NOT NULL REFERENCES predictions(prediction_id),
    actual_winner_id INTEGER REFERENCES fighters(fighter_id),
    actual_outcome VARCHAR(8),  -- W/L, L/W, NC/NC, D/D
    was_correct BOOLEAN,
    confidence_level DECIMAL(5,2),
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for model training metadata
CREATE TABLE IF NOT EXISTS model_versions (
    version_id SERIAL PRIMARY KEY,
    version_name VARCHAR(32) UNIQUE NOT NULL,
    training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    num_training_samples INTEGER,
    num_features INTEGER,
    accuracy_score DECIMAL(5,4),
    precision_score DECIMAL(5,4),
    recall_score DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    model_params JSONB,
    notes TEXT
);

-- Indexes for performance
CREATE INDEX idx_upcoming_fights_card ON upcoming_fights(card_id);
CREATE INDEX idx_upcoming_fights_fighters ON upcoming_fights(fighter1_id, fighter2_id);
CREATE INDEX idx_predictions_upcoming_fight ON predictions(upcoming_fight_id);
CREATE INDEX idx_predictions_created ON predictions(created_at DESC);
CREATE INDEX idx_model_performance_prediction ON model_performance(prediction_id);
CREATE INDEX idx_upcoming_cards_date ON upcoming_fight_cards(event_date DESC);
CREATE INDEX idx_upcoming_cards_processed ON upcoming_fight_cards(is_processed);

-- View for easy prediction retrieval
CREATE OR REPLACE VIEW upcoming_predictions AS
SELECT
    ufc.card_id,
    ufc.event_name,
    ufc.event_date,
    ufc.location,
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
    p.model_version,
    p.created_at AS prediction_date
FROM upcoming_fight_cards ufc
JOIN upcoming_fights uf ON ufc.card_id = uf.card_id
JOIN fighters f1 ON uf.fighter1_id = f1.fighter_id
JOIN fighters f2 ON uf.fighter2_id = f2.fighter_id
LEFT JOIN predictions p ON uf.upcoming_fight_id = p.upcoming_fight_id
ORDER BY ufc.event_date DESC, uf.fight_order ASC;

COMMENT ON TABLE upcoming_fight_cards IS 'Stores information about upcoming UFC events';
COMMENT ON TABLE upcoming_fights IS 'Stores individual fights for upcoming events';
COMMENT ON TABLE predictions IS 'Stores XGBoost model predictions for upcoming fights';
COMMENT ON TABLE model_performance IS 'Tracks actual vs predicted outcomes for model evaluation';
COMMENT ON TABLE model_versions IS 'Metadata about trained model versions';
