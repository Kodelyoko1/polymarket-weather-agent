"""XGBoost weather probability forecasting models."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from datetime import datetime

try:
    import xgboost as xgb
except ImportError:
    xgb = None

logger = logging.getLogger(__name__)

# Weather event types the agent can trade on
EVENT_TYPES = [
    "temp_above_90f",
    "temp_above_32f",
    "precip_any",
    "precip_1in",
    "wind_above_25mph",
]

# Feature columns used by the model
FEATURE_COLS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
    "temp_range",
    "temp_anomaly_7d",
    "precip_7d_sum",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
]


def engineer_features(records: List[Dict]) -> List[Dict]:
    """Engineer features for XGBoost from raw weather records.

    Args:
        records: List of daily weather dicts

    Returns:
        List of records with engineered features
    """
    if not records:
        return []

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Fill NaN values
    for col in [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
        "shortwave_radiation_sum",
        "et0_fao_evapotranspiration",
    ]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # Temperature range
    df["temp_range"] = df["temperature_2m_max"] - df["temperature_2m_min"]

    # Temperature anomaly vs 7-day rolling average
    temp_7d_avg = df["temperature_2m_max"].rolling(7, center=True).mean()
    df["temp_anomaly_7d"] = df["temperature_2m_max"] - temp_7d_avg
    df["temp_anomaly_7d"] = df["temp_anomaly_7d"].fillna(0.0)

    # Precipitation 7-day sum
    df["precip_7d_sum"] = df["precipitation_sum"].rolling(7, center=True).sum()
    df["precip_7d_sum"] = df["precip_7d_sum"].fillna(0.0)

    # Month sin/cos for seasonality
    months = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * (months - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (months - 1) / 12)

    # Day of year sin/cos for seasonality
    doy = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * (doy - 1) / 365)
    df["doy_cos"] = np.cos(2 * np.pi * (doy - 1) / 365)

    return df.to_dict(orient="records")


@dataclass
class WeatherForecastModel:
    """XGBoost binary classifier for weather outcomes."""

    event_type: str
    model: Optional[object] = None
    metrics: Dict = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}

    def fit(self, records: List[Dict], event_type: str) -> Dict:
        """Train XGBoost model on historical weather data.

        Args:
            records: List of daily weather records (from all cities pooled)
            event_type: Event type to predict

        Returns:
            Metrics dict (accuracy, auc, logloss, etc.)
        """
        if xgb is None:
            logger.error("xgboost not installed")
            return {}

        # Engineer features
        engineered = engineer_features(records)
        if not engineered:
            logger.error("Failed to engineer features")
            return {}

        # Build labels
        labels = []
        for record in engineered:
            label = self._get_label(record, event_type)
            labels.append(label)

        # Create feature matrix
        X_list = []
        for record in engineered:
            row = [record.get(col, 0.0) for col in FEATURE_COLS]
            X_list.append(row)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(labels, dtype=np.float32)

        if len(X) < 30:
            logger.warning(f"Not enough data to train {event_type}: {len(X)} records")
            return {}

        # Train/test split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # XGBoost native API
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "seed": 42,
        }

        evals = [(dtrain, "train"), (dtest, "eval")]
        self.model = xgb.train(params, dtrain, num_boost_round=200, evals=evals, verbose_eval=False)

        # Compute metrics
        y_pred_proba = self.model.predict(dtest)
        y_pred = (y_pred_proba > 0.5).astype(int)

        accuracy = np.mean(y_pred == y_test)
        brier = np.mean((y_pred_proba - y_test) ** 2)
        
        # Brier Skill Score
        baseline_brier = np.mean(y_test * (1 - y_test))
        brier_skill = 1 - (brier / baseline_brier) if baseline_brier > 0 else 0

        # AUC
        try:
            from sklearn.metrics import roc_auc_score, log_loss
            auc = roc_auc_score(y_test, y_pred_proba)
            logloss = log_loss(y_test, y_pred_proba)
        except Exception as e:
            logger.warning(f"Failed to compute AUC/logloss: {e}")
            auc = 0.5
            logloss = 0.0

        self.metrics = {
            "event_type": event_type,
            "accuracy": float(accuracy),
            "brier_score": float(brier),
            "brier_skill": float(brier_skill),
            "auc": float(auc),
            "logloss": float(logloss),
            "n_samples": len(X),
            "n_test": len(X_test),
        }

        # Save model
        self._save_model(event_type)
        logger.info(f"Trained {event_type}: accuracy={accuracy:.3f}, auc={auc:.3f}")

        return self.metrics

    def predict_proba(self, records: List[Dict]) -> List[float]:
        """Predict probabilities for records.

        Args:
            records: List of daily weather records

        Returns:
            List of probabilities (0.0-1.0)
        """
        if self.model is None:
            return [0.5] * len(records)

        engineered = engineer_features(records)
        X_list = []
        for record in engineered:
            row = [record.get(col, 0.0) for col in FEATURE_COLS]
            X_list.append(row)

        if not X_list:
            return []

        X = np.array(X_list, dtype=np.float32)
        dmatrix = xgb.DMatrix(X)
        predictions = self.model.predict(dmatrix)
        return [float(p) for p in predictions]

    def predict_single(self, record: Dict) -> float:
        """Predict probability for a single record.

        Args:
            record: Single daily weather record

        Returns:
            Probability (0.0-1.0)
        """
        probs = self.predict_proba([record])
        return probs[0] if probs else 0.5

    def _get_label(self, record: Dict, event_type: str) -> int:
        """Generate label for a record based on event type."""
        if event_type == "temp_above_90f":
            return 1 if record.get("temperature_2m_max", 0) > 32.2 else 0
        elif event_type == "temp_above_32f":
            return 1 if record.get("temperature_2m_max", 0) > 0 else 0
        elif event_type == "precip_any":
            return 1 if record.get("precipitation_sum", 0) > 0.1 else 0
        elif event_type == "precip_1in":
            return 1 if record.get("precipitation_sum", 0) > 25.4 else 0
        elif event_type == "wind_above_25mph":
            return 1 if record.get("wind_speed_10m_max", 0) > 40.2 else 0
        return 0

    def _save_model(self, event_type: str) -> None:
        """Save model and metrics to disk."""
        model_dir = Path("data/pw_models")
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = model_dir / f"{event_type}.ubj"
        if self.model:
            self.model.save_model(str(model_path))

        # Save metrics
        metrics_path = model_dir / f"{event_type}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)

    @classmethod
    def load(cls, event_type: str) -> "WeatherForecastModel":
        """Load model from disk."""
        if xgb is None:
            logger.error("xgboost not installed")
            return cls(event_type=event_type)

        model_path = Path(f"data/pw_models/{event_type}.ubj")
        metrics_path = Path(f"data/pw_models/{event_type}_metrics.json")

        model = None
        metrics = {}

        if model_path.exists():
            try:
                model = xgb.Booster()
                model.load_model(str(model_path))
                logger.info(f"Loaded model for {event_type}")
            except Exception as e:
                logger.warning(f"Failed to load model for {event_type}: {e}")

        if metrics_path.exists():
            try:
                with open(metrics_path) as f:
                    metrics = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metrics for {event_type}: {e}")

        return cls(event_type=event_type, model=model, metrics=metrics)


def train_all_models(weather_by_city: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    """Train all event type models on pooled city data.

    Args:
        weather_by_city: Dict mapping city -> list of daily records

    Returns:
        Dict mapping event_type -> metrics dict
    """
    # Pool all records from all cities
    all_records = []
    for city, records in weather_by_city.items():
        all_records.extend(records)

    if not all_records:
        logger.warning("No records to train on")
        return {}

    results = {}
    for event_type in EVENT_TYPES:
        logger.info(f"Training {event_type}...")
        model = WeatherForecastModel(event_type=event_type)
        metrics = model.fit(all_records, event_type)
        results[event_type] = metrics

    # Save last trained timestamp
    trained_file = Path("data/pw_models/last_trained.txt")
    trained_file.parent.mkdir(parents=True, exist_ok=True)
    trained_file.write_text(datetime.now().isoformat())

    return results
