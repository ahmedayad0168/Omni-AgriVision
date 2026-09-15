import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class YieldPredictor:
    def __init__(self, model_type: str = "gradient_boosting", model_path: Optional[str] = None):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        if model_path and Path(model_path).exists():
            self.load(model_path)

    def _create_model(self):
        if self.model_type == "gradient_boosting":
            return GradientBoostingRegressor(n_estimators= 200, learning_rate= 0.05, max_depth= 5,
                                              min_samples_split= 10,random_state= 42)
        elif self.model_type == "random_forest":
            return RandomForestRegressor(n_estimators= 200, max_depth= 10, min_samples_split= 10, random_state= 42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def extract_features(self, obs: pd.DataFrame, weather: Optional[pd.DataFrame] = None, 
                         satellite: Optional[pd.DataFrame] = None) -> np.ndarray:
        features = []
        features.append(len(obs))  # plant_count
        healthy_ratio = obs['is_healthy'].mean() if len(obs) > 0 and 'is_healthy' in obs else 0.7
        features.append(healthy_ratio * 100)  # healthy_percentage
        severity_vals = obs['severity'] if 'severity' in obs and len(obs) > 0 else []
        features.append(np.mean(severity_vals) if len(severity_vals) > 0 else 0.0)  # disease_severity_mean
        pest_count = len(obs[obs['class_name'].str.contains('mite|pest', case=False, na=False)]) if 'class_name' in obs and len(obs) > 0 else 0
        features.append(pest_count / max(len(obs), 1))  # pest_density
        # Weather
        if weather is not None:
            features.extend([
                weather.get('temperature_mean', 25.0),
                weather.get('humidity_mean', 60.0),
                weather.get('precipitation_total', 0.0)
            ])
        else:
            features.extend([25.0, 60.0, 0.0])
        # Satellite
        if satellite is not None:
            features.append(satellite.get('ndvi_mean', 0.5))
        else:
            features.append(0.5)
        return np.array(features).reshape(1, -1)

    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Dict[str, float]:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        self.model = self._create_model()
        self.model.fit(X_train_scaled, y_train)
        y_pred = self.model.predict(X_test_scaled)
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()
        return metrics

    def predict(self, features: np.ndarray, return_interval: bool = True) -> Dict[str, Any]:
        if self.model is None:
            raise ValueError("Model not trained")
        features_scaled = self.scaler.transform(features)
        pred = self.model.predict(features_scaled)[0]
        if return_interval:
            uncertainty = 0.15 * pred
            return {
                'predicted_yield': float(pred),
                'lower_bound': float(pred - uncertainty),
                'upper_bound': float(pred + uncertainty),
                'confidence': 0.78
            }
        return {'predicted_yield': float(pred)}

    def save(self, path: str):
        joblib.dump({'model': self.model, 'scaler': self.scaler, 'model_type': self.model_type}, path)

    def load(self, path: str):
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.model_type = data.get('model_type', self.model_type)