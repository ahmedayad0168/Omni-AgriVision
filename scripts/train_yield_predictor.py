import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import logging
from src.prediction.yield_predictor import YieldPredictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def generate_agronomic_dataset(num_samples=1500, random_state=42):
    rng = np.random.RandomState(random_state)
    plant_count = rng.uniform(80, 450, num_samples)
    healthy_pct = rng.uniform(20.0, 98.0, num_samples)
    disease_sev = (100.0 - healthy_pct) / 100.0 * rng.uniform(0.3, 0.9, num_samples)
    pest_density = rng.exponential(0.08, num_samples).clip(0.0, 0.6)
    temp_mean = rng.normal(25.0, 4.5, num_samples).clip(14.0, 38.0)
    humidity_mean = rng.normal(65.0, 12.0, num_samples).clip(25.0, 95.0)
    precip_total = rng.gamma(shape=2.5, scale=15.0, size=num_samples).clip(0.0, 150.0)
    ndvi_mean = (0.25 + 0.65 * (healthy_pct / 100.0) - 0.15 * disease_sev + rng.normal(0, 0.04, num_samples)).clip(0.15, 0.92)

    X = np.column_stack([plant_count, healthy_pct, disease_sev, pest_density, temp_mean, humidity_mean, precip_total, ndvi_mean])
    temp_factor = np.exp(-((temp_mean - 24.0) ** 2) / (2 * 6.0 ** 2))
    precip_factor = np.clip(precip_total / 45.0, 0.4, 1.2)
    health_factor = (healthy_pct / 100.0) * (1.0 - 0.7 * disease_sev)
    pest_factor = np.maximum(0.3, 1.0 - 1.2 * pest_density)
    ndvi_factor = np.clip(ndvi_mean / 0.7, 0.3, 1.3)
    plant_factor = np.clip(plant_count / 250.0, 0.5, 1.3)

    base_yield = 52.0
    yield_val = base_yield * plant_factor * health_factor * pest_factor * temp_factor * precip_factor * ndvi_factor + rng.normal(0, 1.8, num_samples)
    y = np.maximum(5.0, yield_val)
    return X, y

def train_and_save_model(output_path='models/yield_predictor.pkl', model_type='gradient_boosting', samples=1500):
    logger.info('Generating calibrated agronomic training samples...')
    X, y = generate_agronomic_dataset(num_samples=samples)
    logger.info('Training yield predictor model...')
    predictor = YieldPredictor(model_type=model_type)
    metrics = predictor.train(X, y)
    logger.info('Training completed!')
    logger.info('MAE: %s, RMSE: %s, R2: %s', metrics['mae'], metrics['rmse'], metrics['r2'])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    predictor.save(output_path)
    logger.info('Saved model to %s', output_path)
    return predictor, metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='models/yield_predictor.pkl')
    parser.add_argument('--model-type', default='gradient_boosting')
    parser.add_argument('--samples', type=int, default=1500)
    args = parser.parse_args()
    train_and_save_model(output_path=args.output, model_type=args.model_type, samples=args.samples)
