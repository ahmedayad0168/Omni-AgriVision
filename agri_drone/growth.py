from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-leaf rows into per-frame canopy-growth features."""
    df = df.copy()
    if "mask_cov_%" not in df.columns:
        logger.warning("No 'mask_cov_%' column; assuming 100%% leaf coverage per box.")
        df["mask_cov_%"] = 100.0

    df["canopy_area"] = df["w"] * df["h"] * (df["mask_cov_%"] / 100.0)

    frame = df.groupby("frame").agg(
        time_sec=("time_sec", "first"),
        total_canopy_area=("canopy_area", "sum"),
        leaf_count=("leaf", "count"),
        mean_conf=("conf_%", "mean"),
        mean_mask_cov=("mask_cov_%", "mean"),
    ).reset_index().sort_values("time_sec").reset_index(drop=True)

    frame["dt"] = frame["time_sec"].diff().fillna(0)
    frame["d_area"] = frame["total_canopy_area"].diff().fillna(0)
    frame["growth_velocity"] = np.where(frame["dt"] > 0, frame["d_area"] / frame["dt"], 0.0)
    frame["canopy_area_smooth_3"] = frame["total_canopy_area"].rolling(3, min_periods=1).mean()
    frame["canopy_area_std_3"] = frame["total_canopy_area"].rolling(3, min_periods=1).std().fillna(0)
    frame["biomass_integral"] = frame["total_canopy_area"].cumsum()
    frame["canopy_area_lag_1"] = frame["total_canopy_area"].shift(1).bfill()
    frame["leaf_count_lag_1"] = frame["leaf_count"].shift(1).bfill()
    return frame


class CropGrowthEnsembleRegressor:
    """Blend of XGBoost, RandomForest and a robust Huber linear model."""

    def __init__(self):
        import xgboost as xgb
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import HuberRegressor
        self.xgb = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.04,
                                    subsample=0.8, colsample_bytree=0.8, random_state=42)
        self.rf = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
        self.huber = HuberRegressor(max_iter=1000)

    def fit(self, X, y):
        self.xgb.fit(X, y)
        self.rf.fit(X, y)
        self.huber.fit(X, y)
        return self

    def predict(self, X):
        return 0.5 * self.xgb.predict(X) + 0.3 * self.rf.predict(X) + 0.2 * self.huber.predict(X)


def _feature_columns(frame: pd.DataFrame, target_cols: List[str]) -> List[str]:
    exclude = {"frame", "time_sec", *target_cols}
    return [c for c in frame.columns if c not in exclude]


def run(metadata_csv: str, out_dir: str = "outputs", labels_csv: Optional[str] = None) -> pd.DataFrame:
    """Engineer features and, if a real target exists, train + evaluate the regressor."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metadata_csv)
    frame = engineer_features(df)

    target_cols = [c for c in frame.columns if "target" in c.lower()]
    if labels_csv:
        labels = pd.read_csv(labels_csv)
        frame = frame.merge(labels, on="frame", how="left")
        target_cols = [c for c in labels.columns if c != "frame"]

    frame.to_csv(out / "growth_features.csv", index=False)
    logger.info("Saved per-frame growth features -> %s", out / "growth_features.csv")

    if not target_cols:
        logger.warning("No real growth/yield target provided; emitted growth features only. "
                       "Supply --labels CSV (columns: frame, <target>) to train the regressor.")
        return frame

    feature_cols = _feature_columns(frame, target_cols)
    X = frame[feature_cols]
    for target in target_cols:
        y = frame[target]
        keep = y.notna()
        Xc, yc = X[keep], y[keep]
        if len(Xc) < 12:
            logger.warning("Target '%s': only %d labelled frames (<12); skipping.", target, len(Xc))
            continue
        X_tr, X_te, y_tr, y_te = train_test_split(Xc, yc, test_size=0.2, shuffle=False)
        model = CropGrowthEnsembleRegressor().fit(X_tr, y_tr)
        pred = model.predict(X_te)
        logger.info("Target '%s': MAE=%.4f RMSE=%.4f R2=%.4f", target,
                    mean_absolute_error(y_te, pred),
                    np.sqrt(mean_squared_error(y_te, pred)),
                    r2_score(y_te, pred))
        pd.DataFrame({"GroundTruth": y_te, "Prediction": pred,
                      "Residual": y_te - pred}).to_csv(out / f"evaluation_{target}.csv", index=False)
    return frame
