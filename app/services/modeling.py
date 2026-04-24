from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import RLock

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.config import get_settings
from app.services.historical_data import (
    FEATURE_ORDER,
    FeatureRepository,
    TrainingSample,
    build_training_samples,
    download_ipl_dataset,
    load_ipl_matches,
)

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional at runtime
    XGBClassifier = None


_bundle_cache: ModelBundle | None = None
_bundle_lock = RLock()


@dataclass
class ModelBundle:
    model_name: str
    trained_at: str
    feature_order: list[str]
    estimator: object
    metrics: dict[str, dict[str, float]]
    repository_state: dict
    sample_count: int
    _repository_cache: FeatureRepository | None = field(default=None, init=False, repr=False)

    @property
    def repository(self) -> FeatureRepository:
        if self._repository_cache is None:
            self._repository_cache = FeatureRepository.from_state(self.repository_state)
        return self._repository_cache

    def predict_proba(self, feature_values: dict[str, float]) -> float:
        vector = np.array([[feature_values[name] for name in self.feature_order]], dtype=float)
        probability = float(self.estimator.predict_proba(vector)[0, 1])
        return max(0.02, min(0.98, probability))


def _samples_to_arrays(samples: list[TrainingSample]) -> tuple[np.ndarray, np.ndarray]:
    x = np.array(
        [[sample.feature_values[name] for name in FEATURE_ORDER] for sample in samples],
        dtype=float,
    )
    y = np.array([sample.label for sample in samples], dtype=int)
    return x, y


def _evaluate_model(model, x_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    probabilities = np.clip(model.predict_proba(x_test)[:, 1], 1e-6, 1 - 1e-6)
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "log_loss": float(log_loss(y_test, probabilities)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "brier_score": float(brier_score_loss(y_test, probabilities)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }


def train_model_bundle(refresh_data: bool = False) -> ModelBundle:
    global _bundle_cache
    with _bundle_lock:
        settings = get_settings()
        download_ipl_dataset(force=refresh_data)
        matches = load_ipl_matches(settings.ipl_data_zip)
        samples, repository = build_training_samples(matches)
        if len(samples) < 200:
            raise RuntimeError("Not enough IPL training samples were built.")

        split_index = max(100, int(len(samples) * 0.8))
        train_samples = samples[:split_index]
        test_samples = samples[split_index:]
        x_train, y_train = _samples_to_arrays(train_samples)
        x_test, y_test = _samples_to_arrays(test_samples)

        candidate_models: dict[str, object] = {
            "logistic_regression": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=2000)),
                ]
            )
        }

        if XGBClassifier is not None:
            candidate_models["xgboost"] = XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
            )

        trained_models: dict[str, object] = {}
        metrics: dict[str, dict[str, float]] = {}
        for model_name, model in candidate_models.items():
            model.fit(x_train, y_train)
            trained_models[model_name] = model
            metrics[model_name] = _evaluate_model(model, x_test, y_test)

        best_model_name = min(metrics, key=lambda name: metrics[name]["log_loss"])
        bundle = ModelBundle(
            model_name=best_model_name,
            trained_at=datetime.utcnow().isoformat(),
            feature_order=FEATURE_ORDER,
            estimator=trained_models[best_model_name],
            metrics=metrics,
            repository_state=repository.to_state(),
            sample_count=len(samples),
        )

        settings.model_artifact.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, settings.model_artifact)
        _bundle_cache = bundle
        return bundle


def load_model_bundle(path: Path | None = None) -> ModelBundle | None:
    global _bundle_cache
    settings = get_settings()
    target = path or settings.model_artifact
    if not target.exists():
        return None
    with _bundle_lock:
        if _bundle_cache is None:
            _bundle_cache = joblib.load(target)
        return _bundle_cache


def get_or_train_model_bundle(
    auto_train: bool = False,
    refresh_data: bool = False,
) -> ModelBundle | None:
    bundle = load_model_bundle()
    if bundle is not None:
        return bundle
    if not auto_train:
        return None
    with _bundle_lock:
        if _bundle_cache is not None:
            return _bundle_cache
        return train_model_bundle(refresh_data=refresh_data)
