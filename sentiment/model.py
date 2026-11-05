"""Core sentiment model: a TF-IDF + Logistic Regression scikit-learn pipeline.

Provides helpers to train, save, load and predict with the model.
The trained artifact is a joblib-serialized Pipeline stored at
``models/sentiment_pipeline.joblib`` by default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "sentiment_pipeline.joblib",
)

LABELS = ["negative", "neutral", "positive"]


def build_pipeline() -> Pipeline:
    """Return an untrained TF-IDF + LogisticRegression pipeline."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1,
                                      sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, C=5.0,
                                       class_weight="balanced")),
        ]
    )


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load a labelled dataset with ``text`` and ``label`` columns."""
    df = pd.read_csv(csv_path)
    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    return df.dropna(subset=["text", "label"]).reset_index(drop=True)


@dataclass
class TrainResult:
    accuracy: float
    report: str
    n_train: int
    n_test: int
    labels: list[str] = field(default_factory=lambda: list(LABELS))


def train(csv_path: str, test_size: float = 0.2,
          random_state: int = 42) -> tuple[Pipeline, TrainResult]:
    """Train the pipeline on a CSV dataset and return (pipeline, metrics)."""
    df = load_dataset(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=test_size,
        random_state=random_state, stratify=df["label"],
    )
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    result = TrainResult(
        accuracy=accuracy_score(y_test, preds),
        report=classification_report(y_test, preds, digits=3),
        n_train=len(X_train),
        n_test=len(X_test),
    )
    return pipeline, result


def save_model(pipeline: Pipeline, path: str = DEFAULT_MODEL_PATH) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_model(path: str = DEFAULT_MODEL_PATH) -> Pipeline:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No trained model found at {path}. Run `python -m sentiment.cli train` first."
        )
    return joblib.load(path)


@dataclass
class Prediction:
    label: str
    confidence: float
    scores: dict[str, float]


def predict_one(pipeline: Pipeline, text: str) -> Prediction:
    """Predict sentiment for a single piece of text."""
    probs = pipeline.predict_proba([text])[0]
    classes = list(pipeline.classes_)
    scores = {c: round(float(p), 4) for c, p in zip(classes, probs)}
    best_idx = int(probs.argmax())
    return Prediction(
        label=classes[best_idx],
        confidence=round(float(probs[best_idx]), 4),
        scores=scores,
    )


def predict_many(pipeline: Pipeline, texts: Iterable[str]) -> list[Prediction]:
    texts = list(texts)
    if not texts:
        return []
    probs = pipeline.predict_proba(texts)
    classes = list(pipeline.classes_)
    out = []
    for row in probs:
        best_idx = int(row.argmax())
        out.append(Prediction(
            label=classes[best_idx],
            confidence=round(float(row[best_idx]), 4),
            scores={c: round(float(p), 4) for c, p in zip(classes, row)},
        ))
    return out
