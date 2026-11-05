"""Tests for the Sentimind sentiment toolkit."""

from __future__ import annotations

import os

import pytest

from sentiment.model import (
    build_pipeline,
    load_dataset,
    predict_many,
    predict_one,
    train,
)

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "data", "sentiments.csv")


@pytest.fixture(scope="module")
def fitted_pipeline():
    df = load_dataset(DATA_PATH)
    pipeline = build_pipeline()
    pipeline.fit(df["text"], df["label"])
    return pipeline


def test_load_dataset_columns():
    df = load_dataset(DATA_PATH)
    assert {"text", "label"} <= set(df.columns)
    assert len(df) >= 30
    assert set(df["label"].unique()) == {"positive", "negative", "neutral"}


def test_predict_positive(fitted_pipeline):
    pred = predict_one(fitted_pipeline,
                       "I love this amazing product, absolutely fantastic!")
    assert pred.label == "positive"
    assert 0 < pred.confidence <= 1
    assert set(pred.scores) == {"positive", "negative", "neutral"}


def test_predict_negative(fitted_pipeline):
    pred = predict_one(fitted_pipeline,
                       "Terrible awful junk, worst purchase ever, I hate it.")
    assert pred.label == "negative"


def test_predict_batch(fitted_pipeline):
    texts = ["Wonderful and delightful!", "Horrible and disgusting."]
    preds = predict_many(fitted_pipeline, texts)
    assert len(preds) == 2
    assert preds[0].label == "positive"
    assert preds[1].label == "negative"


def test_empty_batch(fitted_pipeline):
    assert predict_many(fitted_pipeline, []) == []


def test_train_returns_metrics():
    pipeline, result = train(DATA_PATH)
    assert 0.0 <= result.accuracy <= 1.0
    assert result.n_train + result.n_test == len(load_dataset(DATA_PATH))
