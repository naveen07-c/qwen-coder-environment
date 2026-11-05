"""Tests for the Sentimind FastAPI endpoints."""

from __future__ import annotations

import os

from fastapi.testclient import TestClient

from sentiment.api import app
from sentiment.model import DEFAULT_MODEL_PATH, save_model, train

# Ensure a trained model exists for API tests.
if not os.path.exists(DEFAULT_MODEL_PATH):
    pipeline, _ = train(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                     "data", "sentiments.csv"))
    save_model(pipeline, DEFAULT_MODEL_PATH)

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_analyze_endpoint():
    resp = client.post("/analyze", json={"text": "I love it, simply wonderful!"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["label"] in {"positive", "negative", "neutral"}
    assert 0 < body["confidence"] <= 1


def test_analyze_validation_error():
    resp = client.post("/analyze", json={"text": ""})
    assert resp.status_code == 422


def test_batch_endpoint():
    resp = client.post("/analyze/batch",
                       json={"texts": ["Great job!", "Awful service."]})
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 2
    assert results[0]["label"] == "positive"
    assert results[1]["label"] == "negative"
