"""FastAPI REST service for Sentimind sentiment analysis."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .model import load_model, predict_many, predict_one

app = FastAPI(
    title="Sentimind API",
    description="Lightweight sentiment analysis REST API (TF-IDF + Logistic Regression).",
    version=__version__,
)


class TextIn(BaseModel):
    text: str = Field(..., min_length=1, examples=["I love this product!"])


class BatchIn(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=500)


class PredictionOut(BaseModel):
    label: str
    confidence: float
    scores: dict[str, float]


class BatchOut(BaseModel):
    results: list[PredictionOut]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.post("/analyze", response_model=PredictionOut)
def analyze(payload: TextIn) -> PredictionOut:
    try:
        pipeline = load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    pred = predict_one(pipeline, payload.text)
    return PredictionOut(label=pred.label, confidence=pred.confidence,
                         scores=pred.scores)


@app.post("/analyze/batch", response_model=BatchOut)
def analyze_batch(payload: BatchIn) -> BatchOut:
    try:
        pipeline = load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    preds = predict_many(pipeline, payload.texts)
    return BatchOut(results=[
        PredictionOut(label=p.label, confidence=p.confidence, scores=p.scores)
        for p in preds
    ])
