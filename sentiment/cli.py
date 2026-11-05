"""Command-line interface for Sentimind.

Usage examples:
    python -m sentiment.cli train --data data/sentiments.csv
    python -m sentiment.cli predict "I hate waiting in long lines"
    python -m sentiment.cli serve --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import argparse
import json
import sys

from .model import (
    DEFAULT_MODEL_PATH,
    load_model,
    predict_many,
    predict_one,
    save_model,
    train,
)


def cmd_train(args: argparse.Namespace) -> int:
    pipeline, result = train(args.data, test_size=args.test_size)
    path = save_model(pipeline, args.model)
    print(f"Trained on {result.n_train} samples, tested on {result.n_test}.")
    print(f"Test accuracy: {result.accuracy:.3f}")
    print(result.report)
    print(f"Model saved to {path}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    pipeline = load_model(args.model)
    texts = args.texts if args.texts else [sys.stdin.read()]
    preds = predict_many(pipeline, texts)
    if args.json:
        out = [{"text": t, **p.__dict__} for t, p in zip(texts, preds)]
        print(json.dumps(out, indent=2))
    else:
        for t, p in zip(texts, preds):
            print(f"[{p.label:>8} | conf={p.confidence:.2f}] {t}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("sentiment.api:app", host=args.host, port=args.port,
                reload=args.reload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sentiment",
        description="Sentimind - lightweight sentiment analysis toolkit.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train the model on a labelled CSV.")
    p_train.add_argument("--data", default="data/sentiments.csv",
                         help="Path to CSV with 'text' and 'label' columns.")
    p_train.add_argument("--model", default=DEFAULT_MODEL_PATH,
                         help="Where to save the trained pipeline.")
    p_train.add_argument("--test-size", type=float, default=0.2)
    p_train.set_defaults(func=cmd_train)

    p_pred = sub.add_parser("predict", help="Predict sentiment for text(s).")
    p_pred.add_argument("texts", nargs="*",
                        help="One or more strings; reads stdin if omitted.")
    p_pred.add_argument("--model", default=DEFAULT_MODEL_PATH)
    p_pred.add_argument("--json", action="store_true",
                        help="Emit JSON output.")
    p_pred.set_defaults(func=cmd_predict)

    p_serve = sub.add_parser("serve", help="Run the FastAPI REST server.")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")
    p_serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
