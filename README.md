# Sentimind

A lightweight sentiment analysis toolkit: train a TF-IDF + Logistic Regression
classifier, use it from the CLI, or serve it as a REST API with FastAPI.

## Features

- **3-class sentiment**: `positive`, `negative`, `neutral` with confidence scores
- **CLI**: train, predict (single/batch/stdin, plain or JSON output)
- **REST API**: `/health`, `/analyze`, `/analyze/batch` with interactive docs at `/docs`
- **Tests**: pytest suite covering the model and API endpoints

## Project layout

```
sentiment/        # package: model.py, api.py, cli.py
data/             # bundled sample dataset (CSV: text,label)
models/           # trained pipeline artifact (gitignored)
tests/            # pytest suite
requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. Train on the bundled sample data (or your own CSV with text,label columns)
python -m sentiment.cli train --data data/sentiments.csv

# 2. Predict from the command line
python -m sentiment.cli predict "I love this product, it's fantastic!" \
                               "Terrible experience, total waste of money."
# JSON output:
python -m sentiment.cli predict --json "The meeting is at noon"

# 3. Or run the REST API
python -m sentiment.cli serve --host 127.0.0.1 --port 8000
# then open http://127.0.0.1:8000/docs
```

### API examples

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Absolutely wonderful, I love it!"}'

curl -X POST http://127.0.0.1:8000/analyze/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great job!", "Awful service."]}'
```

## Training on your own data

Provide a CSV with two columns, `text` and `label` (labels: positive / negative / neutral):

```bash
python -m sentiment.cli train --data path/to/your_data.csv
```

Note: the bundled sample dataset is tiny (~60 rows) and exists so the project
runs end-to-end out of the box. For production accuracy, retrain on a larger
corpus (e.g., thousands of labelled reviews).

## Tests

```bash
pytest
```

## License

MIT
