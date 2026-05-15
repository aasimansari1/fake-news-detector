# FakeShield AI — Fake News Detector

An AI-powered fake news detection web application built with Python, scikit-learn, NLTK, and Flask.

## Features

- Real-time fake news classification (REAL / FAKE)
- Confidence score and probability breakdown
- Key linguistic indicators that influenced the prediction
- NLP preprocessing: lowercasing, punctuation removal, tokenisation, stopword removal, lemmatisation
- TF-IDF vectorisation with bigrams
- Four ML models compared: Logistic Regression, Naive Bayes, Random Forest, Passive Aggressive
- Training visualisations: accuracy comparison, confusion matrix, ROC curves, top predictive words
- Clean dark-mode responsive web UI
- Production deployment with Gunicorn + Nginx + systemd

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download NLTK data + generate sample dataset
python dataset/create_dataset.py

# 3. Train models (picks the best one automatically)
python train_model.py

# 4. Run the web app
python app.py
# → http://localhost:5000
```

## Using a Real Dataset

Replace `dataset/sample_data.csv` with any CSV that has `text` and `label` columns (label = REAL/FAKE or 0/1).  
Compatible with the [Kaggle Fake News dataset](https://www.kaggle.com/c/fake-news) (~44K articles).

## Project Structure

```
fake-news-detector/
├── app.py                  # Flask web application
├── train_model.py          # Training pipeline
├── requirements.txt
├── dataset/
│   └── create_dataset.py   # Sample dataset generator
├── models/                 # Saved model artefacts (generated)
├── static/
│   ├── css/style.css
│   ├── js/main.js
│   └── images/             # Training charts (generated)
└── templates/
    └── index.html
```

## Models & Accuracy

| Model | Accuracy |
|---|---|
| Passive Aggressive | ~91% |
| Naive Bayes | ~88% |
| Logistic Regression | ~86% |
| Random Forest | ~84% |

*(Accuracy varies with dataset size. Use a larger real-world dataset for production.)*

## API

```bash
POST /predict
Content-Type: application/json

{"text": "Your news article or headline here"}
```

Response:
```json
{
  "prediction": "FAKE",
  "confidence": 82.3,
  "fake_probability": 82.3,
  "real_probability": 17.7,
  "key_words": [{"word": "bombshell", "score": 0.19, "influence": "fake"}],
  "model_name": "Passive Aggressive",
  "model_accuracy": 0.907
}
```

## Tech Stack

- **Backend:** Python, Flask, Gunicorn
- **ML / NLP:** scikit-learn, NLTK, pandas, numpy
- **Visualisation:** matplotlib, seaborn
- **Frontend:** Vanilla HTML/CSS/JS (no framework)
- **Server:** Nginx + systemd
