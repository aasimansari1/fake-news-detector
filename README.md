<div align="center">

# 🛡️ FakeShield AI — Fake News Detector

### Paste any headline or article. Get an instant REAL / FAKE verdict with confidence score and linguistic evidence.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![NLTK](https://img.shields.io/badge/NLTK-3.8-154f3c?style=for-the-badge)](https://nltk.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

</div>

---

> **Misinformation spreads 6x faster than the truth.**
> FakeShield uses NLP and machine learning to classify news articles in milliseconds — with a confidence score, fake/real probability breakdown, and the exact words that triggered the verdict.

---

## ✨ Features

<table>
<tr>
<td width="50%">

**🧠 Multi-Model ML**
- 4 models compared: Passive Aggressive, Naive Bayes, Logistic Regression, Random Forest
- Best model auto-selected at training time
- ~91% accuracy on Kaggle Fake News dataset

</td>
<td width="50%">

**🔍 Explainable AI**
- Confidence score + fake/real probability
- Key words that influenced the verdict
- Per-word influence direction (fake vs real)

</td>
</tr>
<tr>
<td width="50%">

**📊 Training Visualisations**
- Accuracy comparison across all 4 models
- Confusion matrix per model
- ROC curves
- Top 20 most predictive words

</td>
<td width="50%">

**🌐 REST API**
- `POST /predict` endpoint
- JSON in, JSON out
- Drop-in for any app or browser extension

</td>
</tr>
</table>

---

## 🚀 Quick Start

```bash
git clone https://github.com/aasimansari1/fake-news-detector.git
cd fake-news-detector

pip install -r requirements.txt

# Generate sample dataset + download NLTK data
python dataset/create_dataset.py

# Train all 4 models (best one saved automatically)
python train_model.py

# Launch the web app
python app.py
# → http://localhost:5000
```

> **Want better accuracy?** Replace `dataset/sample_data.csv` with the [Kaggle Fake News dataset](https://www.kaggle.com/c/fake-news) (~44K articles). Any CSV with `text` and `label` columns (REAL/FAKE or 0/1) works.

---

## 🏗️ How It Works

```
  📰 Article / Headline
          │
          ▼
  ┌───────────────────┐
  │   NLP Pipeline    │
  │  lowercase →      │
  │  punctuation →    │
  │  tokenise →       │
  │  stopwords →      │
  │  lemmatise        │
  └────────┬──────────┘
           │
           ▼
  ┌───────────────────┐
  │  TF-IDF (bigrams) │
  │  Vectoriser       │
  └────────┬──────────┘
           │
           ▼
  ┌───────────────────┐
  │  Best ML Model    │  ← Passive Aggressive (~91%)
  └────────┬──────────┘
           │
           ▼
  ✅ REAL (confidence: 94%)
  ❌ FAKE (confidence: 82%)
  + key words + probabilities
```

---

## 📡 API

```bash
POST /predict
Content-Type: application/json

{"text": "Scientists discover that vaccines cause autism, government hiding truth"}
```

```json
{
  "prediction": "FAKE",
  "confidence": 89.4,
  "fake_probability": 89.4,
  "real_probability": 10.6,
  "key_words": [
    {"word": "hiding", "score": 0.21, "influence": "fake"},
    {"word": "truth", "score": 0.18, "influence": "fake"}
  ],
  "model_name": "Passive Aggressive",
  "model_accuracy": 0.907
}
```

---

## 📈 Model Accuracy

| Model | Accuracy | Notes |
|---|:---:|---|
| **Passive Aggressive** | **~91%** | Best overall — auto-selected |
| Naive Bayes | ~88% | Fastest inference |
| Logistic Regression | ~86% | Most interpretable |
| Random Forest | ~84% | Most robust to noise |

*Accuracy on sample dataset. Use a larger real-world dataset for production-grade results.*

---

## 🗂️ Project Structure

```
fake-news-detector/
├── app.py                        # Flask web app + /predict API
├── train_model.py                # Training pipeline, model comparison
├── requirements.txt
│
├── src/                          # Reusable data science module
│   ├── preprocessing.py          # Text cleaning, label normalisation, dataset loader
│   ├── features.py               # TF-IDF vectoriser builder
│   └── models.py                 # Train all classifiers, pick best, ROC data
│
├── notebooks/                    # Jupyter data science pipeline
│   ├── 01_EDA.ipynb              # Exploratory data analysis
│   ├── 02_Feature_Engineering.ipynb  # Preprocessing + TF-IDF
│   ├── 03_Model_Training.ipynb   # Train & compare 4 models
│   └── 04_Model_Evaluation.ipynb # Confusion matrix, ROC, feature importance, CV
│
├── data/
│   ├── raw/                      # Source datasets (CSV)
│   └── processed/                # Train/test splits + vectoriser (generated)
│
├── models/                       # Saved model + vectoriser (generated)
├── reports/figures/              # All visualisation outputs (generated)
│
├── dataset/
│   └── create_dataset.py         # Sample dataset generator
├── static/
│   ├── css/style.css
│   ├── js/main.js
│   └── images/
└── templates/
    └── index.html
```

### Running the Notebooks

```bash
pip install -r requirements.txt
jupyter notebook
```

Run in order: `01_EDA` → `02_Feature_Engineering` → `03_Model_Training` → `04_Model_Evaluation`

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask, Gunicorn |
| ML / NLP | scikit-learn, NLTK, pandas, numpy |
| Visualisation | matplotlib, seaborn |
| Frontend | Vanilla HTML/CSS/JS, dark mode |
| Server | Nginx + systemd |

---

## 🤝 Contributing

Ideas for contributions:
- 🌐 Browser extension that checks articles in-page
- 🤗 Fine-tune a BERT/RoBERTa model for better accuracy
- 📱 Mobile-friendly UI improvements
- 🗃️ Support for more languages

```bash
git checkout -b feature/your-feature
git commit -m 'Add your feature'
git push origin feature/your-feature
# Open a Pull Request
```

---

## 📄 License

MIT © [Mohd Aasim Ansari](https://github.com/aasimansari1)

---

<div align="center">

**Fighting misinformation one article at a time. If this helped, please ⭐ star the repo!**

</div>
