"""
Flask web application for fake news detection.
Usage: python app.py
"""

import os
import re
import json
import pickle
import numpy as np
import nltk
import warnings
from flask import Flask, render_template, request, jsonify, send_from_directory

warnings.filterwarnings('ignore')

app = Flask(__name__)

# ── NLTK setup ────────────────────────────────────────────────────────────────
def _setup_nltk():
    for res in ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'punkt_tab']:
        try:
            nltk.download(res, quiet=True)
        except Exception:
            pass

_setup_nltk()

# ── Load artefacts ────────────────────────────────────────────────────────────
MODEL_PATH      = os.path.join('models', 'fake_news_model.pkl')
VECTORIZER_PATH = os.path.join('models', 'tfidf_vectorizer.pkl')
METADATA_PATH   = os.path.join('models', 'model_metadata.json')

model      = None
vectorizer = None
metadata   = {}

def load_artefacts():
    global model, vectorizer, metadata
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH) as f:
                metadata = json.load(f)
        return True
    return False

model_loaded = load_artefacts()

# ── Text preprocessing (mirrors train_model.py) ───────────────────────────────
def preprocess_text(text: str) -> str:
    if not text:
        return ''
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    try:
        from nltk.tokenize import word_tokenize
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()

    try:
        from nltk.corpus import stopwords as sw
        stop = set(sw.words('english'))
    except Exception:
        stop = set()

    tokens = [t for t in tokens if t not in stop and len(t) > 2]

    try:
        from nltk.stem import WordNetLemmatizer
        lem = WordNetLemmatizer()
        tokens = [lem.lemmatize(t) for t in tokens]
    except Exception:
        pass

    return ' '.join(tokens)


# ── Word importance (LR / linear classifiers) ─────────────────────────────────
def get_word_importance(processed: str, n: int = 8):
    if not hasattr(model, 'coef_') or not processed:
        return []
    try:
        features   = vectorizer.get_feature_names_out()
        tfidf_vec  = vectorizer.transform([processed])
        nz_idx     = tfidf_vec.nonzero()[1]
        if len(nz_idx) == 0:
            return []
        tfidf_vals = np.asarray(tfidf_vec[0, nz_idx].todense()).flatten()
        coefs      = np.asarray(model.coef_[0][nz_idx]).flatten()
        scores     = tfidf_vals * coefs
        top_idx    = np.argsort(np.abs(scores))[::-1][:n]
        return [
            {
                'word':      features[nz_idx[i]],
                'score':     float(round(abs(scores[i]), 4)),
                'influence': 'fake' if scores[i] > 0 else 'real',
            }
            for i in top_idx
        ]
    except Exception:
        return []


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html',
                           model_loaded=model_loaded,
                           metadata=metadata)


@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return jsonify({'error': 'Model not loaded. Run python train_model.py first.'}), 503

    payload = request.get_json(silent=True) or {}
    text = payload.get('text', '').strip()

    if not text:
        return jsonify({'error': 'Please enter some text to analyse.'}), 400
    if len(text.split()) < 3:
        return jsonify({'error': 'Text is too short — please provide at least a few words.'}), 400

    try:
        processed = preprocess_text(text)
        if not processed:
            return jsonify({'error': 'Text could not be processed. Try different content.'}), 400

        tfidf_vec  = vectorizer.transform([processed])
        prediction = int(model.predict(tfidf_vec)[0])

        if hasattr(model, 'predict_proba'):
            proba      = model.predict_proba(tfidf_vec)[0]
            fake_prob  = float(proba[1] * 100)
            real_prob  = float(proba[0] * 100)
            confidence = float(max(proba) * 100)
        elif hasattr(model, 'decision_function'):
            dec       = float(model.decision_function(tfidf_vec)[0])
            # Map decision score to 50-100% confidence range
            raw_conf  = abs(dec) / (1.0 + abs(dec))   # 0→1
            confidence = 50.0 + raw_conf * 50.0
            if prediction == 1:
                fake_prob, real_prob = confidence, 100.0 - confidence
            else:
                real_prob, fake_prob = confidence, 100.0 - confidence
        else:
            confidence = fake_prob = real_prob = 50.0

        label        = 'FAKE' if prediction == 1 else 'REAL'
        key_words    = get_word_importance(processed)

        return jsonify({
            'prediction':       label,
            'confidence':       round(confidence, 1),
            'fake_probability': round(fake_prob, 1),
            'real_probability': round(real_prob, 1),
            'word_count':       len(text.split()),
            'char_count':       len(text),
            'key_words':        key_words,
            'model_name':       metadata.get('best_model', 'Unknown'),
            'model_accuracy':   metadata.get('accuracy', None),
        })

    except Exception as exc:
        return jsonify({'error': f'Prediction failed: {exc}'}), 500


@app.route('/stats')
def stats():
    return jsonify(metadata)


@app.route('/static/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(os.path.join('static', 'images'), filename)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', '1') == '1'
    print(f"\n  Fake News Detector running at http://localhost:{port}")
    print(f"  Model loaded : {model_loaded}")
    if not model_loaded:
        print("  ⚠  Run `python train_model.py` first to train the model.\n")
    app.run(debug=debug, host='0.0.0.0', port=port)
