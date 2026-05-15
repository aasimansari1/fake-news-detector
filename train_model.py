"""
Train fake news detection models and save the best one.
Usage: python train_model.py
"""

import pandas as pd
import numpy as np
import nltk
import pickle
import os
import re
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_curve, auc
)
import warnings
warnings.filterwarnings('ignore')


def setup_nltk():
    resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'punkt_tab']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass


def preprocess_text(text):
    if pd.isna(text) or str(text).strip() == '':
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
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
    except Exception:
        stop_words = set()

    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]

    try:
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    except Exception:
        pass

    return ' '.join(tokens)


def load_dataset(data_dir='dataset'):
    """Try to load dataset; fall back to sample_data.csv."""
    priority = [
        'fake_or_real_news.csv',
        'train.csv',
        'news.csv',
        'sample_data.csv',
    ]
    for fname in priority:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        print(f"Loaded: {path}  ({len(df)} rows)")

        # Normalize to (text, label) columns
        if 'text' not in df.columns:
            if 'title' in df.columns and 'body' in df.columns:
                df['text'] = df['title'].fillna('') + ' ' + df['body'].fillna('')
            elif 'title' in df.columns:
                df['text'] = df['title'].fillna('')
            elif 'content' in df.columns:
                df['text'] = df['content'].fillna('')
            else:
                print(f"  Skipping {fname}: no recognisable text column")
                continue

        if 'label' not in df.columns:
            for col in ['Label', 'category', 'class', 'fake']:
                if col in df.columns:
                    df = df.rename(columns={col: 'label'})
                    break
            else:
                print(f"  Skipping {fname}: no recognisable label column")
                continue

        return df[['text', 'label']].copy()

    raise FileNotFoundError(
        "No dataset found in dataset/. "
        "Run `python dataset/create_dataset.py` to generate the sample dataset."
    )


def normalise_labels(series):
    """Convert any label format to binary int (0=REAL, 1=FAKE)."""
    s = series.astype(str).str.strip().str.upper()
    mapping = {'REAL': 0, 'TRUE': 0, '0': 0, 'FAKE': 1, 'FALSE': 1, '1': 1}
    s = s.map(mapping)
    return pd.to_numeric(s, errors='coerce').astype('Int64')


def train_all_models(X_tr_tfidf, y_tr, X_te_tfidf, y_te):
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, C=1.0, solver='lbfgs', random_state=42
        ),
        'Naive Bayes': MultinomialNB(alpha=0.1),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=20, random_state=42, n_jobs=-1
        ),
        'Passive Aggressive': PassiveAggressiveClassifier(
            max_iter=1000, random_state=42, tol=1e-3
        ),
    }

    results = {}
    trained = {}

    for name, clf in models.items():
        print(f"\n  Training {name}...")
        clf.fit(X_tr_tfidf, y_tr)
        y_pred = clf.predict(X_te_tfidf)
        acc = accuracy_score(y_te, y_pred)
        report = classification_report(y_te, y_pred, output_dict=True)
        cm = confusion_matrix(y_te, y_pred)

        results[name] = {
            'accuracy': float(acc),
            'report': report,
            'confusion_matrix': cm.tolist(),
        }
        trained[name] = clf
        print(f"    Accuracy: {acc:.4f}")

    return trained, results


def plot_model_comparison(results, save_dir):
    names = list(results.keys())
    accs = [results[m]['accuracy'] * 100 for m in names]
    colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe']

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#f8f9fa')
    bars = ax.bar(names, accs, color=colors, edgecolor='white', linewidth=1.5, width=0.5)
    ax.set_ylim(0, 110)
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Model Accuracy Comparison', fontsize=15, fontweight='bold', pad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2., bar.get_height() + 1,
            f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11
        )
    plt.xticks(rotation=15, ha='right', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(results, best_name, save_dir):
    cm = np.array(results[best_name]['confusion_matrix'])
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['REAL', 'FAKE'], yticklabels=['REAL', 'FAKE'],
        linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8},
        annot_kws={'size': 14, 'weight': 'bold'}
    )
    ax.set_title(f'Confusion Matrix — {best_name}', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_curves(trained_models, vectorizer, X_te, y_te, save_dir):
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('#ffffff')
    colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe']
    X_te_tfidf = vectorizer.transform(X_te)

    for (name, clf), color in zip(trained_models.items(), colors):
        if hasattr(clf, 'predict_proba'):
            scores = clf.predict_proba(X_te_tfidf)[:, 1]
        elif hasattr(clf, 'decision_function'):
            scores = clf.decision_function(X_te_tfidf)
        else:
            continue
        fpr, tpr, _ = roc_curve(y_te, scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{name} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curves — All Models', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_top_features(clf, vectorizer, save_dir, n=20):
    """Plot top words most predictive of FAKE and REAL news."""
    if not hasattr(clf, 'coef_'):
        return
    coef = clf.coef_[0]
    features = vectorizer.get_feature_names_out()

    top_fake_idx = np.argsort(coef)[-n:][::-1]
    top_real_idx = np.argsort(coef)[:n]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor('#ffffff')
    fig.suptitle('Top Predictive Words by Class', fontsize=15, fontweight='bold', y=1.02)

    for ax, indices, title, color in zip(
        axes,
        [top_fake_idx, top_real_idx],
        ['Top FAKE Indicators', 'Top REAL Indicators'],
        ['#e17055', '#00b894']
    ):
        words = [features[i] for i in indices]
        scores = [abs(coef[i]) for i in indices]
        ax.set_facecolor('#f8f9fa')
        bars = ax.barh(words[::-1], scores[::-1], color=color, alpha=0.85)
        ax.set_title(title, fontsize=13, fontweight='bold', color=color)
        ax.set_xlabel('Coefficient Magnitude', fontsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'top_features.png'), dpi=150, bbox_inches='tight')
    plt.close()


def main():
    setup_nltk()

    print("=" * 60)
    print("  Fake News Detection — Model Training")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────
    df = load_dataset()
    df['label'] = normalise_labels(df['label'])
    df = df.dropna(subset=['text', 'label'])
    df = df[df['text'].str.strip().ne('')]
    print(f"\nClean samples : {len(df)}")
    print(f"Label counts  :\n{df['label'].value_counts().rename({0: 'REAL', 1: 'FAKE'})}")

    # ── Preprocess ────────────────────────────────────────────
    print("\nPreprocessing text …")
    df['proc'] = df['text'].apply(preprocess_text)
    df = df[df['proc'].str.len() > 0]

    X_train, X_test, y_train, y_test = train_test_split(
        df['proc'], df['label'].astype(int),
        test_size=0.2, random_state=42, stratify=df['label']
    )
    print(f"Train : {len(X_train)}  |  Test : {len(X_test)}")

    # ── TF-IDF ────────────────────────────────────────────────
    print("\nFitting TF-IDF vectorizer …")
    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )
    X_tr_tfidf = vectorizer.fit_transform(X_train)
    X_te_tfidf = vectorizer.transform(X_test)
    print(f"Vocabulary size : {len(vectorizer.get_feature_names_out()):,}")

    # ── Train ─────────────────────────────────────────────────
    print("\nTraining models …")
    trained_models, results = train_all_models(X_tr_tfidf, y_train, X_te_tfidf, y_test)

    # ── Pick best ─────────────────────────────────────────────
    best_name = max(results, key=lambda m: results[m]['accuracy'])
    best_clf = trained_models[best_name]
    best_acc = results[best_name]['accuracy']

    print("\n" + "=" * 60)
    print(f"  Best model : {best_name}  (accuracy={best_acc:.4f})")
    print("=" * 60)
    y_pred = best_clf.predict(X_te_tfidf)
    print(classification_report(y_test, y_pred, target_names=['REAL', 'FAKE']))

    # ── Visualisations ────────────────────────────────────────
    img_dir = os.path.join('static', 'images')
    os.makedirs(img_dir, exist_ok=True)

    print("Generating visualisations …")
    plot_model_comparison(results, img_dir)
    plot_confusion_matrix(results, best_name, img_dir)
    plot_roc_curves(trained_models, vectorizer, X_test, y_test, img_dir)
    plot_top_features(best_clf, vectorizer, img_dir)
    print(f"  Saved to {img_dir}/")

    # ── Save artefacts ────────────────────────────────────────
    os.makedirs('models', exist_ok=True)

    with open('models/fake_news_model.pkl', 'wb') as f:
        pickle.dump(best_clf, f, protocol=4)
    with open('models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f, protocol=4)

    metadata = {
        'best_model': best_name,
        'accuracy': round(best_acc, 4),
        'all_models': {
            k: {'accuracy': round(v['accuracy'], 4)}
            for k, v in results.items()
        },
        'vocab_size': int(len(vectorizer.get_feature_names_out())),
        'train_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'label_counts': df['label'].value_counts().rename({0: 'REAL', 1: 'FAKE'}).to_dict(),
    }
    with open('models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\nAll artefacts saved to models/")
    print("Run  python app.py  to launch the web application.\n")


if __name__ == '__main__':
    main()
