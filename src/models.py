from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_curve, auc
)
import numpy as np


CLASSIFIERS = {
    'Logistic Regression': lambda: LogisticRegression(
        max_iter=1000, C=1.0, solver='lbfgs', random_state=42
    ),
    'Naive Bayes': lambda: MultinomialNB(alpha=0.1),
    'Random Forest': lambda: RandomForestClassifier(
        n_estimators=200, max_depth=20, random_state=42, n_jobs=-1
    ),
    'Passive Aggressive': lambda: PassiveAggressiveClassifier(
        max_iter=1000, random_state=42, tol=1e-3
    ),
}


def train_all(X_tr, y_tr, X_te, y_te) -> tuple[dict, dict]:
    """Train all classifiers, return (trained_models, results)."""
    trained, results = {}, {}
    for name, factory in CLASSIFIERS.items():
        clf = factory()
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        results[name] = {
            'accuracy': float(acc),
            'report': classification_report(y_te, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_te, y_pred).tolist(),
        }
        trained[name] = clf
    return trained, results


def pick_best(trained: dict, results: dict) -> tuple[str, object]:
    """Return (best_name, best_classifier) by accuracy."""
    best_name = max(results, key=lambda m: results[m]['accuracy'])
    return best_name, trained[best_name]


def roc_data(trained: dict, vectorizer, X_te_raw, y_te) -> dict:
    """Compute ROC curve data for each model that supports scoring."""
    X_te = vectorizer.transform(X_te_raw)
    curves = {}
    for name, clf in trained.items():
        if hasattr(clf, 'predict_proba'):
            scores = clf.predict_proba(X_te)[:, 1]
        elif hasattr(clf, 'decision_function'):
            scores = clf.decision_function(X_te)
        else:
            continue
        fpr, tpr, _ = roc_curve(y_te, scores)
        curves[name] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': float(auc(fpr, tpr))}
    return curves
