from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp


def build_vectorizer(
    max_features: int = 15000,
    ngram_range: tuple = (1, 2),
    min_df: int = 1,
    max_df: float = 0.95,
) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
    )


def vectorize(vectorizer: TfidfVectorizer, X_train, X_test):
    """Fit on train, transform both splits. Returns (X_tr, X_te)."""
    X_tr = vectorizer.fit_transform(X_train)
    X_te = vectorizer.transform(X_test)
    return X_tr, X_te
