import re
import pandas as pd
import nltk

def _ensure_nltk():
    for res in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']:
        try:
            nltk.download(res, quiet=True)
        except Exception:
            pass

_ensure_nltk()


def preprocess_text(text: str) -> str:
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
        stop = set(stopwords.words('english'))
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


def normalise_labels(series: pd.Series) -> pd.Series:
    """Map any label format → binary int (0=REAL, 1=FAKE)."""
    s = series.astype(str).str.strip().str.upper()
    mapping = {'REAL': 0, 'TRUE': 0, '0': 0, 'FAKE': 1, 'FALSE': 1, '1': 1}
    return pd.to_numeric(s.map(mapping), errors='coerce').astype('Int64')


def load_dataset(data_dir: str = 'data/raw') -> pd.DataFrame:
    """Load dataset from data/raw, normalise to (text, label) columns."""
    import os
    priority = ['fake_or_real_news.csv', 'train.csv', 'news.csv', 'sample_data.csv']
    for fname in priority:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if 'text' not in df.columns:
            for col in ['title', 'body', 'content']:
                if col in df.columns:
                    df['text'] = df[col].fillna('')
                    break
        if 'label' not in df.columns:
            for col in ['Label', 'category', 'class', 'fake']:
                if col in df.columns:
                    df = df.rename(columns={col: 'label'})
                    break
        if 'text' in df.columns and 'label' in df.columns:
            return df[['text', 'label']].copy()
    raise FileNotFoundError(f"No usable dataset found in {data_dir}/")
