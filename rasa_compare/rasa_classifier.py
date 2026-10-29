"""Load the best saved model:  from rasa_classifier import predict, LABELS ; predict(["text", ...]) -> (n, 10)."""
from pathlib import Path

import joblib
import numpy as np

LABELS = ["Shringara", "Hasya", "Karuna", "Raudra", "Veera",
          "Bhayanaka", "Bibhatsa", "Adbhuta", "Shanta", "None"]
_DIR = Path(__file__).resolve().parent / "models" / "best"
_cache = {}


def _load():
    if not _cache:
        meta = joblib.load(_DIR / "model.joblib")
        assert meta["labels"] == LABELS
        if meta["kind"] == "logreg":
            from sentence_transformers import SentenceTransformer
            _cache.update(kind="logreg", clf=meta["clf"], enc=SentenceTransformer(meta["encoder"]))
        else:
            from setfit import SetFitModel
            _cache.update(kind="setfit", model=SetFitModel.from_pretrained(str(_DIR / meta["path"])))
    return _cache


def predict(texts):
    m = _load()
    texts = list(texts)
    out = np.zeros((len(texts), len(LABELS)))
    if m["kind"] == "logreg":
        X = m["enc"].encode(texts, normalize_embeddings=True)
        out[:, m["clf"].classes_] = m["clf"].predict_proba(X)
    else:
        out[:] = np.asarray(m["model"].predict_proba(texts))
    return out
