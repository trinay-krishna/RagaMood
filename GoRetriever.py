# pip install transformers
import numpy as np
import pandas as pd
from transformers import pipeline

from common import RASAS
PLUTCHIK = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]

# ---------- Table 1: GoEmotions label -> Plutchik emotions (weights per label; edit freely) ----------
GO_TO_PLUTCHIK = {
    "joy": {"joy": 1}, "amusement": {"joy": 1}, "relief": {"joy": .4, "trust": .6},
    "excitement": {"joy": .6, "anticipation": .4},
    "love": {"joy": .5, "trust": .5}, "gratitude": {"joy": .5, "trust": .5},
    "admiration": {"trust": .7, "joy": .3}, "approval": {}, "caring": {"trust": .5, "sadness": .5},
    "optimism": {"anticipation": .6, "joy": .4}, "pride": {"joy": .6, "anticipation": .4},
    "desire": {"anticipation": .7, "joy": .3}, "curiosity": {"anticipation": .6, "surprise": .4},
    "surprise": {"surprise": 1}, "realization": {"surprise": 1}, "confusion": {"surprise": 1},
    "fear": {"fear": 1}, "nervousness": {"fear": .7, "anticipation": .3},
    "sadness": {"sadness": 1}, "grief": {"sadness": 1}, "disappointment": {"sadness": 1},
    "remorse": {"sadness": .7, "disgust": .3}, "embarrassment": {"sadness": .5, "fear": .5},
    "anger": {"anger": 1}, "annoyance": {"anger": 1},
    "disapproval": {"disgust": .5, "anger": .5}, "disgust": {"disgust": 1},
    "neutral": {},                       # no emotion: deliberately maps to nothing
}

# ---------- Table 2: Plutchik emotion -> rasa  ----------
PLUTCHIK_TO_RASA = {
    "joy":          {"Hasya": 1, "Shringara": .4},
    "trust":        {"Shringara": .6, "Shanta": .6, "Veera": .3},
    "fear":         {"Bhayanaka": 1},
    "surprise":     {"Adbhuta": 1},
    "sadness":      {"Karuna": 1},
    "disgust":      {"Bibhatsa": 1},
    "anger":        {"Raudra": 1},
    "anticipation": {"Veera": .6, "Adbhuta": .3, "Shringara": .3},
}

# ---------- Load the classifier and read its label order from the model itself ----------
go_clf = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions",
                  top_k=None, truncation=True)
GO_LABELS = [go_clf.model.config.id2label[i] for i in range(len(go_clf.model.config.id2label))]
missing = set(GO_LABELS) - set(GO_TO_PLUTCHIK)
assert not missing, f"add these labels to GO_TO_PLUTCHIK: {missing}"

# ---------- Compose the two tables: M[label, rasa] = sum over Plutchik of A * B ----------
A = np.array([[GO_TO_PLUTCHIK[g].get(p, 0) for p in PLUTCHIK] for g in GO_LABELS])   # (28, 8)
B = np.array([[PLUTCHIK_TO_RASA[p].get(r, 0) for r in RASAS] for p in PLUTCHIK])     # (8, 9)
M = A @ B                                                                              # (28, 9)


def go_label_probs(sents):
    """Sentences -> (n, 28) GoEmotions scores (each label is an independent 0..1 probability)."""
    out = np.zeros((len(sents), len(GO_LABELS)))
    for i, scores in enumerate(go_clf(list(sents), batch_size=32)):
        for s in scores:
            out[i, GO_LABELS.index(s["label"])] = s["score"]
    return out


ALPHA = 0.3     # prior strength; higher = more cautious with weak evidence

def go_probs(sents):
    L = go_label_probs(sents)                          # (n, 28)
    evidence = L @ M                                   # (n, 9) raw evidence per rasa
    neutral = L[:, [GO_LABELS.index("neutral")]]       # (n, 1) "no emotion" probability
    prior = (ALPHA + neutral) / len(RASAS)             # spread evenly over the 9 rasas
    P = evidence + prior
    return P / P.sum(axis=1, keepdims=True)


def explain(text, top=5):
    """Debug helper: which GoEmotions labels fired, and what rasas they gave."""
    p = go_label_probs([text])[0]
    print(text)
    for j in np.argsort(-p)[:top]:
        print(f"  {GO_LABELS[j]:15} {p[j]:.2f}")
    print(pd.Series(go_probs([text])[0], index=RASAS).round(2).sort_values(ascending=False).head(4).to_string())


if __name__ == "__main__":
    from sample_dataset import labeled
    from retriever_eval import run_evaluation
    explain("I'm so nervous about the surgery, my heart is racing")
    explain("play something energetic")
    run_evaluation(labeled, go_probs, name="GoEmotions")