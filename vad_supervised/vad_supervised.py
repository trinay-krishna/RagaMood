"""VAD-only retriever calibrated on rasa_dataset.csv's Low/Medium/High valence and arousal labels.

Text features are only the 3 VAD numbers from VADRetriever.nrc_vad_sentences. No rasa classifier on
sentence embeddings, and nothing from rasa_compare. Everything it writes lives in this folder.

Run:  python vad_supervised/vad_supervised.py        (5-fold CV report, then fits and saves the final model)
Use:  from vad_supervised.vad_supervised import vad_probs   ->  (n, 9) probabilities in common.RASAS order
"""
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from common import RASAS

CACHE = HERE / "cache"
OUT = HERE / "results"
LEVELS = ["Low", "Medium", "High"]
CLASSES = RASAS + ["None"]               # the dataset also has a "None" (no emotion) class
BHAYA_BIBHATSA = [RASAS.index("Bhayanaka"), RASAS.index("Bibhatsa")]
ALPHA = 1.0                              # smoothing for the P(rasa | valence, arousal) table
D_SD_FLOOR = 0.05                        # min std of the per-rasa dominance Gaussian
W_GRID = [0.0, 0.5, 1.0]                 # weight of the dominance tie-break (0 = off)
TEMP_GRID = [0.5, 1.0, 1.5, 2.0, 3.0]


# ============ 1. Data and features ============
def load_data():
    """-> texts, valence level, arousal level, class index (0..9, 9 = None)."""
    df = pd.read_csv(ROOT / "rasa_dataset.csv", encoding="utf-8-sig", keep_default_na=False)
    lvl = lambda col: df[col].str.strip().str.capitalize().map(LEVELS.index)
    v, a = lvl("Valence"), lvl("arousal")
    y = df["primary_rasa"].str.strip().map(CLASSES.index)
    assert not (v.isna().any() or a.isna().any() or y.isna().any()), "unexpected label value"
    return df["user_text"].tolist(), v.to_numpy(int), a.to_numpy(int), y.to_numpy(int)


def vad_features(texts):
    from VADRetriever import nrc_vad_sentences      # heavy import: loads the encoder + ridge model
    return nrc_vad_sentences(texts)


def dataset_features(texts):
    """VAD for the dataset texts, cached by a hash of the texts."""
    key = hashlib.md5("\n".join(texts).encode("utf-8")).hexdigest()[:12]
    path = CACHE / f"dataset_vad_{key}.npy"
    if path.exists():
        return np.load(path)
    X = vad_features(texts)
    CACHE.mkdir(exist_ok=True)
    np.save(path, X)
    return X


# ============ 2. The model ============
def softmax(s):
    s = s - s.max(axis=1, keepdims=True)
    p = np.exp(s)
    return p / p.sum(axis=1, keepdims=True)


class Model:
    """x = (V, A, D) -> P(level | V), P(level | A) -> P(rasa | V-level, A-level) table -> rasa scores.

    Dominance is only a tie-break: a per-rasa Gaussian on D, weighted by w.
    """

    def __init__(self, w=0.0, temp=1.0):
        self.w, self.temp = w, temp

    def fit(self, X, v, a, y):
        self.vcal = LogisticRegression(C=10).fit(X[:, [0]], v)     # 1-D: just calibrates V -> level
        self.acal = LogisticRegression(C=10).fit(X[:, [1]], a)
        counts = np.full((3, 3, len(CLASSES)), ALPHA)
        np.add.at(counts, (v, a, y), 1)
        self.table = counts / counts.sum(axis=2, keepdims=True)    # P(rasa | V-level, A-level)
        self.mu = np.array([X[y == c, 2].mean() for c in range(len(CLASSES))])
        self.sd = np.array([max(X[y == c, 2].std(), D_SD_FLOOR) for c in range(len(CLASSES))])
        return self

    @staticmethod
    def _levels(cal, x):
        p = np.zeros((len(x), 3))
        p[:, cal.classes_] = cal.predict_proba(x[:, None])
        return p

    def levels(self, X):
        return self._levels(self.vcal, X[:, 0]), self._levels(self.acal, X[:, 1])

    def log_scores(self, X):
        pv, pa = self.levels(X)
        mix = np.einsum("ni,nj,ijc->nc", pv, pa, self.table)
        z = (X[:, [2]] - self.mu) / self.sd
        return np.log(mix) + self.w * (-0.5 * z ** 2 - np.log(self.sd))

    def predict_proba(self, X):
        return softmax(self.log_scores(X) / self.temp)


# ============ 3. Public API (same shape as VADRetriever.vad_probs) ============
_model = None

def _get_model():
    global _model
    if _model is None:
        path = CACHE / "model.joblib"
        if not path.exists():
            texts, v, a, y = load_data()
            _model = Model(w=0.0).fit(dataset_features(texts), v, a, y)
            CACHE.mkdir(exist_ok=True)
            joblib.dump(_model, path)
        else:
            _model = joblib.load(path)
    return _model


def vad_probs(sents):
    """Sentences -> (n, 9) probabilities over RASAS (the None column is dropped and the rest renormalised)."""
    P = _get_model().predict_proba(vad_features(list(sents)))[:, :len(RASAS)]
    return P / P.sum(axis=1, keepdims=True)


# ============ 4. Cross-validated evaluation ============
def metrics(S, y):
    """S = (n, 9) scores, y = true rasa index 0..8."""
    order = np.argsort(-S, axis=1)
    pred = order[:, 0]
    return {
        "top1": (pred == y).mean(),
        "top3": (order[:, :3] == y[:, None]).any(axis=1).mean(),
        "macro_f1": f1_score(y, pred, labels=range(len(RASAS)), average="macro", zero_division=0),
        "top1_merged": ((pred == y) | (np.isin(pred, BHAYA_BIBHATSA) & np.isin(y, BHAYA_BIBHATSA))).mean(),
    }


def recall_by_rasa(S, y):
    pred = S.argmax(axis=1)
    return [(pred[y == r] == r).mean() for r in range(len(RASAS))]


def cross_val(X, v, a, y):
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    scores = {w: np.zeros((len(y), len(CLASSES))) for w in W_GRID}
    pv_oof, pa_oof = np.zeros((len(y), 3)), np.zeros((len(y), 3))
    oracle = np.zeros((len(y), len(CLASSES)))
    for tr, te in skf.split(X, y):
        m = Model().fit(X[tr], v[tr], a[tr], y[tr])
        pv_oof[te], pa_oof[te] = m.levels(X[te])
        oracle[te] = m.table[v[te], a[te]]               # what the true (V, A) levels would give
        for w in W_GRID:
            m.w = w
            scores[w][te] = m.log_scores(X[te])
    return scores, pv_oof, pa_oof, oracle


# ============ 5. Inspection outputs ============
def _fig_html(fig):
    import base64
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    fig.clf()
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def write_report(texts, X, v, a, y, pv, pa, S, w, table, recalls):
    """predictions.csv, confusion_best.csv and report.html from the out-of-fold results.

    S = (n, 10) out-of-fold log-scores at dominance weight w.
    """
    n, k = len(y), len(RASAS)
    P = softmax(S[:, :k])
    pred = P.argmax(axis=1)
    rasa_rows = y < k
    top3 = np.argsort(-P, axis=1)[:, :3]
    pred_df = pd.DataFrame({
        "text": texts, "true": [CLASSES[i] for i in y], "pred": [RASAS[i] for i in pred],
        "confidence": P.max(axis=1).round(3),
        "top3": [", ".join(RASAS[j] for j in r) for r in top3],
        "correct": np.where(rasa_rows, pred == y, np.nan),
        "V": X[:, 0].round(3), "A": X[:, 1].round(3), "D": X[:, 2].round(3),
        "true_V": [LEVELS[i] for i in v], "pred_V": [LEVELS[i] for i in pv.argmax(1)],
        "true_A": [LEVELS[i] for i in a], "pred_A": [LEVELS[i] for i in pa.argmax(1)],
    })
    OUT.mkdir(exist_ok=True)
    pred_df.to_csv(OUT / "predictions.csv", index=False, encoding="utf-8-sig")

    conf = pd.DataFrame(0, index=RASAS, columns=RASAS)
    for t, p in zip(y[rasa_rows], pred[rasa_rows]):
        conf.iloc[t, p] += 1
    conf.to_csv(OUT / "confusion_best.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipped report.html")
        return

    html = [f"<h2>vad_supervised: out-of-fold results (dominance w={w})</h2>",
            "<h3>Metrics</h3>", table.round(3).to_html()]

    rec = pd.DataFrame(recalls, index=RASAS)
    rec.plot.bar(figsize=(10, 4), width=0.8)
    plt.ylabel("recall"); plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    html += ["<h3>Per-class recall</h3>", _fig_html(plt.gcf())]

    norm = conf.div(conf.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(norm.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(k), RASAS, rotation=45, ha="right"); ax.set_yticks(range(k), RASAS)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    for i in range(k):
        for j in range(k):
            ax.text(j, i, conf.iloc[i, j], ha="center", va="center",
                    color="white" if norm.iloc[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, label="row-normalised")
    html += ["<h3>Confusion matrix (counts, shading = row share)</h3>", _fig_html(fig)]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, name, true, p in zip(axes, ["valence", "arousal"], [v, a], [pv, pa]):
        cm = np.zeros((3, 3), int)
        np.add.at(cm, (true, p.argmax(1)), 1)
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(3), LEVELS); ax.set_yticks(range(3), LEVELS)
        ax.set_title(f"{name} level (acc {np.trace(cm) / n:.2f})")
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.tight_layout()
    html += ["<h3>Valence / arousal level confusion</h3>", _fig_html(fig)]

    fig, ax = plt.subplots(figsize=(7, 6))
    for c, name in enumerate(CLASSES):
        m = y == c
        ax.scatter(X[m, 0], X[m, 1], s=14, alpha=0.6, label=name)
    ax.set_xlabel("valence"); ax.set_ylabel("arousal"); ax.legend(fontsize=8, ncol=2)
    html += ["<h3>VAD features by true class</h3>", _fig_html(fig)]

    wrong = pred_df[rasa_rows & (pred != y)].sort_values("confidence", ascending=False).head(20)
    html += ["<h3>Most confident mistakes</h3>",
             wrong[["text", "true", "pred", "confidence", "top3", "V", "A", "D"]].to_html(index=False)]

    css = "<style>body{font-family:sans-serif;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:3px 8px;font-size:13px}</style>"
    (OUT / "report.html").write_text("<meta charset='utf-8'>" + css + "\n".join(html), encoding="utf-8")


def main():
    texts, v, a, y = load_data()
    X = dataset_features(texts)
    rasa_rows = y < len(RASAS)                           # ranking metrics skip the "None" rows

    scores, pv, pa, oracle = cross_val(X, v, a, y)
    rows, recalls = {}, {}

    from VADRetriever import vad_probs as base_probs      # current hand-anchored retriever, untrained
    base = base_probs([t for t, k in zip(texts, rasa_rows) if k])
    rows["baseline VADRetriever"] = metrics(base, y[rasa_rows])
    recalls["baseline VADRetriever"] = recall_by_rasa(base, y[rasa_rows])

    rows["oracle (true V,A levels)"] = metrics(oracle[rasa_rows, :len(RASAS)], y[rasa_rows])
    for w in W_GRID:
        name = f"calibrated, dominance w={w}"
        S = scores[w][rasa_rows, :len(RASAS)]
        rows[name] = metrics(S, y[rasa_rows])
        recalls[name] = recall_by_rasa(S, y[rasa_rows])

    best_w = max(W_GRID, key=lambda w: (rows[f"calibrated, dominance w={w}"]["top1"], -w))
    best = scores[best_w]
    nll = {t: -np.log(softmax(best / t)[np.arange(len(y)), y]).mean() for t in TEMP_GRID}
    best_t = min(nll, key=nll.get)

    table = pd.DataFrame(rows).T
    print(table.round(3).to_string())
    print(f"\nrows used for ranking metrics: {rasa_rows.sum()} (None rows skipped)")
    print(f"valence level acc {(pv.argmax(1) == v).mean():.3f}   arousal level acc {(pa.argmax(1) == a).mean():.3f}")
    print(f"chosen dominance weight {best_w}, temperature {best_t} (OOF log loss {nll[best_t]:.3f})")
    print("\nper-rasa recall")
    print(pd.DataFrame(recalls, index=RASAS).round(2).to_string())

    OUT.mkdir(exist_ok=True)
    table.to_csv(OUT / "cv_metrics.csv")
    pd.DataFrame(recalls, index=RASAS).to_csv(OUT / "per_class_recall.csv")
    write_report(texts, X, v, a, y, pv, pa, best, best_w, table, recalls)

    # final model on all rows, saved for vad_probs()
    final = Model(w=best_w, temp=best_t).fit(X, v, a, y)
    CACHE.mkdir(exist_ok=True)
    joblib.dump(final, CACHE / "model.joblib")
    global _model
    _model = final
    demo = ["I can't stop smiling today.", "I'm terrified of what the results might say.",
            "I feel completely at peace with everything.", "I will destroy you all"]
    print("\n" + pd.DataFrame(vad_probs(demo), columns=RASAS, index=demo).round(2).to_string())


if __name__ == "__main__":
    main()
