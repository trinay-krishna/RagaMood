"""Rasa reranker: fuses vad_supervised, embeddingRetriever and rasa_classifier into one (n, 9) distribution.

Stage 1: rasa_classifier's top-K rasas are the candidates.
Stage 2: a conditional logit (softmax over the candidates, shared weights + per-rasa bias) rescores them
from each system's log-prob and top-1 flag. Fit on out-of-fold component outputs so nothing leaks.
Everything it writes lives in this folder.

Run:  python rerank/rerank.py        (CV report vs single systems / RRF / log-linear fusion, then fits and saves)
Use:  from rerank.rerank import rerank_probs   ->  (n, 9) probabilities in common.RASAS order
"""
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rasa_compare"))
sys.stdout.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from common import RASAS
from vad_supervised.vad_supervised import (CLASSES, Model, cross_val, dataset_features, load_data,
                                           metrics, recall_by_rasa, softmax)

CACHE = HERE / "cache"
OUT = HERE / "results"
K = len(RASAS)
SYSTEMS = ["vad", "emb", "clf"]
VAD_W = 0.5                              # dominance weight chosen by vad_supervised's CV
TOP_K = 3                                # candidates passed from stage 1 to the reranker
L2 = 0.1
RRF_K = 60
EPS = 1e-4                               # mass kept for non-candidates so top-3 still has an order


# ============ 1. Component probabilities ============
def _renorm9(P):
    P = P[:, :K]
    return P / P.sum(axis=1, keepdims=True)


def component_probs(texts):
    """Full-data components, for inference -> {system: (n, 9)}."""
    from embeddingRetriever import emb_probs
    from rasa_classifier import predict
    from vad_supervised.vad_supervised import vad_probs
    return {"vad": vad_probs(texts), "emb": emb_probs(texts), "clf": _renorm9(predict(texts))}


def oof_components(texts, X, y):
    """Out-of-fold components on the dataset, one shared split -> {system: (n, 9)}."""
    key = hashlib.md5("\n".join(texts).encode("utf-8")).hexdigest()[:12]
    path = CACHE / f"oof_{key}.npz"
    if path.exists():
        z = np.load(path)
        return {s: z[s] for s in SYSTEMS}

    _, v, a, _ = load_data()
    scores, *_ = cross_val(X, v, a, y)                       # StratifiedKFold(5, shuffle, seed 0) on y
    vad = softmax(scores[VAD_W][:, :K])

    from embeddingRetriever import emb_probs
    emb = emb_probs(texts)

    meta = joblib.load(ROOT / "rasa_compare" / "models" / "best" / "model.joblib")
    from sentence_transformers import SentenceTransformer
    E = SentenceTransformer(meta["encoder"]).encode(texts, normalize_embeddings=True, batch_size=64)
    C = getattr(meta.get("clf"), "C", 1.0)
    clf = np.zeros((len(y), K))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(E, y):
        m = LogisticRegression(C=C, max_iter=2000).fit(E[tr], y[tr])
        P = np.zeros((len(te), len(CLASSES)))
        P[:, m.classes_] = m.predict_proba(E[te])
        clf[te] = _renorm9(P)

    CACHE.mkdir(exist_ok=True)
    np.savez(path, vad=vad, emb=emb, clf=clf)
    return {"vad": vad, "emb": emb, "clf": clf}


# ============ 2. The reranker ============
def _features(comp, cand):
    """-> (n, k, 6): per candidate, log-prob and top-1 flag of each system."""
    f = []
    for s in SYSTEMS:
        P = np.clip(comp[s], 1e-9, None)
        lp = np.log(P)
        top = P.argmax(axis=1)[:, None] == cand
        f += [np.take_along_axis(lp, cand, 1), top.astype(float)]
    return np.stack(f, axis=2)


class Reranker:
    """Conditional logit over the stage-1 candidates. top_k = 9 gives plain log-linear fusion."""

    def __init__(self, top_k=TOP_K, l2=L2):
        self.top_k, self.l2 = top_k, l2

    def _cand(self, comp):
        return np.argsort(-comp["clf"], axis=1)[:, :self.top_k]

    def _scores(self, theta, F, cand):
        nf = F.shape[2]
        return F @ theta[:nf] + theta[nf:][cand]

    def fit(self, comp, y):
        cand = self._cand(comp)
        F = _features(comp, cand)
        hit = (cand == y[:, None])
        keep = hit.any(axis=1)                                 # rows whose true rasa is a candidate
        F, cand, hit = F[keep], cand[keep], hit[keep]
        nf = F.shape[2]

        def loss(theta):
            p = softmax(self._scores(theta, F, cand))
            g = (p - hit) / len(F)                             # d loss / d score
            gb = np.zeros(K)
            np.add.at(gb, cand, g)
            grad = np.concatenate([np.einsum("nk,nkf->f", g, F), gb]) + self.l2 * theta
            nll = -np.log((p * hit).sum(axis=1)).mean()
            return nll + 0.5 * self.l2 * theta @ theta, grad

        theta0 = np.zeros(nf + K)
        theta0[0] = 1.0
        self.theta = minimize(loss, theta0, jac=True, method="L-BFGS-B").x
        return self

    def predict_proba(self, comp):
        cand = self._cand(comp)
        pc = softmax(self._scores(self.theta, _features(comp, cand), cand))
        P = EPS * comp["clf"]
        np.put_along_axis(P, cand, pc, axis=1)
        return P / P.sum(axis=1, keepdims=True)


# ============ 3. Baselines ============
def rrf(comp, k=RRF_K):
    S = np.zeros_like(comp["clf"])
    for s in SYSTEMS:
        rank = np.argsort(np.argsort(-comp[s], axis=1), axis=1)
        S += 1.0 / (k + rank + 1)
    return S


def cv_predict(make, comp, y, seed=1):
    """Out-of-fold probabilities from a second CV over the stacked component outputs."""
    out = np.zeros_like(comp["clf"])
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(y, y):
        r = make().fit({s: comp[s][tr] for s in SYSTEMS}, y[tr])
        out[te] = r.predict_proba({s: comp[s][te] for s in SYSTEMS})
    return out


# ============ 4. Public API ============
_reranker = None

def _get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = joblib.load(CACHE / "reranker.joblib")
    return _reranker


def rerank_probs(sents):
    """Sentences -> (n, 9) probabilities over RASAS."""
    return _get_reranker().predict_proba(component_probs(list(sents)))


# ============ 5. HTML report ============
def write_report(texts, yr, c, systems, table, recalls, final, final_name):
    """report.html: metrics, recall, confusion, what the reranker fixed / broke, learned weights."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipped report.html")
        return
    from vad_supervised.vad_supervised import _fig_html

    P = systems[final_name]
    pred, base = P.argmax(1), c["clf"].argmax(1)
    gain = (pred == yr).mean() - (base == yr).mean()
    best = table.idxmax()
    pick = lambda S: [RASAS[i] for i in S.argmax(1)]
    df = pd.DataFrame({"text": texts, "true": [RASAS[i] for i in yr], "reranked": pick(P),
                       "classifier": pick(c["clf"]), "embedding": pick(c["emb"]), "vad": pick(c["vad"]),
                       "confidence": P.max(1).round(3)})
    fixed = df[(pred == yr) & (base != yr)]
    broken = df[(pred != yr) & (base == yr)]

    sty = table.round(3).style.highlight_max(color="#c8e6c9").format("{:.3f}")
    html = [f"<h2>rerank: out-of-fold results on {len(yr)} rasa sentences</h2>",
            f"<p><b>{final_name}</b> top-1 {(pred == yr).mean():.3f} vs rasa_classifier {(base == yr).mean():.3f} "
            f"({gain * 100:+.1f} pts). Best per column highlighted.</p>", sty.to_html()]

    ax = table[["top1", "top3"]].plot.barh(figsize=(8, 4.5), width=0.8)
    ax.invert_yaxis(); ax.set_xlim(0, 1); ax.legend(loc="lower right")
    plt.tight_layout()
    html += ["<h3>Top-1 / top-3 by system</h3>", _fig_html(plt.gcf())]

    rec = pd.DataFrame(recalls, index=RASAS)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.imshow(rec.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(rec.shape[1]), rec.columns, rotation=30, ha="right"); ax.set_yticks(range(len(RASAS)), RASAS)
    for i in range(rec.shape[0]):
        for j in range(rec.shape[1]):
            ax.text(j, i, f"{rec.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if rec.iloc[i, j] > 0.6 else "black")
    fig.colorbar(im, label="recall")
    html += ["<h3>Per-rasa recall</h3>", _fig_html(fig)]

    d = (rec[final_name] - rec["rasa_classifier"])
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(RASAS, d, color=["#2e7d32" if x >= 0 else "#c62828" for x in d])
    ax.axhline(0, color="gray", lw=0.8); ax.set_ylabel("recall change"); plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    html += ["<h3>Reranker vs rasa_classifier, recall change per rasa</h3>", _fig_html(fig)]

    k = len(RASAS)
    conf = np.zeros((k, k), int)
    np.add.at(conf, (yr, pred), 1)
    norm = conf / conf.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(k), RASAS, rotation=45, ha="right"); ax.set_yticks(range(k), RASAS)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    for i in range(k):
        for j in range(k):
            ax.text(j, i, conf[i, j], ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, label="row share")
    html += ["<h3>Confusion matrix (reranker)</h3>", _fig_html(fig)]

    cols = ["text", "true", "reranked", "classifier", "embedding", "vad"]
    html += [f"<h3>Fixed by the reranker ({len(fixed)})</h3>", "<p>Right now, classifier was wrong.</p>",
             fixed[cols].to_html(index=False),
             f"<h3>Broken by the reranker ({len(broken)})</h3>", "<p>Classifier was right, reranker is wrong.</p>",
             broken[cols].to_html(index=False)]

    nf = 2 * len(SYSTEMS)
    names = [f"{s} {t}" for s in SYSTEMS for t in ("logp", "top1")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), gridspec_kw={"width_ratios": [1, 1.4]})
    axes[0].barh(names, final.theta[:nf]); axes[0].invert_yaxis(); axes[0].set_title("feature weights")
    axes[1].bar(RASAS, final.theta[nf:]); axes[1].set_title("per-rasa bias"); axes[1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    html += ["<h3>Learned weights</h3>", _fig_html(fig)]

    wrong = df[pred != yr].sort_values("confidence", ascending=False).head(20)
    html += ["<h3>Most confident mistakes</h3>", wrong[cols + ["confidence"]].to_html(index=False),
             "<p><small>Dataset is synthetic with single-author labels, so absolute numbers are optimistic. "
             "With 450 rows one standard error is about 2 points. The stacking CV is not nested.</small></p>"]

    css = ("<style>body{font-family:sans-serif;margin:24px}table{border-collapse:collapse}"
           "td,th{border:1px solid #ccc;padding:3px 8px;font-size:13px}</style>")
    (OUT / "report.html").write_text("<meta charset='utf-8'>" + css + "\n".join(html), encoding="utf-8")


# ============ 6. Run ============
def main():
    texts, v, a, y = load_data()
    X = dataset_features(texts)
    comp = oof_components(texts, X, y)

    rasa_rows = y < K
    c = {s: comp[s][rasa_rows] for s in SYSTEMS}
    yr = y[rasa_rows]

    systems = {"vad_supervised": c["vad"], "embeddingRetriever": c["emb"], "rasa_classifier": c["clf"],
               "RRF (k=60)": rrf(c)}
    systems["log-linear fusion (all 9)"] = cv_predict(lambda: Reranker(top_k=K), c, yr)
    for k in (2, TOP_K, 4):
        systems[f"rerank top-{k} candidates"] = cv_predict(lambda k=k: Reranker(top_k=k), c, yr)
    final_name = f"rerank top-{TOP_K} candidates"

    rows = {n: metrics(S, yr) for n, S in systems.items()}
    recalls = {n: recall_by_rasa(S, yr) for n, S in systems.items()}
    table = pd.DataFrame(rows).T
    print(table.round(3).to_string())
    print(f"\nrows used: {rasa_rows.sum()} (None rows skipped); candidate recall of rasa_classifier top-{TOP_K}: "
          f"{(np.argsort(-c['clf'], 1)[:, :TOP_K] == yr[:, None]).any(1).mean():.3f}")
    print("\nper-rasa recall")
    print(pd.DataFrame(recalls, index=RASAS).round(2).to_string())

    OUT.mkdir(exist_ok=True)
    table.to_csv(OUT / "cv_metrics.csv")
    pd.DataFrame(recalls, index=RASAS).to_csv(OUT / "per_class_recall.csv")
    P = systems[final_name]
    pd.DataFrame({
        "text": np.array(texts)[rasa_rows], "true": [RASAS[i] for i in yr],
        "pred": [RASAS[i] for i in P.argmax(1)], "clf_pred": [RASAS[i] for i in c["clf"].argmax(1)],
        "confidence": P.max(1).round(3), "correct": P.argmax(1) == yr,
    }).to_csv(OUT / "predictions.csv", index=False, encoding="utf-8-sig")

    final = Reranker().fit(c, yr)
    write_report(np.array(texts)[rasa_rows], yr, c, systems, table, recalls, final, final_name)
    CACHE.mkdir(exist_ok=True)
    joblib.dump(final, CACHE / "reranker.joblib")
    nf = 2 * len(SYSTEMS)
    print("\nlearned weights:", dict(zip([f"{s}_{t}" for s in SYSTEMS for t in ("logp", "top1")],
                                         final.theta[:nf].round(2))))
    print("per-rasa bias:", dict(zip(RASAS, final.theta[nf:].round(2))))

    global _reranker
    _reranker = final
    demo = ["I can't stop smiling today.", "I'm terrified of what the results might say.",
            "I feel completely at peace with everything.", "I will destroy you all"]
    print("\n" + pd.DataFrame(rerank_probs(demo), columns=RASAS, index=demo).round(2).to_string())


if __name__ == "__main__":
    main()
