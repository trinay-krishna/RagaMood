"""Compare zero-shot / LogReg / SetFit on rasa_dataset.csv with shared stratified 5-fold CV.

Run:  python compare_models.py [--setfit-iters 10] [--setfit-epochs 1] [--only zeroshot,logreg,setfit]
Everything is written to ./results ; the best model is saved to ./models/best.
Results are on SYNTHETIC data and are optimistic.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "rasa_dataset.csv"
RES = HERE / "results"
ENC = "sentence-transformers/all-MiniLM-L6-v2"
LABELS = ["Shringara", "Hasya", "Karuna", "Raudra", "Veera",
          "Bhayanaka", "Bibhatsa", "Adbhuta", "Shanta", "None"]
RASAS = LABELS[:9]
NONE = 9
SEED = 0

# copied from ../eval_baseline.py (not imported: that file runs on import)
RASA_EXAMPLES = {
    "Adbhuta":   ["I want a song that is amazing", "something awe-inspiring that blows my mind", "I feel amazed and full of wonder"],
    "Karuna":    ["I'm feeling sad and heartbroken", "play sad, emotional songs", "I miss someone and I feel empty"],
    "Hasya":     ["I want something fun and funny", "a cheerful song that makes me laugh", "I'm in a playful, joking mood"],
    "Shanta":    ["I want to relax and feel calm", "peaceful music to meditate to", "I need quiet and stillness"],
    "Veera":     ["I want a powerful song that pumps me up", "music to feel brave and determined", "give me an anthem to push through this"],
    "Raudra":    ["I'm furious and want angry music", "something intense to vent my rage", "I'm so angry I can't think straight"],
    "Shringara": ["I want a romantic song", "music for being in love", "I'm missing my partner and feeling tender"],
    "Bhayanaka": ["I feel scared and anxious", "something eerie and tense", "I'm terrified and can't relax"],
    "Bibhatsa":  ["I feel disgusted and sick of everything", "I'm repulsed by what I saw today", "I can't stand how corrupt it all is"],
}


def load():
    df = pd.read_csv(DATA, encoding="utf-8-sig", keep_default_na=False)
    y = df["primary_rasa"].map({l: i for i, l in enumerate(LABELS)})
    assert y.notna().all(), "unknown primary_rasa label"
    sec = [[LABELS.index(s) for s in v.split("|") if s] for v in df["secondary_rasas"]]
    return df["user_text"].tolist(), y.astype(int).to_numpy(), sec


# ---------------- model 1: zero-shot ----------------
class ZeroShot:
    def __init__(self, enc, temperature=0.05):
        self.t = temperature
        pairs = [(r, t) for r in RASAS for t in RASA_EXAMPLES[r]]
        self.E = enc.encode([t for _, t in pairs], normalize_embeddings=True)
        self.owner = np.array([RASAS.index(r) for r, _ in pairs])

    def probs(self, Q):                       # Q: normalized embeddings -> (n, 10); None column = 0
        S = Q @ self.E.T
        sc = np.stack([S[:, self.owner == k].max(axis=1) for k in range(9)], axis=1)
        z = sc / self.t
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        return np.hstack([p, np.zeros((len(p), 1))])


# ---------------- model 2: logistic regression on frozen embeddings ----------------
def fit_logreg(X, y):
    gs = GridSearchCV(LogisticRegression(max_iter=2000),
                      {"C": [0.1, 0.3, 1, 3, 10, 30, 100]},
                      cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                      scoring="f1_macro", n_jobs=1)
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_["C"]


def full_probs(clf, X):
    out = np.zeros((len(X), 10))
    out[:, clf.classes_] = clf.predict_proba(X)
    return out


# ---------------- model 3: SetFit ----------------
def fit_setfit(texts, y, epochs, iters, batch):
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments
    model = SetFitModel.from_pretrained(ENC, labels=LABELS)
    args = TrainingArguments(batch_size=batch, num_epochs=epochs, num_iterations=iters,
                             seed=SEED, show_progress_bar=False, report_to="none")
    ds = Dataset.from_dict({"text": texts, "label": [int(v) for v in y]})
    Trainer(model=model, args=args, train_dataset=ds).train()
    return model


def setfit_probs(model, texts):
    return np.asarray(model.predict_proba(texts, as_numpy=True) if "as_numpy" in
                      model.predict_proba.__code__.co_varnames else model.predict_proba(texts))


# ---------------- metrics ----------------
def rank_rasas(P):                            # ordering over the 9 rasas only
    return np.argsort(-P[:, :9], axis=1, kind="stable")


def fold_metrics(P, y, sec, idx):
    """Metrics on the emotional rows of one fold (idx = global indices of fold's test rows)."""
    emo = y[idx] != NONE
    ii = idx[emo]
    Pe, ye = P[emo], y[ii]
    order = rank_rasas(Pe)
    top1 = np.mean(order[:, 0] == ye)
    top2 = np.mean([ye[k] in order[k, :2] for k in range(len(ye))])
    pred = order[:, 0]
    m = {"top1": top1, "top2": top2,
         "macro_f1": f1_score(ye, pred, labels=range(9), average="macro", zero_division=0)}
    rec = recall_score(ye, pred, labels=range(9), average=None, zero_division=0)
    prec = precision_score(ye, pred, labels=range(9), average=None, zero_division=0)
    for k, r in enumerate(RASAS):
        m[f"recall_{r}"], m[f"precision_{r}"] = rec[k], prec[k]
    has = [k for k, g in enumerate(ii) if sec[g]]
    for n in (2, 3):
        m[f"sec_hit_top{n}"] = (np.mean([any(s in order[k, :n] for s in sec[ii[k]]) for k in has])
                                if has else np.nan)
    # None handling
    nn = y[idx] == NONE
    m["maxprob_emotional"] = Pe[:, :9].max(axis=1).mean()
    m["maxprob_none"] = P[nn][:, :9].max(axis=1).mean() if nn.any() else np.nan
    m["maxprob_gap"] = m["maxprob_emotional"] - m["maxprob_none"]
    m["none_pred_as_rasa"] = (np.mean(P[nn].argmax(axis=1) != NONE) if nn.any() else np.nan)
    m["emo_pred_as_none"] = np.mean(P[emo].argmax(axis=1) == NONE)
    return m


def cm_plot(y_true, y_pred, name, labels, path):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    ax.set(xlabel="Predicted", ylabel="True", title=f"Pooled out-of-fold: {name}")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return cm


def top_confusions(cm, labels, k=5):
    pairs = [(cm[i, j], labels[i], labels[j]) for i in range(len(labels))
             for j in range(len(labels)) if i != j and cm[i, j]]
    return sorted(pairs, reverse=True)[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setfit-iters", type=int, default=10)
    ap.add_argument("--setfit-epochs", type=int, default=1)
    ap.add_argument("--setfit-batch", type=int, default=16)
    ap.add_argument("--only", default="zeroshot,logreg,setfit")
    ap.add_argument("--max-folds", type=int, default=5, help="debug / timing: stop after N folds")
    a = ap.parse_args()
    which = a.only.split(",")

    RES.mkdir(exist_ok=True)
    texts, y, sec = load()
    n = len(texts)
    print(f"{n} rows, {np.sum(y == NONE)} None, {sum(1 for s in sec if s)} with secondary")

    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(texts, y))
    seen = np.concatenate([te for _, te in folds])
    assert sorted(seen) == list(range(n)), "folds must partition the data"

    enc = SentenceTransformer(ENC)
    X = enc.encode(texts, normalize_embeddings=True, batch_size=64)
    zs = ZeroShot(enc)

    oof = {m: np.zeros((n, 10)) for m in which}
    rows = []
    chosen_C = []
    for f, (tr, te) in enumerate(folds[:a.max_folds]):
        assert not set(tr) & set(te)
        t0 = time.time()
        if "zeroshot" in which:
            oof["zeroshot"][te] = zs.probs(X[te])
        if "logreg" in which:
            clf, C = fit_logreg(X[tr], y[tr])
            chosen_C.append(C)
            oof["logreg"][te] = full_probs(clf, X[te])
        if "setfit" in which:
            ts = time.time()
            sm = fit_setfit([texts[i] for i in tr], y[tr], a.setfit_epochs, a.setfit_iters, a.setfit_batch)
            oof["setfit"][te] = setfit_probs(sm, [texts[i] for i in te])
            print(f"  fold {f}: setfit {time.time() - ts:.0f}s")
        for m in which:
            rows.append({"model": m, "fold": f, **fold_metrics(oof[m][te], y, sec, te)})
        print(f"fold {f} done in {time.time() - t0:.0f}s")

    done = np.concatenate([te for _, te in folds[:a.max_folds]])
    pf = pd.DataFrame(rows)
    pf.to_csv(RES / "metrics_per_fold.csv", index=False)
    summ = pf.drop(columns="fold").groupby("model").agg(["mean", "std"])
    summ.columns = [f"{c}_{s}" for c, s in summ.columns]
    summ.to_csv(RES / "metrics_summary.csv")

    key = ["top1", "top2", "macro_f1", "sec_hit_top2", "sec_hit_top3", "maxprob_gap",
           "none_pred_as_rasa", "emo_pred_as_none"]
    print("\n=== mean +/- std over folds (emotional rows; synthetic data) ===")
    for m in which:
        g = pf[pf.model == m]
        print(f"{m:9}", "  ".join(f"{k}={g[k].mean():.3f}±{g[k].std():.3f}" for k in key))
    print("\n=== per-class recall (mean over folds) ===")
    pc = pd.DataFrame({m: {r: pf[pf.model == m][f"recall_{r}"].mean() for r in RASAS} for m in which})
    pcp = pd.DataFrame({m: {r: pf[pf.model == m][f"precision_{r}"].mean() for r in RASAS} for m in which})
    print(pc.round(2))
    pd.concat({"recall": pc, "precision": pcp}, axis=1).to_csv(RES / "per_class.csv")

    conf = {}
    for m in which:
        pd.DataFrame(oof[m][done], columns=LABELS).assign(text=[texts[i] for i in done],
                                                          true=[LABELS[i] for i in y[done]]
                                                          ).to_csv(RES / f"oof_probs_{m}.csv", index=False)
        labs = LABELS if m != "zeroshot" else RASAS
        emo = y[done] != NONE if m == "zeroshot" else np.ones(len(done), bool)
        pred = oof[m][done][emo].argmax(axis=1)
        cm = cm_plot(y[done][emo], pred, m, LABELS, RES / f"confusion_{m}.png")
        conf[m] = top_confusions(cm, LABELS)
        print(f"\n{m} top confusions (count, true -> predicted):", conf[m])

    if chosen_C:
        print("\nLogReg chosen C per fold:", chosen_C)

    # paired per-fold differences vs. best model
    best = max(which, key=lambda m: pf[pf.model == m]["macro_f1"].mean())
    print(f"\nBest by mean macro-F1: {best}")
    for m in which:
        if m != best:
            d = pf[pf.model == best].set_index("fold")["macro_f1"] - pf[pf.model == m].set_index("fold")["macro_f1"]
            print(f"  {best} - {m}: macro-F1 diff mean {d.mean():+.3f}, std {d.std():.3f}, "
                  f"wins {int((d > 0).sum())}/{len(d)} folds")
    json.dump({"best": best, "logreg_C": chosen_C,
               "setfit": {"epochs": a.setfit_epochs, "iters": a.setfit_iters, "batch": a.setfit_batch}},
              open(RES / "run_info.json", "w"), indent=2)

    if a.max_folds == 5 and best != "zeroshot":
        save_best(best, texts, X, y, a)
    elif best == "zeroshot":
        print("zero-shot is best: no trained artifact to save (it has no None class).")


def save_best(best, texts, X, y, a):
    out = HERE / "models" / "best"
    out.mkdir(parents=True, exist_ok=True)
    if best == "logreg":
        clf, C = fit_logreg(X, y)
        joblib.dump({"kind": "logreg", "clf": clf, "encoder": ENC, "labels": LABELS}, out / "model.joblib")
    else:
        sm = fit_setfit(texts, y, a.setfit_epochs, a.setfit_iters, a.setfit_batch)
        sm.save_pretrained(str(out / "setfit"))
        joblib.dump({"kind": "setfit", "path": "setfit", "labels": LABELS}, out / "model.joblib")
    print(f"saved best model ({best}) to {out}")


if __name__ == "__main__":
    main()
