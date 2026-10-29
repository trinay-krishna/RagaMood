import contextlib
import io
import json
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # save figures to disk, never block on a window
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from common import RASAS, RESULTS_DIR


def _slug(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()


def run_evaluation(labeled, prob_fn, name="model", equivalent=None, n_boot=1000, seed=0, out_dir=None):
    """
    labeled    : [(sentence, rasa_or_set_of_rasas), ...]
    prob_fn    : function(list_of_sentences) -> (n, 9) probabilities in RASAS order
    equivalent : groups this scorer can't tell apart, e.g. [{"Bhayanaka", "Bibhatsa"}]
    out_dir    : where results are saved; defaults to results/<name>/ (overwritten each run)
    """
    out_dir = RESULTS_DIR / _slug(name) if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    sents = [s for s, _ in labeled]
    n = len(sents)
    gold = [{g} if isinstance(g, str) else set(g) for _, g in labeled]
    for grp in (equivalent or []):
        gold = [g | grp if g & grp else g for g in gold]

    P = np.asarray(prob_fn(sents))
    order = np.argsort(-P, axis=1, kind="stable")                    # best first; ties keep RASAS order
    pred1 = [RASAS[o[0]] for o in order]

    hit1 = np.array([pred1[i] in gold[i] for i in range(n)], dtype=float)
    hit3 = np.array([any(RASAS[j] in gold[i] for j in order[i, :3]) for i in range(n)], dtype=float)
    rr = np.array([next((1 / (r + 1) for r, j in enumerate(order[i]) if RASAS[j] in gold[i]), 0)
                   for i in range(n)])

    pred_raw = list(pred1)                                           # before merging, for predictions.csv
    y_true = [pred1[i] if hit1[i] else sorted(gold[i])[0] for i in range(n)]
    for grp in (equivalent or []):                                   # show merged rasas as one class
        merged = "/".join(sorted(grp))
        y_true = [merged if t in grp else t for t in y_true]
        pred1 = [merged if p in grp else p for p in pred1]
    macro_f1 = f1_score(y_true, pred1, average="macro", zero_division=0)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    ci = lambda v: np.percentile(v[idx].mean(axis=1), [2.5, 97.5])
    chance = 1 / len(RASAS)
    ci1, ci3 = ci(hit1), ci(hit3)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):                            # captured, then printed and saved
        print(f"\n===== {name}  (n={n}) =====")
        print(f"top-1 accuracy : {hit1.mean():.0%}   95% CI {ci1[0]:.0%} to {ci1[1]:.0%}   (chance ~{chance:.0%})")
        print(f"top-3 accuracy : {hit3.mean():.0%}   95% CI {ci3[0]:.0%} to {ci3[1]:.0%}   (chance ~{3 * chance:.0%})")
        print(f"MRR            : {rr.mean():.2f}")
        print(f"macro-F1       : {macro_f1:.2f}")
        report = classification_report(y_true, pred1, zero_division=0)
        print("\nPer-rasa precision / recall / F1:")
        print(report)

        wrong = [(sents[i], sorted(gold[i]), pred1[i]) for i in range(n) if not hit1[i]]
        print(f"\nMisclassified ({len(wrong)}):")
        for s, g, p in wrong:
            print(f"  correct={g}  predicted={p}  | {s[:80]}")
    text = buf.getvalue()
    print(text)

    labels = sorted(set(y_true) | set(pred1),
                    key=lambda r: RASAS.index(r) if r in RASAS else len(RASAS))
    cm = confusion_matrix(y_true, pred1, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(xlabel="Predicted", ylabel="Correct", title=f"Confusion matrix: {name}")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    metrics = {"name": name, "n": n,
               "top1": float(hit1.mean()), "top1_ci": [float(x) for x in ci1],
               "top3": float(hit3.mean()), "top3_ci": [float(x) for x in ci3],
               "mrr": float(rr.mean()), "macro_f1": float(macro_f1)}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "summary.txt").write_text(text, encoding="utf-8")
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    (out_dir / "misclassified.txt").write_text(
        "\n".join(f"correct={g}  predicted={p}  | {s}" for s, g, p in wrong), encoding="utf-8")

    pred_df = pd.DataFrame({
        "sentence": sents,
        "gold": ["|".join(sorted(g)) for g in gold],
        "pred_top1": pred_raw,
        "pred_top3": ["|".join(RASAS[j] for j in order[i, :3]) for i in range(n)],
        "correct_top1": hit1.astype(bool),
    })
    for k, r in enumerate(RASAS):
        pred_df[f"p_{r}"] = P[:, k]
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    print(f"Saved results to {out_dir}")

    return {"top1": hit1.mean(), "top3": hit3.mean(), "mrr": rr.mean(), "macro_f1": macro_f1}
