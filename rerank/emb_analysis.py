"""Does the example-phrase matcher (embeddingRetriever, "emb") earn its place in the fused result?

Compares log-linear fusion with emb (vad+emb+clf) against without it (vad+clf) on the cached out-of-fold components:
significance of the gain, where it helps, complementarity, calibration, exemplar quality (curated phrases vs random
phrases vs class centroids) and low-data behaviour. Writes results/emb_analysis.html and emb_analysis.csv.

Run:  python rerank/emb_analysis.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rasa_compare"))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from common import RASAS
from rerank import K, OUT, Reranker, cv_predict, fig_html, oof_components, table_png
from vad_supervised.vad_supervised import dataset_features, load_data, metrics, recall_by_rasa

FULL = ["vad", "emb", "clf"]
NOEMB = ["vad", "clf"]
SEED = 1                 # second-level CV seed used by rerank.py's report
N_RANDOM = 200           # random phrase sets scored on their own
N_RANDOM_FUSED = 40      # ... and also pushed through the fusion
TEMP = 0.05              # emb_probs temperature


def fuse(c, y, systems, seed=SEED):
    return cv_predict(lambda: Reranker(top_k=K, systems=systems), c, y, seed=seed, systems=systems)


def top1(P, y):
    return P.argmax(1) == y


def top3(P, y):
    return (np.argsort(-P, 1)[:, :3] == y[:, None]).any(1)


def probs_from(Q, E, lab):
    """Same scoring as embeddingRetriever.emb_probs (max cosine per rasa, softmax at TEMP)."""
    S = Q @ E.T
    sc = np.stack([S[:, lab == k].max(1) for k in range(K)], axis=1) / TEMP
    sc -= sc.max(1, keepdims=True)
    p = np.exp(sc)
    return p / p.sum(1, keepdims=True)


def ece(P, y, bins=10):
    conf, ok = P.max(1), top1(P, y)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return e


def nll(P, y):
    return -np.log(np.clip(P[np.arange(len(y)), y], 1e-9, None)).mean()


def main():
    texts, v, a, y = load_data()
    comp = oof_components(texts, dataset_features(texts), y)
    rows = y < K
    c = {s: comp[s][rows] for s in FULL}
    yr, tx = y[rows], np.array(texts)[rows]
    n = len(yr)
    html, csv = [], {}

    P_full, P_no = fuse(c, yr, FULL), fuse(c, yr, NOEMB)
    A, B = top1(P_full, yr), top1(P_no, yr)
    A3, B3 = top3(P_full, yr), top3(P_no, yr)

    # 1. significance ---------------------------------------------------------------------------
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n, size=(2000, n))
    pf, pn = P_full.argmax(1), P_no.argmax(1)
    f1 = lambda p, i: f1_score(yr[i], p[i], labels=range(K), average="macro", zero_division=0)
    d1 = (A[idx].mean(1) - B[idx].mean(1))
    d3 = (A3[idx].mean(1) - B3[idx].mean(1))
    dF = np.array([f1(pf, i) - f1(pn, i) for i in idx[:500]])
    ci = lambda d: f"[{np.percentile(d, 2.5) * 100:+.1f}, {np.percentile(d, 97.5) * 100:+.1f}]"
    b, cc = int((A & ~B).sum()), int((~A & B).sum())
    p_mc = binomtest(min(b, cc), b + cc, 0.5).pvalue if b + cc else 1.0
    seed_gain = np.array([top1(fuse(c, yr, FULL, s), yr).mean() - top1(fuse(c, yr, NOEMB, s), yr).mean()
                          for s in range(10)])
    sig = pd.DataFrame({
        "with emb": [A.mean(), A3.mean(), f1(pf, slice(None))],
        "without emb": [B.mean(), B3.mean(), f1(pn, slice(None))],
        "gain (pts)": [(A.mean() - B.mean()) * 100, (A3.mean() - B3.mean()) * 100, (f1(pf, slice(None)) - f1(pn, slice(None))) * 100],
        "95% bootstrap CI (pts)": [ci(d1), ci(d3), ci(dF)]}, index=["top-1", "top-3", "macro-F1"])
    html += ["<h2>Does emb earn its place in the fusion?</h2><h3>1. Is the gain real?</h3>",
             sig.round(3).to_html(),
             f"<p>McNemar exact test on top-1: emb fixes {b}, breaks {cc} sentences, p = {p_mc:.3f}. "
             f"Over 10 second-level CV seeds the top-1 gain is {seed_gain.mean() * 100:+.2f} "
             f"+- {seed_gain.std() * 100:.2f} pts (min {seed_gain.min() * 100:+.1f}, max {seed_gain.max() * 100:+.1f}).</p>"]
    csv.update(gain_top1=A.mean() - B.mean(), mcnemar_p=p_mc, seed_gain_mean=seed_gain.mean(), seed_gain_sd=seed_gain.std())

    # 2. where it helps --------------------------------------------------------------------------
    rec = pd.DataFrame({"without emb": recall_by_rasa(P_no, yr), "with emb": recall_by_rasa(P_full, yr)}, index=RASAS)
    rec["delta"] = rec["with emb"] - rec["without emb"]
    df = pd.DataFrame({"text": tx, "true": [RASAS[i] for i in yr], "with emb": [RASAS[i] for i in pf],
                       "without emb": [RASAS[i] for i in pn]})
    html += ["<h3>2. Where it helps</h3>", rec.round(3).style.format("{:.3f}").background_gradient(
        subset=["delta"], cmap="RdYlGn", vmin=-0.2, vmax=0.2).to_html(),
             f"<h4>Fixed by emb ({b})</h4>", df[A & ~B].to_html(index=False),
             f"<h4>Broken by emb ({cc})</h4>", df[~A & B].to_html(index=False)]

    # 3. complementarity -------------------------------------------------------------------------
    ok = {s: top1(c[s], yr) for s in FULL}
    comp_tab = pd.DataFrame({
        "solo top-1": {s: ok[s].mean() for s in FULL},
        "right where clf is wrong": {s: (ok[s] & ~ok["clf"]).sum() for s in FULL},
        "right where all others are wrong": {s: (ok[s] & ~np.any([ok[o] for o in FULL if o != s], axis=0)).sum() for s in FULL}})
    oracle = lambda S: np.any([ok[s] for s in S], axis=0).mean()
    html += ["<h3>3. Complementarity (base systems, before fusion)</h3>", comp_tab.round(3).to_html(),
             f"<p>Oracle top-1 (right if any listed system is right): clf alone {oracle(['clf']):.3f}; "
             f"clf+emb {oracle(['clf', 'emb']):.3f}; clf+vad {oracle(['clf', 'vad']):.3f}; all three {oracle(FULL):.3f}.</p>"]
    csv.update(oracle_clf=oracle(["clf"]), oracle_clf_emb=oracle(["clf", "emb"]), oracle_clf_vad=oracle(["clf", "vad"]))

    # 4. calibration -------------------------------------------------------------------------------
    cal = pd.DataFrame({"with emb": [nll(P_full, yr), ece(P_full, yr)], "without emb": [nll(P_no, yr), ece(P_no, yr)]},
                       index=["NLL (lower better)", "ECE (lower better)"])
    html += ["<h3>4. Probability quality</h3><p>Matters because the playlist splits slots by rasa probability.</p>",
             cal.round(4).to_html()]
    csv.update(nll_with=cal.loc["NLL (lower better)", "with emb"], nll_without=cal.loc["NLL (lower better)", "without emb"])

    # 5. exemplar quality --------------------------------------------------------------------------
    import embeddingRetriever as er
    Q = er.model.encode(list(tx), normalize_embeddings=True)
    assert np.allclose(probs_from(Q, er.E_ex, er.ex_rasa), c["emb"], atol=1e-4), "curated scoring mismatch"
    cur_ok = top1(c["emb"], yr)

    rs = np.random.default_rng(1)
    by_rasa = [np.flatnonzero(yr == k) for k in range(K)]
    solo_gap, solo_rand = [], []
    fused_gap, fused_rand = [], []
    for t in range(N_RANDOM):
        pick = np.array([rs.choice(by_rasa[k], 3, replace=False) for k in range(K)])    # (K, 3) row indices
        lab = np.repeat(np.arange(K), 3)
        Pr = probs_from(Q, Q[pick.ravel()], lab)
        keep = np.ones(n, bool)
        keep[pick.ravel()] = False                                                       # never score a phrase on itself
        solo_rand.append(top1(Pr, yr)[keep].mean())
        solo_gap.append(cur_ok[keep].mean())
        if t < N_RANDOM_FUSED:
            cr = dict(c, emb=Pr)
            fused_rand.append(top1(fuse(cr, yr, FULL), yr)[keep].mean())
            fused_gap.append(A[keep].mean())
    solo_rand, solo_gap = np.array(solo_rand), np.array(solo_gap)
    fused_rand, fused_gap = np.array(fused_rand), np.array(fused_gap)
    pct_solo = (solo_rand < solo_gap).mean() * 100
    pct_fused = (fused_rand < fused_gap).mean() * 100

    cent_P = np.zeros((n, K))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Q, yr):
        cen = np.stack([Q[tr][yr[tr] == k].mean(0) for k in range(K)])
        cen /= np.linalg.norm(cen, axis=1, keepdims=True)
        cent_P[te] = probs_from(Q[te], cen, np.arange(K))
    cent_fused = top1(fuse(dict(c, emb=cent_P), yr, FULL), yr).mean()

    sat = {}
    for j in (1, 2, 3):
        sel = np.concatenate([np.flatnonzero(er.ex_rasa == k)[:j] for k in range(K)])
        Pj = probs_from(Q, er.E_ex[sel], er.ex_rasa[sel])
        sat[j] = (top1(Pj, yr).mean(), top1(fuse(dict(c, emb=Pj), yr, FULL), yr).mean())

    ex_tab = pd.DataFrame({
        "solo top-1": {"curated (3/rasa)": cur_ok.mean(), "random 3/rasa (mean)": solo_rand.mean(),
                       "random 3/rasa (best)": solo_rand.max(), "class centroids (OOF)": top1(cent_P, yr).mean()},
        "fused top-1": {"curated (3/rasa)": A.mean(), "random 3/rasa (mean)": fused_rand.mean(),
                        "random 3/rasa (best)": fused_rand.max(), "class centroids (OOF)": cent_fused}})
    satdf = pd.DataFrame(sat, index=["solo top-1", "fused top-1"]).T
    satdf.index = [f"{j} curated phrase(s) per rasa" for j in sat]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(solo_rand, bins=25, color="#9ecae1")
    ax.axvline(cur_ok.mean(), color="#c62828", label=f"curated ({cur_ok.mean():.3f}, percentile {pct_solo:.0f})")
    ax.set_xlabel("solo top-1 of emb"); ax.set_ylabel("random phrase sets"); ax.legend()
    fig.tight_layout()
    html += ["<h3>5. Are the curated phrases representative?</h3>",
             f"<p>Random sets pick 3 training sentences per rasa as exemplars ({N_RANDOM} sets solo, {N_RANDOM_FUSED} through the fusion); "
             "a set is never scored on its own sentences. Centroids are fit out-of-fold.</p>",
             ex_tab.round(3).to_html(), fig_html(fig, "emb_random_phrase_sets"),
             f"<p>Curated phrases beat {pct_solo:.0f}% of random sets solo and {pct_fused:.0f}% when fused.</p>",
             "<h4>Saturation (how many phrases are needed)</h4>", satdf.round(3).to_html()]
    csv.update(pct_solo=pct_solo, pct_fused=pct_fused, centroid_solo=top1(cent_P, yr).mean(), centroid_fused=cent_fused)

    # 6. low-data ------------------------------------------------------------------------------------
    low = {}
    for frac in (0.1, 0.25, 0.5, 1.0):
        g = []
        for r in range(5):
            sel = np.arange(n) if frac == 1.0 else train_test_split(np.arange(n), train_size=frac, stratify=yr, random_state=r)[0]
            cs = {s: c[s][sel] for s in FULL}
            g.append((top1(fuse(cs, yr[sel], FULL, r), yr[sel]).mean(), top1(fuse(cs, yr[sel], NOEMB, r), yr[sel]).mean()))
        g = np.array(g)
        low[f"{int(frac * 100)}% ({int(n * frac)} rows)"] = {"with emb": g[:, 0].mean(), "without emb": g[:, 1].mean(),
                                                           "gain (pts)": (g[:, 0] - g[:, 1]).mean() * 100}
    low = pd.DataFrame(low).T
    table_png(low, "emb_lowdata")
    table_png(ex_tab, "emb_exemplar_comparison")
    table_png(sig, "emb_significance", highlight_max=False)
    html += ["<h3>6. Low-data behaviour</h3><p>Fusion trained on a fraction of the rows (5 stratified subsamples, top-1 on the same rows by CV).</p>",
             low.round(3).to_html()]

    # 7. data-driven summary -------------------------------------------------------------------------
    lo, hi = np.percentile(d1, [2.5, 97.5])
    claims = []
    claims.append(f"Adding emb changes fused top-1 by {(A.mean() - B.mean()) * 100:+.1f} pts (95% CI {lo * 100:+.1f} to {hi * 100:+.1f}; McNemar p = {p_mc:.2f})"
                  + ("; the interval excludes zero." if lo > 0 else "; the interval includes zero, so the gain is not statistically established at this sample size."))
    claims.append(f"It is right on {comp_tab.loc['emb', 'right where clf is wrong']} sentences the classifier misses, raising the oracle top-1 of clf from "
                  f"{oracle(['clf']):.3f} to {oracle(['clf', 'emb']):.3f}.")
    claims.append(f"Probability quality: NLL {cal.loc['NLL (lower better)', 'without emb']:.3f} -> {cal.loc['NLL (lower better)', 'with emb']:.3f}.")
    claims.append(f"The curated phrases outperform {pct_solo:.0f}% of random same-size phrase sets"
                  + (" (supporting that they are better than arbitrary exemplars)." if pct_solo >= 90 else
                     " (so 'most representative' is NOT supported; describe them as an interpretable, training-free prior instead)."))
    html += ["<h3>What the evidence supports</h3><ul>" + "".join(f"<li>{x}</li>" for x in claims) + "</ul>"]

    css = ("<style>body{font-family:sans-serif;margin:24px}table{border-collapse:collapse;margin:6px 0}"
           "td,th{border:1px solid #ccc;padding:3px 8px;font-size:13px}</style>")
    OUT.mkdir(exist_ok=True)
    (OUT / "emb_analysis.html").write_text("<meta charset='utf-8'>" + css + "\n".join(html), encoding="utf-8")
    pd.Series(csv).to_csv(OUT / "emb_analysis.csv", header=["value"])
    print(sig.round(3).to_string())
    print(f"\nMcNemar fixes={b} breaks={cc} p={p_mc:.3f}; seed gain {seed_gain.mean() * 100:+.2f} +- {seed_gain.std() * 100:.2f}")
    print("\n".join(claims))
    print("\n", ex_tab.round(3).to_string(), "\n", satdf.round(3).to_string(), "\n", low.round(3).to_string())


if __name__ == "__main__":
    main()
