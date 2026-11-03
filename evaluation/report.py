"""Score the agent log and the retrieval metrics -> evaluation/results/{report.html, *.csv, figures/*.png}.

Run:  python evaluation/report.py [--log evaluation/results/agent_log.jsonl]
Re-renders from the logs; the LLM is not called.
"""
import argparse
import base64
import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rasa_compare"))
sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import RASAS
from evaluation.cases import load_cases
from evaluation.metrics import TOOLS, build_vocab, keyword_router, mood_text, score_turns, summarize

OUT = HERE / "results"
FIG = OUT / "figures"
PAIR = ["#0072B2", "#E69F00", "#009E73"]
SHORT = {"recommend_by_mood": "mood", "search_catalog": "catalog", "recommend_mixed": "mixed",
         "none": "none", "clarify": "clarify"}


def embed(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def table_png(df, name, fmt="{:.3f}"):
    FIG.mkdir(parents=True, exist_ok=True)
    cells = [[fmt.format(x) if isinstance(x, (float, np.floating)) and not np.isnan(x) else ("" if isinstance(x, float) else str(x))
              for x in r] for r in df.values]
    fig, ax = plt.subplots(figsize=(1.8 + 1.2 * df.shape[1], 0.5 + 0.35 * (len(df) + 1)))
    ax.axis("off")
    t = ax.table(cellText=cells, colLabels=list(df.columns), rowLabels=[str(i) for i in df.index], loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(10)
    t.scale(1, 1.4)
    for (r, _), cell in t.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#eeeeee")
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def last_turns(log, cases):
    byid = {r["id"]: r for r in log}
    keep = [c for c in cases if c["id"] in byid]
    return keep, [byid[c["id"]]["turns"][-1] for c in keep]


def agent_section(logs):
    """logs: {label: path}, the last one is the primary run (detailed tables); all are compared with the baseline."""
    from pipeline import _scorer

    def probs(turns_):
        texts = [mood_text(t) for t in turns_]
        P = [None] * len(texts)
        idx = [i for i, x in enumerate(texts) if x]
        if idx:
            for i, p in zip(idx, _scorer([texts[i] for i in idx])):
                P[i] = p
        return P

    scored = {}
    for label, path in logs.items():
        log = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        cs, ts = last_turns(log, load_cases())
        scored[label] = (cs, ts, score_turns(cs, ts, probs(ts), RASAS))
    primary = list(logs)[-1]
    cases, turns, llm = scored[primary]

    from raaga_search.registry import LOADERS
    from pipeline import DATA_ROOT, TRADITIONS
    from rasa_raga_map import RASA_TO_RAGAS, search_names
    songs = [s for t in TRADITIONS for s in LOADERS[t](DATA_ROOT).load_songs()]
    vocab = build_vocab(songs, [n for rs in RASA_TO_RAGAS.values() for r in rs for n in search_names(r, equivalents=True)])
    base_turns = [keyword_router(c["turns"][-1], vocab) for c in cases]
    base = score_turns(cases, base_turns, probs(base_turns), RASAS)

    OUT.mkdir(exist_ok=True)
    llm.to_csv(OUT / "agent_scored.csv", index=False, encoding="utf-8-sig")
    s_llm, s_base = summarize(llm, "category"), summarize(base, "category")
    s_llm.to_csv(OUT / "agent_metrics.csv")
    s_base.to_csv(OUT / "agent_metrics_keyword_baseline.csv")
    html = ["<h2>Agent evaluation</h2>",
            f"<p>{len(cases)} labelled cases (author-written). Agent = Qwen3-8B via Ollama; baseline = regex/vocabulary router with no query rewriting. "
            "The reply after the tool result is stubbed in this run, so latency covers routing + rewriting + retrieval only.</p>",
            "<h3>LLM agent</h3>", s_llm.round(3).to_html(), "<h3>Keyword baseline</h3>", s_base.round(3).to_html()]
    table_png(s_llm[["n", "tool_acc", "rewrite_hit1", "rewrite_hit3", "entity_acc", "error_rate"]], "agent_metrics_table")

    # before / after comparison
    summ = {lab: summarize(sc[2], "category") for lab, sc in scored.items()}
    summ["keyword baseline"] = s_base
    cmp_cols = ["tool_acc", "rewrite_hit3", "entity_acc", "fabricated_rate", "retried_rate"]
    cmp = pd.DataFrame({lab: S.loc["ALL", cmp_cols] for lab, S in summ.items()}).T
    cmp.to_csv(OUT / "agent_comparison.csv")
    table_png(cmp, "agent_comparison_table")
    html += ["<h3>Before / after (all categories)</h3>", cmp.round(3).to_html()]

    # per-category tool accuracy
    cats = [c for c in s_llm.index if c != "ALL"] + ["ALL"]
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(cats))
    labs = list(summ)
    w = 0.8 / len(labs)
    cols = PAIR + ["#999999"]
    for k, lab in enumerate(labs):
        ax.bar(x + (k - (len(labs) - 1) / 2) * w, summ[lab].loc[cats, "tool_acc"], w, label=lab, color=cols[k % len(cols)])
    ax.set_xticks(x, cats, rotation=20, ha="right"); ax.set_ylim(0, 1.05); ax.set_ylabel("tool-selection accuracy")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=len(labs), frameon=False)
    html += ["<h3>Tool selection by category</h3>", embed(fig, "agent_tool_accuracy")]

    # confusion
    def conf(df):
        gold = [r.pred_tool if r.tool_ok else r.gold_tool.split("|")[0] for r in df.itertuples()]
        m = pd.crosstab(pd.Categorical(gold, TOOLS), pd.Categorical(df["pred_tool"], TOOLS), dropna=False)
        return m
    m = conf(llm)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(m.values, cmap="Blues")
    ax.set_xticks(range(len(TOOLS)), [SHORT[t] for t in TOOLS], rotation=30, ha="right")
    ax.set_yticks(range(len(TOOLS)), [SHORT[t] for t in TOOLS])
    ax.set_xlabel("agent chose"); ax.set_ylabel("expected")
    for i in range(len(TOOLS)):
        for j in range(len(TOOLS)):
            ax.text(j, i, m.values[i, j], ha="center", va="center", color="white" if m.values[i, j] > m.values.max() / 2 else "black")
    fig.tight_layout()
    html += ["<h3>Routing confusion matrix (LLM agent)</h3>", embed(fig, "agent_confusion")]

    # rewrite hit rates
    rc = [c for c in cats if c != "ALL" and s_llm.loc[c, "rewrite_hit1"] == s_llm.loc[c, "rewrite_hit1"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(rc))
    w = 0.2
    for k, (lab, S, col) in enumerate([("LLM top-1", s_llm, PAIR[0]), ("LLM top-3", s_llm, "#56B4E9"),
                                      ("baseline top-1", s_base, PAIR[1]), ("baseline top-3", s_base, "#F0C27B")]):
        ax.bar(x + (k - 1.5) * w, S.loc[rc, "rewrite_hit1" if "top-1" in lab else "rewrite_hit3"], w, label=lab, color=col)
    ax.set_xticks(x, rc); ax.set_ylim(0, 1.05); ax.set_ylabel("intended rasa in classifier's top-k of the rewritten mood")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False)
    html += ["<h3>Does the rewritten mood express the intended rasa?</h3>", embed(fig, "agent_rewrite_hit")]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    groups = [llm.loc[llm.category == c, "latency"].dropna().values for c in cats[:-1]]
    ax.boxplot(groups, tick_labels=cats[:-1]); ax.set_ylabel("seconds per turn (CPU)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    html += ["<h3>Latency</h3>", embed(fig, "agent_latency")]

    bad = llm[~llm["tool_ok"]][["id", "gold_tool", "pred_tool", "mood_text", "entity_text"]]
    html += [f"<h3>Wrong tool choices ({len(bad)})</h3>", bad.to_html(index=False)]
    badr = llm[(llm.get("rewrite_hit3") == False)][["id", "mood_text"]]
    html += [f"<h3>Rewrites that miss the intended rasa in top-3 ({len(badr)})</h3>", badr.to_html(index=False)]
    return html


def retrieval_section(path):
    r = json.loads(path.read_text(encoding="utf-8"))
    html = ["<h2>Retrieval evaluation</h2>"]
    ki = pd.DataFrame(r["known_item"])
    html += ["<h3>Known-item search (hybrid.py)</h3><p>Query built from a song's metadata; hit = a song with the same title is returned.</p>",
             ki.round(3).to_html(index=False)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    for ax, metric in zip(axes, ("recall@10", "mrr", "ndcg@10")):
        p = ki.pivot(index="query", columns="mode", values=metric)[["sparse", "dense", "hybrid"]]
        p.plot.bar(ax=ax, color=PAIR, rot=15, width=0.8)
        ax.set_ylim(0, 1.05); ax.set_ylabel(metric); ax.set_xlabel("")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
    fig.tight_layout()
    html += [embed(fig, "retrieval_known_item")]
    table_png(ki.set_index(["mode", "query"])[["recall@1", "recall@5", "recall@10", "mrr", "ndcg@10"]], "retrieval_known_item_table")
    html += ["<h4>Attribute queries (artist + form)</h4>", pd.DataFrame(r["attribute"]).round(3).to_html(index=False)]

    mp = r["mood_playlists"]
    html += [f"<h3>Mood playlists</h3><p>{mp['n_queries']} dataset sentences, 10 songs each. Precision = share of songs whose raga is in the "
             "paper's raga list for the sentence's rasa (a consistency measure, since the pipeline uses the same map). "
             f"Empty-result rate {mp['empty_rate']:.3f}. Mean diversity per playlist: {mp['diversity']}.</p>"]
    prec = pd.Series(mp["precision@10"])
    pr = pd.DataFrame(mp["per_rasa"]).set_index("rasa")
    if "ndcg@10" in mp:
        nd = pd.Series(mp["ndcg@10"])
        html += ["<h4>nDCG@10 (binary relevance: raga in the sentence's rasa)</h4>", nd.round(3).to_frame("ndcg@10").to_html()]
        fig, ax = plt.subplots(figsize=(5, 3.4))
        ax.bar(["playlist", "dense search", "random"], nd[["playlist", "dense", "random"]], color=[PAIR[0], PAIR[1], "#999999"])
        ax.set_ylim(0, 1.05); ax.set_ylabel("nDCG@10")
        fig.tight_layout()
        html += [embed(fig, "retrieval_mood_ndcg")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1, 1.8]})
    axes[0].bar(["playlist", "dense search", "random"], prec[["playlist", "dense", "random"]], color=[PAIR[0], PAIR[1], "#999999"])
    axes[0].set_ylim(0, 1.05); axes[0].set_ylabel("precision@10"); axes[0].tick_params(axis="x", rotation=15)
    pr[["playlist", "dense", "random"]].plot.bar(ax=axes[1], color=[PAIR[0], PAIR[1], "#999999"], rot=20, width=0.8)
    axes[1].set_ylim(0, 1.05); axes[1].set_xlabel("")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
    fig.tight_layout()
    html += [embed(fig, "retrieval_mood_precision"), pr.round(3).to_html()]

    mx = r["mixed"]
    html += ["<h3>Mixed queries</h3>", pd.Series(mx).to_frame("value").round(3).to_html()]

    tb = pd.DataFrame(r["time_boost"]).set_index("hour")
    fig, ax = plt.subplots(figsize=(6, 3.4))
    tb[["boost", "no boost"]].plot.bar(ax=ax, color=PAIR[:2], rot=0, width=0.7)
    ax.set_xlabel("hour of day"); ax.set_ylabel("share of time-matched songs")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, frameon=False)
    fig.tight_layout()
    html += ["<h3>Time-of-day boost</h3><p>Share of songs in the playlist whose raga belongs to the hour's slot. "
             "Only 5 mapped ragas can match, so absolute values are small.</p>", embed(fig, "retrieval_time_boost"), tb.round(3).to_html()]
    return html


def main():
    ap = argparse.ArgumentParser()
    a = ap.parse_args()
    html = []
    names = [("before: original prompt", "agent_log_v1.jsonl"), ("prompt only", "agent_log_v2np.jsonl"),
             ("prompt + guard", "agent_log_v2.jsonl")]
    logs = {lab: OUT / f for lab, f in names if (OUT / f).exists()}
    if logs:
        html += agent_section(logs)
    else:
        print("no agent log: skipped")
    rp = OUT / "retrieval_metrics.json"
    if rp.exists():
        html += retrieval_section(rp)
    else:
        print("no retrieval metrics: skipped")
    html += ["<h2>Limitations</h2><ul>"
             "<li>Agent test cases are author-written (about 110); labels should be reviewed and the set is small, so differences of a few points are noise.</li>"
             "<li>Rewrite check uses the project's own rasa classifier as the judge; the 450-row rasa dataset is synthetic with single-author labels.</li>"
             "<li>Mood-playlist precision uses the paper's rasa-to-raga map, the same map the pipeline retrieves with, so it measures consistency, not independent ground truth.</li>"
             "<li>Saraga has 359 songs and no per-song emotion labels; no human ratings are included yet.</li></ul>"]
    css = ("<style>body{font-family:sans-serif;margin:24px}table{border-collapse:collapse;margin:6px 0}"
           "td,th{border:1px solid #ccc;padding:3px 8px;font-size:13px}img{max-width:100%}</style>")
    (OUT / "report.html").write_text("<meta charset='utf-8'>" + css + "\n".join(html), encoding="utf-8")
    print("wrote", OUT / "report.html")


if __name__ == "__main__":
    main()
