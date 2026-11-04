"""Collect the recommendation-accuracy results for the paper into one report (reads existing result files only).

Run:  python evaluation/make_paper_report.py
-> evaluation/results/paper_report.html, evaluation/results/paper_tables/{T*.csv, T*.png}
Run the evaluation scripts first (rerank/rerank.py, run_retrieval_eval.py, run_hard_search_eval.py, run_agent_eval.py + report.py,
run_e2e_eval.py); a table whose source file is missing is skipped with a note.
"""
import html
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

RERANK_DIR = ROOT / "rerank" / "results"
EVAL_DIR = HERE / "results"
MODES = ("sparse", "dense", "hybrid")
MODE_LABEL = {"sparse": "sparse (BM25)", "dense": "dense", "hybrid": "hybrid (RRF)"}
SYSTEM = {"agent_v1": "agent, original prompt", "agent_v2": "agent, prompt + guard", "keyword": "keyword router (no LLM)",
          "oracle": "oracle (perfect call)"}
# row label in the compact keyword table -> (source group, query name in that group)
KEYWORD_ROWS = [
    ("exact title (sanity check)", "easy", "title"),
    ("title, one typo", "easy", "title (typo)"),
    ("partial title", "target", "partial title"),
    ("title, two typos", "target", "two typos"),
    ("spelling variant", "target", "spelling variant"),
    ("artist + raga", "target", "artist + raga"),
    ("form + raga + artist", "target", "form + raga + artist"),
    ("raga only (many correct)", "set", "raga only"),
    ("artist + form (many correct)", "set", "artist + form"),
    ("tradition + form + raga (many correct)", "set", "tradition + form + raga"),
]


def _read_csv(path):
    return pd.read_csv(path, index_col=0) if path.exists() else None


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _t(tid, title, df, note, shade=None, appendix=False):
    """shade: None | ("col",) | ("row", [[col positions], ...]) -> best value highlighted in the PNG / HTML."""
    return {"id": tid, "title": title, "df": df, "note": note, "shade": shade, "appendix": appendix}


def keyword_tables(hard, easy):
    """-> (compact T3, appendix A3a long target/sanity table, appendix A3b long set table)."""
    src = {"easy": {(r["query"], r["mode"]): r for r in (easy or {}).get("known_item", [])},
           "target": {(r["query"], r["mode"]): r for r in hard["target"]},
           "set": {(r["query"], r["mode"]): r for r in hard["set"]}}
    rows = {}
    for label, group, q in KEYWORD_ROWS:
        row = {}
        for m in MODES:
            r = src[group].get((q, m))
            row[f"nDCG@10 {m}"] = r["ndcg@10"] if r else np.nan
        for m in MODES:
            r = src[group].get((q, m))
            row[f"top-1 {m}"] = (r["recall@1"] if group != "set" else r["top1_relevant"]) if r else np.nan
        rows[label] = row
    compact = pd.DataFrame(rows).T.dropna(how="all")
    compact = compact[~compact.isna().all(axis=1)]
    shade = ("row", [[0, 1, 2], [3, 4, 5]])
    note = ("Left block: nDCG@10; right block: share of queries whose first result is correct (Recall@1 for single-piece queries, "
            "top-1 relevant for many-correct queries). Best per row shaded. The (sanity check) rows query with the song's own title; "
            "the other rows are degraded or underspecified queries.")
    t3 = _t("T3", "Keyword search: sparse vs dense vs hybrid", compact, note, shade)

    long = [dict(r, group="hard") for r in hard["target"]] + [dict(r, group="sanity") for r in src["easy"].values()]
    d = pd.DataFrame(long)
    d["mode"] = d["mode"].map(MODE_LABEL)
    a = d.set_index(["query", "mode"])[["n", "recall@1", "recall@10", "mrr", "ndcg@10"]]
    a.index = [f"{q} / {m}" for q, m in a.index]
    a3a = _t("A3a", "Appendix: single-piece queries, all metrics", a,
             "Full per-mode numbers behind T3 for queries with one correct piece (exact-title rows are the sanity check).",
             appendix=True)
    s = pd.DataFrame(hard["set"])
    s["mode"] = s["mode"].map(MODE_LABEL)
    b = s.set_index(["query", "mode"])[["n", "precision@10", "ndcg@10", "top1_relevant"]]
    b.index = [f"{q} / {m}" for q, m in b.index]
    a3b = _t("A3b", "Appendix: many-correct queries, all metrics", b,
             "Relevance = the song carries every attribute named in the query.", appendix=True)
    return t3, a3a, a3b


def mood_tables(easy):
    mp = easy["mood_playlists"]
    t4 = pd.DataFrame({"precision@10": mp["precision@10"], "nDCG@10": mp["ndcg@10"]})
    t4.index = ["playlist (this system)", "dense search of the mood text", "random songs"]
    out = [_t("T4", f"Mood playlists ({mp['n_queries']} sentences, 6 rasas with ragas)", t4,
              "Relevance = the song's raga is in the paper's map for the sentence's rasa.", ("col",))]

    pr = pd.DataFrame(mp["per_rasa"]).set_index("rasa")
    t4c = pd.DataFrame({"playlist P@10": pr["playlist"], "dense P@10": pr["dense"], "random P@10": pr["random"],
                        "playlist nDCG@10": pr["ndcg_playlist"], "dense nDCG@10": pr["ndcg_dense"],
                        "random nDCG@10": pr["ndcg_random"]})
    out.append(_t("T4c", "Mood playlists by rasa", t4c,
                  "Same measure as T4 split by the sentence's rasa (20 sentences each). Shows whether any rasa is weaker than the average.",
                  ("row", [[0, 1, 2], [3, 4, 5]])))

    mx, div = easy["mixed"], mp["diversity"]
    rows = {
        "Mixed (mood + named raga): requested songs found by the named raga": (mx["requested_entity_hit"], np.nan),
        "Mixed: share of the list that is the requested raga": (mx["requested_share"], np.nan),
        "Mixed: precision of the mood-filled part for the rasa": (mx["mood_section_precision@rasa"], np.nan),
        "Diversity: distinct artists per playlist": (div["artists"], np.nan),
        "Diversity: distinct ragas per playlist": (div["ragas"], np.nan),
        "Diversity: distinct forms per playlist": (div["forms"], np.nan),
        "Diversity: duplicate pieces per playlist": (div["dupes"], np.nan),
    }
    for r in easy["time_boost"]:
        rows[f"Time of day {int(r['hour']):02d}:00: share of songs whose raga fits the hour"] = (r["boost"], r["no boost"])
    t5 = pd.DataFrame(rows, index=["value", "without time boost"]).T
    out.append(_t("T5", "Mixed queries, diversity and time-of-day", t5,
                  f"Mixed: {mx['n_queries']} mood + raga queries. Diversity: mean over {mp['n_queries']} playlists of about "
                  f"{div['size']:.1f} songs each; empty playlists {mp['empty_rate']:.1%}. Time of day: the boost ranks ragas that fit "
                  "the hour higher within a rasa; the last column is the same playlist without it."))
    return out


def build_tables(rerank_dir=RERANK_DIR, eval_dir=EVAL_DIR):
    """-> list of table dicts (see _t); missing sources are skipped. Appendix tables come last."""
    out = []

    cv = _read_csv(rerank_dir / "cv_metrics.csv")
    if cv is not None:
        df = cv[["top1", "top3", "macro_f1"]].rename(columns={"top1": "top-1", "top3": "top-3", "macro_f1": "macro-F1"})
        out.append(_t("T1", "Text to rasa (5-fold out-of-fold)", df,
                      "Every score is out-of-fold, so no sentence is scored by a model that trained on it.", ("col",)))

    ab = _read_csv(rerank_dir / "ablation.csv")
    if ab is not None:
        df = ab[["top1", "top3", "macro_f1"]].rename(columns={"top1": "top-1", "top3": "top-3", "macro_f1": "macro-F1"})
        out.append(_t("T2", "Ablation of the three rasa components (log-linear fusion)", df,
                      "vad = VAD regression, emb = example-phrase matcher, clf = rasa classifier.", ("col",)))

    hard = _read_json(eval_dir / "hard_search_metrics.json")
    easy = _read_json(eval_dir / "retrieval_metrics.json")
    appendix = []
    if hard:
        t3, a3a, a3b = keyword_tables(hard, easy)
        out.append(t3)
        appendix += [a3a, a3b]
    if easy:
        out += mood_tables(easy)

    ag = _read_csv(eval_dir / "agent_comparison.csv")
    if ag is not None:
        cols = [c for c in ("tool_acc", "rewrite_hit3", "entity_acc") if c in ag]
        df = ag[cols].rename(columns={"tool_acc": "right tool", "rewrite_hit3": "mood rewrite (rasa in top 3)",
                                      "entity_acc": "keyword rewrite keeps the name"})
        out.append(_t("T6", "Chat agent: routing and query rewriting (109 cases)", df,
                      "Right tool: the agent's last tool call is an accepted one. Mood rewrite: the rewritten mood's top-3 rasas contain "
                      "the intended rasa. Keyword rewrite: the rewritten query keeps the named raga / artist.", ("col",)))

    e2e = _read_csv(eval_dir / "e2e_comparison.csv")
    if e2e is not None:
        df = e2e[["tool_acc", "mood_precision", "mood_ndcg", "entity_hit"]].rename(
            columns={"tool_acc": "right tool", "mood_precision": "mood precision", "mood_ndcg": "mood nDCG@10",
                     "entity_hit": "named raga/artist found"})
        df.index = [SYSTEM.get(i, i) for i in df.index]
        out.append(_t("T7", "End to end: songs returned for the agent's actual tool calls vs a perfect call", df,
                      "The logged tool calls were replayed through the real pipelines; the oracle makes the correct call with a "
                      "canonical rewrite.", ("col",)))
    return out + appendix


def best_mask(df, shade):
    """Boolean frame marking the cell(s) to highlight, from a table's shade spec."""
    mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    if not shade:
        return mask
    num = df.apply(pd.to_numeric, errors="coerce")
    if shade[0] == "col":
        for c in df.columns:
            if num[c].notna().any():
                mask[c] = num[c] == num[c].max()
    else:
        for group in shade[1]:
            cols = [df.columns[i] for i in group]
            sub = num[cols]
            for idx in df.index:
                row = sub.loc[idx]
                if row.notna().any():
                    mask.loc[idx, cols] = (row == row.max()).to_numpy()
    return mask


def _fmt(x):
    if isinstance(x, (float, np.floating)):
        return "" if np.isnan(x) else f"{x:.3f}"
    return str(x)


def _png(df, path, shade=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mask = best_mask(df, shade)
    cells = [[_fmt(x) for x in r] for r in df.values]
    big = len(df) <= 6                                   # small tables get larger type / taller rows so sizes look alike
    fig, ax = plt.subplots(figsize=(2.4 + 1.3 * df.shape[1], 0.8 + (0.55 if big else 0.36) * (len(df) + 1)))
    ax.axis("off")
    t = ax.table(cellText=cells, colLabels=[str(c) for c in df.columns], rowLabels=[str(i) for i in df.index],
                 loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(12 if big else 10)
    t.scale(1, 1.9 if big else 1.4)
    for (r, c), cell in t.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#eeeeee")
        elif c >= 0 and mask.iloc[r - 1, c]:
            cell.set_facecolor("#c8e6c9")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _html_table(df, shade):
    mask = best_mask(df, shade)
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = []
    for idx, row in df.iterrows():
        cells = "".join(f"<td class='{'best' if mask.loc[idx, c] else ''}'>{html.escape(_fmt(row[c]))}</td>" for c in df.columns)
        body.append(f"<tr><th class='rowh'>{html.escape(str(idx))}</th>{cells}</tr>")
    return f"<table><tr><th></th>{head}</tr>{''.join(body)}</table>"


def render_html(tables):
    css = ("body{font-family:Georgia,serif;max-width:1000px;margin:2em auto;padding:0 1em;color:#222}"
           "table{border-collapse:collapse;margin:.6em 0}th,td{border:1px solid #ccc;padding:4px 10px;text-align:right}"
           "th.rowh,th:first-child{text-align:left}th{background:#f3f3f3}td.best{background:#c8e6c9}"
           ".note{color:#555;font-size:.92em}h2{margin-top:1.8em}")
    parts = [f"<!doctype html><meta charset='utf-8'><title>RagaMood: recommendation accuracy</title><style>{css}</style>",
             "<h1>RagaMood: recommendation accuracy</h1>",
             "<p class='note'>Numbers are copied from the result files of the evaluation scripts.</p>"]
    appendix_started = False
    for t in tables:
        if t["appendix"] and not appendix_started:
            parts.append("<h1>Appendix</h1>")
            appendix_started = True
        parts.append(f"<h2>{t['id']}. {html.escape(t['title'])}</h2>")
        parts.append(_html_table(t["df"], t["shade"]))
        parts.append(f"<p class='note'>{html.escape(t['note'])}</p>")
    return "\n".join(parts)


def main():
    tables = build_tables()
    tdir = EVAL_DIR / "paper_tables"
    tdir.mkdir(parents=True, exist_ok=True)
    for old in list(tdir.glob("*.csv")) + list(tdir.glob("*.png")):      # drop tables from an earlier layout
        old.unlink()
    for t in tables:
        t["df"].to_csv(tdir / f"{t['id']}.csv")
        _png(t["df"], tdir / f"{t['id']}.png", t["shade"])
    (EVAL_DIR / "paper_report.html").write_text(render_html(tables), encoding="utf-8")
    print(f"{len(tables)} tables -> {EVAL_DIR / 'paper_report.html'}")
    for t in tables:
        print(t["id"], t["df"].shape, t["title"])


if __name__ == "__main__":
    main()
