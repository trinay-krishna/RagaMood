"""Song counts per mapped raga in the Saraga data -> coverage_report.md.

Run:  python coverage_report.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from rasa_raga_map import RASA_TO_RAGAS, search_names, set_equivalents
from raaga_search import find_songs

DATA_ROOT = ROOT / "data" / "saraga"
TRADITIONS = ("carnatic", "hindustani")
LOW = 2  # <= LOW songs counts as underrepresented


def count(raga):
    found, _ = find_songs(search_names(raga), DATA_ROOT, TRADITIONS)
    return len({s.metadata_path for ss in found.values() for s in ss})


def main():
    set_equivalents("--equivalents" in sys.argv)
    rows = [(rasa, raga, count(raga)) for rasa, ragas in RASA_TO_RAGAS.items() for raga in ragas]
    totals = {r: sum(n for rr, _, n in rows if rr == r) for r in RASA_TO_RAGAS}
    missing = [x for x in rows if x[2] == 0]
    low = [x for x in rows if 0 < x[2] <= LOW]
    ok = [x for x in rows if x[2] > LOW]

    out = ["# Raga coverage in the Saraga dataset", "",
           f"{len(rows)} mapped ragas: {len(missing)} missing, {len(low)} underrepresented "
           f"(1-{LOW} songs), {len(ok)} with {LOW + 1}+ songs. Counts are unique songs across "
           "Carnatic + Hindustani, using the paper's names and same-name spelling variants only (add `--equivalents` to include unverified counterparts).", "",
           "## Songs per rasa", "", "| Rasa | Ragas | Songs | Ragas with 0 songs |", "|---|---|---|---|"]
    for rasa, ragas in RASA_TO_RAGAS.items():
        z = sum(1 for r, _, n in rows if r == rasa and n == 0)
        out.append(f"| {rasa} | {len(ragas)} | {totals[rasa]} | {z} |")

    def section(title, items, aliases=False):
        out.extend(["", f"## {title}", "", "| Rasa | Raga | Songs |" + (" Spellings tried |" if aliases else ""),
                    "|---|---|---|" + ("---|" if aliases else "")])
        for rasa, raga, n in sorted(items, key=lambda x: (x[2], x[0], x[1])):
            out.append(f"| {rasa} | {raga} | {n} |" + (f" {', '.join(search_names(raga))} |" if aliases else ""))

    section("Missing (0 songs)", missing, aliases=True)
    section(f"Underrepresented (1-{LOW} songs)", low)
    section(f"Adequate ({LOW + 1}+ songs)", ok)
    out.extend(["", "## Notes", "",
                "- A zero can mean a spelling mismatch rather than true absence; supply alternate "
                "spellings and add them to `RAGA_VARIANTS` in `rasa_raga_map.py`.",
                f"- The {LOW}-song cutoff is a judgement call, not a statistical threshold.",
                "- Songs are Saraga metadata entries, so one raga can span several concerts or one concert "
                "can contribute several songs."])
    (ROOT / "coverage_report.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("wrote coverage_report.md")


if __name__ == "__main__":
    main()
