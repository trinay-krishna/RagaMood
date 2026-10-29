import argparse
import json

from .registry import LOADERS
from .search import find_songs


def main():
    p = argparse.ArgumentParser(description="Find Saraga songs by raaga")
    p.add_argument("raagas", nargs="+")
    p.add_argument("--root", default="data/saraga", help="path of the cloned saraga repo")
    p.add_argument("--tradition", nargs="+", default=["carnatic"], choices=list(LOADERS))
    p.add_argument("--json", action="store_true")
    a = p.parse_args()
    res, missing = find_songs(a.raagas, a.root, a.tradition)
    if a.json:
        print(json.dumps({r: [s.to_dict() for s in v] for r, v in res.items()}, indent=2, ensure_ascii=False))
    else:
        for r, songs in res.items():
            print(f"{r} ({len(songs)})")
            for s in songs:
                print(f"  {s.title} | {', '.join(s.artists)} | {s.concert}")
    if missing:
        print("No match for:", ", ".join(missing))


main()
