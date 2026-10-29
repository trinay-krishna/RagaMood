import unicodedata
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

from .base import Song
from .registry import LOADERS


def normalize(name: str) -> str:
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return "".join(c for c in s.lower() if c.isalnum())


def find_songs(raagas: Iterable[str], dataset_root, traditions: Sequence[str] = ("carnatic",)
               ) -> Tuple[Dict[str, List[Song]], List[str]]:
    """Return ({query raaga: songs}, [queries with no match])."""
    index = defaultdict(list)
    for t in traditions:
        for song in LOADERS[t](dataset_root).load_songs():
            for key in {normalize(n) for n in song.raagas}:
                index[key].append(song)
    result, missing = {}, []
    for q in raagas:
        songs = index.get(normalize(q), [])
        result[q] = songs
        if not songs:
            missing.append(q)
    return result, missing
