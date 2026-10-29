import json
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List


@dataclass
class Song:
    title: str
    tradition: str
    raagas: List[str]  # all names/aliases, raw
    artists: List[str]
    concert: str
    metadata_path: Path
    mbid: str = ""
    audio_path: Path = None  # set only if the audio was downloaded
    form: str = ""
    taala: str = ""
    laya: str = ""
    length: float = 0.0  # ms, as in the metadata

    def to_dict(self):
        d = {k: (str(v) if isinstance(v, Path) else v) for k, v in self.__dict__.items()}
        return d


class TraditionLoader(ABC):
    """Reads one tradition's metadata. Subclass and set the class attributes."""

    name: str
    subdir: str  # folder under dataset/
    raaga_key: str  # JSON field holding the raaga list

    def __init__(self, dataset_root):
        self.root = Path(dataset_root) / "dataset" / self.subdir

    def raaga_names(self, entries) -> List[str]:
        names = []
        for e in entries or []:
            for key in ("common_name", "name"):
                v = e.get(key) if isinstance(e, dict) else e
                if v:
                    names.extend(p for p in str(v).split("/") if p.strip())
        return names

    @staticmethod
    def _names(*entries) -> str:
        out = []
        for entry in entries:
            for e in entry or []:
                v = (e.get("common_name") or e.get("name")) if isinstance(e, dict) else e
                if v:
                    out.append(str(v))
        return ", ".join(out)

    def load_songs(self) -> Iterable[Song]:
        for path in sorted(self.root.rglob("*.json")):
            meta = json.loads(path.read_text(encoding="utf8"))
            audio = path.with_suffix(".mp3")
            yield Song(
                title=meta.get("title", path.stem),
                tradition=self.name,
                raagas=self.raaga_names(meta.get(self.raaga_key)),
                artists=[a.get("artist", {}).get("name", "") for a in meta.get("artists", [])],
                concert=", ".join(c.get("title", "") for c in meta.get("concert", [])),
                metadata_path=path,
                mbid=meta.get("mbid", ""),
                audio_path=audio if audio.exists() else None,
                form=self._names(meta.get("form"), meta.get("forms")),
                taala=self._names(meta.get("taala"), meta.get("taals")),
                laya=self._names(meta.get("layas")),
                length=float(meta.get("length") or 0),
            )
