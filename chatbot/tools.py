"""The only module that talks to the song pipeline: tool schemas + executors.

Backends are imported lazily and can be replaced, so the rest of the chat bot (and its tests) never loads a model.
"""
from . import config


def _schema(name, description, props, required):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": props, "required": required}}}


_N = {"type": "integer", "description": "How many songs (default 8)."}

TOOL_SCHEMAS = [
    _schema("recommend_by_mood",
            "Recommend songs for a feeling only. mood_text is a one-sentence first-person description of the "
            "mood the user wants to hear, e.g. 'I feel happy and energetic.'",
            {"mood_text": {"type": "string"}, "n": _N}, ["mood_text"]),
    _schema("search_catalog",
            "Search songs by catalogue keywords only: raga, artist, title, form, taala, tempo. "
            "query is bare keywords, e.g. 'Sanjay Subrahmanyan kriti Kalyani'.",
            {"query": {"type": "string"}, "n": _N,
             "mode": {"type": "string", "enum": ["hybrid", "dense", "sparse"]}}, ["query"]),
    _schema("recommend_mixed",
            "Songs for a catalogue keyword AND a mood: the named raga/artist/title first, mood-matched songs fill the rest. "
            "emotion is a one-sentence first-person mood; entity is bare keywords.",
            {"emotion": {"type": "string"}, "entity": {"type": "string"}, "n": _N}, ["emotion", "entity"]),
]


def compact(song):
    """Small, LLM-friendly view of a song dict from any of the three pipelines."""
    raga = song.get("raga") or ", ".join(dict.fromkeys(song.get("raagas") or []))
    out = {"title": song.get("title"), "raga": raga, "artists": song.get("artists"),
           "concert": song.get("concert"), "form": song.get("form"), "tradition": song.get("tradition")}
    for k in ("rasa", "section", "time_match"):
        if song.get(k) not in (None, ""):
            out[k] = song[k]
    return {k: v for k, v in out.items() if v not in (None, "", [])}


class Tools:
    """name -> executor. Pass playlist/search/mixed callables to replace the real pipelines."""

    def __init__(self, playlist=None, search=None, mixed=None, hour=None):
        self._fn = {"recommend_by_mood": playlist or self._playlist,
                    "search_catalog": search or self._search,
                    "recommend_mixed": mixed or self._mixed}
        self.hour = hour
        self._index = None

    def _ix(self):
        if self._index is None:
            from hybrid import Index
            self._index = Index()
        return self._index

    def _playlist(self, mood_text, n=config.DEFAULT_N):
        from playlist import make_playlist
        return make_playlist(mood_text, n=n, hour=self.hour)

    def _search(self, query, n=config.DEFAULT_N, mode="hybrid"):
        from hybrid import search
        return search(query, self._ix(), n=n, mode=mode)

    def _mixed(self, emotion, entity, n=config.DEFAULT_N):
        from mixed import mixed_search
        return mixed_search(emotion, entity, n=n, index=self._ix(), hour=self.hour)

    def call(self, name, args):
        """-> {"songs": [...], ...} or {"error": "..."}; never raises for bad model output."""
        if name not in self._fn:
            return {"error": f"unknown tool '{name}'. Use one of: {', '.join(self._fn)}"}
        try:
            res = self._fn[name](**args)
        except TypeError as e:
            return {"error": f"bad arguments for {name}: {e}"}
        songs = [compact(s) for s in res.get("songs", [])]
        out = {"songs": songs, "count": len(songs)}
        if "rasas" in res:
            out["rasas"] = [r["rasa"] for r in res["rasas"]]
        if res.get("entity_matches") == 0:
            out["note"] = "no songs found for the requested keyword; only mood songs shown"
        if not songs:
            out["note"] = "no songs found"
        return out
