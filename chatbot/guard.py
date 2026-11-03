"""Cheap checks on a reply that came back without any tool call (pure functions, no project imports)."""
import re

NUDGE = ("You must call a tool for this request: search_catalog for raga/artist/title/form keywords, "
         "recommend_by_mood for a feeling, recommend_mixed for both. Do not list or invent songs yourself. "
         "Call the tool now.")
SAFE_REPLY = ("Sorry, I couldn't look that up just now. Could you tell me the mood you're in, "
              "or a raga, artist or form you'd like to hear?")

_ITEM = r"^\s*(?:\d+[.)]\s*|[-*•]\s*)?"
_DASH = r"\s[-–—]\s"
_QUOTED = re.compile(_ITEM + r"[\"“'].+[\"”']\s*[-–—]\s*\S+")
_NUMBERED = re.compile(r"^\s*\d+[.)]\s+.+" + _DASH + r".+" + _DASH + r".+")
_FEELING = re.compile(
    r"\b(sad|happy|angry|furious|calm|relax\w*|scared|terrified|anxious|love|lonely|miss|joy\w*|brave|courage|"
    r"amazed|awe\w*|disgust\w*|stress\w*|feel\w*|mood|heartbroken|cheer\w*|upbeat|energ\w*|peace\w*|romantic|"
    r"worried|down|excited|bored|exhausted|nostalgic|smil\w*|laugh\w*|cry\w*|grie\w*)\b", re.I)


def looks_like_song_list(reply):
    """True for two or more lines shaped like '1. "Title" - Raga - Artist' (a list of songs written by the model)."""
    lines = [l for l in reply.splitlines() if l.strip()]
    return sum(bool(_QUOTED.match(l) or _NUMBERED.match(l)) for l in lines) >= 2


def asks_question(reply):
    return reply.strip().endswith("?")


def has_feeling(text):
    return bool(_FEELING.search(text))
