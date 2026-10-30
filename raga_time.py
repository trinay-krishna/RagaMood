"""Time-of-day raga affinity (from the paper): ragas of the current day slot rank a bit higher.

Each slot's window is the union of the two ranges the paper gives (e.g. 4-7am / 3-6am -> 3-7), so
neighbouring slots overlap and an hour can match more than one.
"""
from datetime import datetime

from raaga_search.search import normalize
from rasa_raga_map import search_names

MATCH_PICKS = 2  # round-robin picks per pass for a time-matched raga (others get 1); see playlist.py

# name: (start hour, end hour, ragas); windows may wrap past midnight
SLOTS = {
    "Dawn": (3, 7, ["Lalit", "Ramkali", "Kaligada", "Bhairav"]),
    "Morning": (6, 10, ["Bilawal", "Todi", "Jaunpuri"]),
    "Late morning": (9, 13, ["Khamaj", "Vrindavani Sarang"]),
    "Afternoon": (12, 16, ["Multani", "Madhuvanti"]),
    "Dusk": (15, 19, ["Puriya", "Marwa", "Yaman"]),
    "Evening": (18, 22, ["Bihag", "Bahar", "Rageshree"]),
    "Night": (21, 1, ["Malkauns", "Bageshree"]),
    "Late night": (0, 4, ["Adana", "Sahana"]),
}


def parse_time(s):
    """'HH:MM' or 'HH' -> hour as float (None passes through)."""
    if s is None:
        return None
    h, _, m = str(s).partition(":")
    return int(h) % 24 + int(m or 0) / 60


def current_slots(hour=None):
    if hour is None:
        now = datetime.now()
        hour = now.hour + now.minute / 60
    hour %= 24
    return {n for n, (a, b, _) in SLOTS.items()
            if (a <= hour < b if a < b else hour >= a or hour < b)}


def time_match(raga, hour=None):
    """True if the (project-named) raga belongs to a slot that contains the hour."""
    want = {normalize(r) for n in current_slots(hour) for r in SLOTS[n][2]}
    return any(normalize(n) in want for n in search_names(raga, equivalents=True))
