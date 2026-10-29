"""Rasa -> raga mapping (from the paper), plus the spellings used to find each raga in Saraga.

Keys of RASA_TO_RAGAS use the rasa names in common.RASAS (Haasya->Hasya, Shantha->Shanta,
Adhbhutha->Adbhuta). Raudra, Bhayanaka and Bibhatsa have no ragas.

Matching policy (see ALIASES.md): by default a raga is searched under the paper's own name and
spelling variants of that same name only. "/" and "–" entries are split into the paper's own parts.
RAGA_EQUIVALENTS holds cross-name / cross-tradition counterparts that the paper does NOT state; they
are off unless enabled with set_equivalents(True) (the --equivalents flag of the commands).
"""
from common import RASAS

RASA_TO_RAGAS = {
    "Adbhuta": ["Abheri/Bhimpalasi", "Arabhi", "Desh", "Hindola", "Malayamarutham"],
    "Hasya": ["Aathana", "Kunthalavarali", "Reethigowla", "Shankarabharanam"],
    "Karuna": ["Ahibhairav", "Bageshri", "Kanada", "Lalith", "Madhuvanti", "Punnagavarali",
               "Shivaranjani", "Shubhapanthuvarali"],
    "Shanta": ["Kalavathi/Valachi", "Mayamalavagowla", "Sama", "Shuddha Saveri – Durga",
               "Sindhu Bhairavi", "Yadhukula Kambhodhi"],
    "Shringara": ["Behaag", "Brindavani", "Kalyani", "Kamas", "Kapi", "Karaharapriya", "Pahaadi",
                  "YamanKalyani"],
    "Veera": ["Kedaragowla", "Madhyamavathi", "Meghamalhaar", "Mohana"],
}

# Spelling variants of the same name (transliteration differences). Always used.
RAGA_VARIANTS = {
    "Abheri/Bhimpalasi": ["Bhimpalas"],
    "Desh": ["Des"],
    "Hindola": ["Hindolam"],
    "Malayamarutham": ["Malayamarutam"],
    "Aathana": ["Athana"],
    "Kunthalavarali": ["Kuntalavarali"],
    "Reethigowla": ["Riti Gaula", "Ritigaula"],
    "Shankarabharanam": ["Sankarabharanam"],
    "Ahibhairav": ["Ahir Bhairav"],
    "Bageshri": ["Bageshree"],
    "Lalith": ["Lalitha", "Lalita"],
    "Punnagavarali": ["Punnaga Varali"],
    "Shubhapanthuvarali": ["Shubhapantuvarali", "Subhapantuvarali"],
    "Kalavathi/Valachi": ["Kalavati"],
    "Mayamalavagowla": ["Mayamalavagaula"],
    "Sindhu Bhairavi": ["Sindhubhairavi"],
    "Yadhukula Kambhodhi": ["Yadukula Kambhoji", "Yadukula Kambodhi"],
    "Behaag": ["Behag"],
    "Pahaadi": ["Pahadi"],
    "YamanKalyani": ["Yaman Kalyan", "Yaman Kalyani"],
    "Kedaragowla": ["Kedaragaula"],
    "Karaharapriya": ["Kharaharapriya"],
    "Madhyamavathi": ["Madyamavati", "Madhyamavati"],
    "Meghamalhaar": ["Megh Malhar", "Meghamalhar"],
    "Mohana": ["Mohanam"],
}

# Counterparts in another name or tradition that the paper does not state. Off by default.
RAGA_EQUIVALENTS = {
    "Ahibhairav": ["Chakravakam"],
    "Hindola": ["Hindol"],
    "Lalith": ["Lalit", "Lalat"],
    "Behaag": ["Bihag"],
    "YamanKalyani": ["Yaman"],
    "Meghamalhaar": ["Megh"],
}

_unknown = set(RASA_TO_RAGAS) - set(RASAS)
assert not _unknown, _unknown

_use_equivalents = False


def set_equivalents(on):
    global _use_equivalents
    _use_equivalents = bool(on)


def search_names(raga, equivalents=None):
    """All spellings to look up in Saraga for one raga entry."""
    names = [raga]
    for sep in ("/", "–"):
        names = [p.strip() for n in names for p in n.split(sep)]
    names += RAGA_VARIANTS.get(raga, [])
    if _use_equivalents if equivalents is None else equivalents:
        names += RAGA_EQUIVALENTS.get(raga, [])
    return list(dict.fromkeys(names))


def rasa_for_raga(raga):
    return next(r for r, rs in RASA_TO_RAGAS.items() if raga in rs)
