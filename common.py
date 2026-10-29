from pathlib import Path

# canonical order: must be identical in every retriever
RASAS = ["Shringara", "Hasya", "Karuna", "Raudra", "Veera",
         "Bhayanaka", "Bibhatsa", "Adbhuta", "Shanta"]

PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"     # one subfolder per retriever
CACHE_DIR = PROJECT_DIR / "cache"
