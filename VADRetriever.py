# ============ 0. Setup ============
# !pip install sentence-transformers scikit-learn scipy matplotlib pandas

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr

from common import RASAS, PROJECT_DIR, CACHE_DIR, RESULTS_DIR


# ============ 1. Load the NRC lexicon ============
# keep_default_na=False stops pandas turning words like "nan" or "null" into missing values
df = pd.read_csv(PROJECT_DIR / "NRC-VAD-Lexicon-v2.1.txt", sep="\t", keep_default_na=False)

df["term"] = df["term"].str.lower()                 # lowercase so lookups match
nrc = df.drop_duplicates("term").set_index("term")  # one row per word; word is the index
VAD_COLS = ["valence", "arousal", "dominance"]


# ============ 2. Anchor words, grouped by rasa ============
clusters = {
    "Shanta":       ["content", "Peace"],
    "Hasya":        ["Joy", "fun"],
    "Adbhuta":      ["awestruck", "amazement", "astonishment", "surprise"],
    "Raudra":       ["anger", "fury"],
    "Karuna":       ["sad", "melancholy"],
    "Shringara":    ["romance", "love"],
    "Fear/Disgust": ["Fear", "Afraid", "Disgust"],   # merged; the other retrievers split it
    # "Veera":      [...],                           # add your own anchor words here
}
FLOOR, CAP = 0.1, 0.25     # min / max cluster spread, in 0..1 units


def get_points(clusters):
    """Look up each anchor word in NRC. Returns words, their rasa, and VAD in 0..1."""
    words, groups = [], []
    for rasa, ws in clusters.items():
        for w in ws:
            if w.lower() in nrc.index:
                words.append(w); groups.append(rasa)
            else:
                print(f"not in NRC, skipped: {w} ({rasa})")
    pts = nrc.loc[[w.lower() for w in words], VAD_COLS].values.astype(float)
    return words, np.array(groups), (pts + 1) / 2        # NRC is -1..1; move to 0..1


def cluster_stats(pts, groups, floor=FLOOR, cap=CAP):
    """Centroid = mean of the cluster's points. Spread = average distance of members to it."""
    cents, spreads = {}, {}
    for r in dict.fromkeys(groups):
        p = pts[groups == r]
        cents[r] = p.mean(axis=0)
        s = np.linalg.norm(p - cents[r], axis=1).mean()
        spreads[r] = float(np.clip(s, floor, cap))       # never too tight or too wide
    return cents, spreads


def classify(x, cents, spreads, top=3):
    """Rank rasas for one VAD point. Distance is in units of each cluster's spread."""
    d = {r: np.linalg.norm(np.asarray(x) - c) / spreads[r] for r, c in cents.items()}
    return [(str(r), round(float(v), 2)) for r, v in sorted(d.items(), key=lambda t: t[1])[:top]]


words, groups, pts = get_points(clusters)
cents, spreads = cluster_stats(pts, groups)
for r in cents:
    print(f"{r:13} centroid={cents[r].round(2)}  spread={spreads[r]:.2f}")


# ============ 3. Teach a model to imitate NRC (with caching) ============
CACHE = str(CACHE_DIR)
os.makedirs(CACHE, exist_ok=True)

ENC_NAME = "all-mpnet-base-v2"
enc = SentenceTransformer(ENC_NAME)        # downloaded once to the HuggingFace cache (~400 MB)

terms = nrc.index.tolist()
Y = (nrc[VAD_COLS].values.astype(float) + 1) / 2

# --- the slow part: embeddings for all NRC words ---
emb_path = f"{CACHE}/nrc_term_embeddings_{ENC_NAME}.npy"
if os.path.exists(emb_path):
    X = np.load(emb_path)
    assert len(X) == len(terms), "cache doesn't match this lexicon; delete the .npy file"
    print("loaded cached embeddings", X.shape)
else:
    X = enc.encode(terms, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    np.save(emb_path, X)

# --- the trained model: ridge regression + the stretch settings ---
model_path = f"{CACHE}/vad_ridge_{ENC_NAME}.joblib"
if os.path.exists(model_path):
    reg, scale, mean = joblib.load(model_path)
    print("loaded cached ridge model")
else:
    Xtr, Xte, ytr, yte = train_test_split(X, Y, test_size=0.2, random_state=0)
    reg = RidgeCV(alphas=np.logspace(-2, 2, 9)).fit(Xtr, ytr)     # study 80%
    pred = reg.predict(Xte)                                        # exam on the other 20%
    for k, axis in enumerate(VAD_COLS):
        rho = spearmanr(pred[:, k], yte[:, k]).correlation
        print(f"{axis:10} held-out spearman = {rho:.2f}   "
              f"pred std {pred[:, k].std():.2f} vs true std {yte[:, k].std():.2f}")
    scale = yte.std(0) / pred.std(0)
    mean = Y.mean(0)
    reg = RidgeCV(alphas=np.logspace(-2, 2, 9)).fit(X, Y)         # final model on all words
    joblib.dump((reg, scale, mean), model_path)


# ============ 4. Score sentences in NRC style ============
STRETCH = 0.5      # 0 = no stretch (cautious), 1 = full stretch; tune on labeled sentences

def nrc_vad_sentences(sents):
    """List of sentences -> array (n, 3) of valence, arousal, dominance in 0..1."""
    E = enc.encode(list(sents), normalize_embeddings=True)   # sentence -> meaning-numbers
    p = reg.predict(E)                                       # meaning-numbers -> scores
    s = 1 + STRETCH * (scale - 1)                            # blend between no stretch and full
    return np.clip((p - mean) * s + mean, 0, 1)              # un-shrink, keep in 0..1


# ============ 5. Probabilities over all 9 rasas (this plugs into the fusion step) ============
TEMP = 1.0       # lower = more decisive; distances run about 1.5 to 7, so 1.0 is a sane start

def vad_probs(sents):
    """Sentences -> (n, 9) probabilities over RASAS, from sentence VAD + centroid distance."""
    sents = list(sents)
    pts_ = nrc_vad_sentences(sents)
    names = list(cents)
    out = np.zeros((len(sents), len(RASAS)))
    for i, x in enumerate(pts_):
        d = np.array([np.linalg.norm(x - cents[r]) / spreads[r] for r in names])
        p = np.exp(-(d - d.min()) / TEMP)                    # softmax over negative distance
        p /= p.sum()
        for r, pr in zip(names, p):
            if r == "Fear/Disgust":                          # VAD can't tell these apart: split evenly
                out[i, RASAS.index("Bhayanaka")] += pr / 2
                out[i, RASAS.index("Bibhatsa")] += pr / 2
            else:
                out[i, RASAS.index(r)] += pr
    out += 1e-6                                              # unseen rasas (Veera) stay nonzero
    return out / out.sum(axis=1, keepdims=True)


# ============ 6. Plot the anchors (optional) ============
def draw_sphere(ax, c, r, color):
    u, v = np.mgrid[0:2 * np.pi:16j, 0:np.pi:10j]
    ax.plot_wireframe(c[0] + r * np.cos(u) * np.sin(v), c[1] + r * np.sin(u) * np.sin(v),
                      c[2] + r * np.cos(v), color=color, alpha=0.2, linewidth=0.6)

def plot_anchors(words, groups, pts, cents, spreads, sentence_pts=None):
    rasas = list(dict.fromkeys(groups))
    colors = dict(zip(rasas, plt.cm.tab10(np.linspace(0, 1, 10))))
    ax = plt.figure(figsize=(9, 8)).add_subplot(projection="3d")
    for r in rasas:
        m, c = groups == r, colors[r]
        ax.scatter(*pts[m].T, color=c, s=50, alpha=0.8)
        for (x, y, z), name in zip(pts[m], np.array(words)[m]):
            ax.text(x, y, z, name, fontsize=8, color="gray")
        draw_sphere(ax, cents[r], spreads[r], c)
        ax.scatter(*cents[r], color=c, s=320, marker="*", edgecolor="black", linewidth=1.2, zorder=10)
        ax.text(*cents[r], r, fontsize=10, fontweight="bold")
    if sentence_pts is not None:                              # scored sentences as black x's
        ax.scatter(*np.asarray(sentence_pts).T, color="black", marker="x", s=60)
    ax.set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1), title="NRC anchors",
           xlabel="Valence", ylabel="Arousal", zlabel="Dominance")


# ============ 7. Accuracy on your own labeled sentences (when you have them) ============
MERGE = {"Bhayanaka": "Fear/Disgust", "Bibhatsa": "Fear/Disgust"}

def evaluate(labeled, cents, spreads):
    """labeled = [(sentence, rasa), ...]. Prints top-1 and top-3 accuracy."""
    rows = [(s, MERGE.get(r, r)) for s, r in labeled if MERGE.get(r, r) in cents]
    X_ = nrc_vad_sentences([s for s, _ in rows])
    ranked = [[n for n, _ in classify(p, cents, spreads, top=3)] for p in X_]
    top1 = np.mean([rk[0] == t for rk, (_, t) in zip(ranked, rows)])
    top3 = np.mean([t in rk for rk, (_, t) in zip(ranked, rows)])
    print(f"top-1 {top1:.0%}  top-3 {top3:.0%}  (n={len(rows)}, chance top-1 ~{1/len(cents):.0%})")

if __name__ == "__main__":
    from sample_dataset import labeled
    from retriever_eval import run_evaluation

    print(df.head())
    for r in cents:
        print(f"{r:13} centroid={cents[r].round(2)}  spread={spreads[r]:.2f}")
    for s in ["I can't stop smiling today.",
              "I'm terrified of what the results might say.",
              "I feel completely at peace with everything."]:
        x = nrc_vad_sentences([s])[0]
        print(s, "->", x.round(2), "->", classify(x, cents, spreads))
    print(pd.DataFrame(vad_probs(["I'm terrified of what the results might say."]),
                       columns=RASAS).round(2))

    out = RESULTS_DIR / "vad_only"
    out.mkdir(parents=True, exist_ok=True)
    plot_anchors(words, groups, pts, cents, spreads)
    plt.savefig(out / "anchors.png", dpi=150)
    plt.close("all")

    run_evaluation(
        labeled, vad_probs, name="VAD only",
        equivalent=[{"Bhayanaka", "Bibhatsa"}],     # VAD splits these evenly, so don't penalize either
    )