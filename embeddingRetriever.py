import numpy as np
from sentence_transformers import SentenceTransformer

from common import RASAS

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

rasa_examples = {
    "Adbhuta":   ["I want a song that is amazing",
                  "something awe-inspiring that blows my mind",
                  "I feel amazed and full of wonder"],
    "Karuna":    ["I'm feeling sad and heartbroken",
                  "play sad, emotional songs",
                  "I miss someone and I feel empty"],
    "Hasya":     ["I want something fun and funny",
                  "a cheerful song that makes me laugh",
                  "I'm in a playful, joking mood"],
    "Shanta":    ["I want to relax and feel calm",
                  "peaceful music to meditate to",
                  "I need quiet and stillness"],
    "Veera":     ["I want a powerful song that pumps me up",
                  "music to feel brave and determined",
                  "give me an anthem to push through this"],
    "Raudra":    ["I'm furious and want angry music",
                  "something intense to vent my rage",
                  "I'm so angry I can't think straight"],
    "Shringara": ["I want a romantic song",
                  "music for being in love",
                  "I'm missing my partner and feeling tender"],
    "Bhayanaka": ["I feel scared and anxious",
                  "something eerie and tense",
                  "I'm terrified and can't relax"],
    "Bibhatsa":  ["I feel disgusted and sick of everything",
                  "I'm repulsed by what I saw today",
                  "I can't stand how corrupt it all is"],
}
assert set(rasa_examples) == set(RASAS)

# flatten: one row per example phrase, remembering which rasa it belongs to
ex_pairs = [(r, t) for r in RASAS for t in rasa_examples[r]]
E_ex = model.encode([t for _, t in ex_pairs], normalize_embeddings=True)       # (n_examples, dim)
ex_rasa = np.array([RASAS.index(r) for r, _ in ex_pairs])

def emb_probs(sents, pool="max", temperature=0.05):
    """Sentences -> (n, 9) probabilities over RASAS, from similarity to the example phrases."""
    Q = model.encode(list(sents), normalize_embeddings=True)                    # (n, dim)
    S = Q @ E_ex.T                                                              # cosine sims (n, n_examples)
    scores = np.zeros((len(Q), len(RASAS)))
    for k in range(len(RASAS)):
        cols = S[:, ex_rasa == k]
        scores[:, k] = cols.max(axis=1) if pool == "max" else cols.mean(axis=1)
    z = scores / temperature
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)

def leakage_check(labeled, thresh=0.85):
    """Flags test sentences that are near-copies of a rasa example phrase (would inflate accuracy)."""
    Q = model.encode([s for s, _ in labeled], normalize_embeddings=True)
    S = Q @ E_ex.T
    flagged = 0
    for i, (s, _) in enumerate(labeled):
        j = S[i].argmax()
        if S[i, j] > thresh:
            flagged += 1
            print(f"sim {S[i, j]:.2f} | test: {s[:60]!r} ~ example [{ex_pairs[j][0]}]: {ex_pairs[j][1]!r}")
    print(f"{flagged} of {len(labeled)} test sentences are near-copies of an example")


if __name__ == "__main__":
    from sample_dataset import labeled
    from retriever_eval import run_evaluation
    leakage_check(labeled)
    run_evaluation(labeled, lambda s: emb_probs(s, pool="max"),  name="Embeddings (MiniLM, max)")
    run_evaluation(labeled, lambda s: emb_probs(s, pool="mean"), name="Embeddings (MiniLM, mean)")