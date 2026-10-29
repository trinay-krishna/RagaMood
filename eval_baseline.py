import sys
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(encoding="utf-8")

RASAS = ["Shringara", "Hasya", "Karuna", "Raudra", "Veera",
         "Bhayanaka", "Bibhatsa", "Adbhuta", "Shanta"]
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

rasa_examples = {
    "Adbhuta":   ["I want a song that is amazing", "something awe-inspiring that blows my mind", "I feel amazed and full of wonder"],
    "Karuna":    ["I'm feeling sad and heartbroken", "play sad, emotional songs", "I miss someone and I feel empty"],
    "Hasya":     ["I want something fun and funny", "a cheerful song that makes me laugh", "I'm in a playful, joking mood"],
    "Shanta":    ["I want to relax and feel calm", "peaceful music to meditate to", "I need quiet and stillness"],
    "Veera":     ["I want a powerful song that pumps me up", "music to feel brave and determined", "give me an anthem to push through this"],
    "Raudra":    ["I'm furious and want angry music", "something intense to vent my rage", "I'm so angry I can't think straight"],
    "Shringara": ["I want a romantic song", "music for being in love", "I'm missing my partner and feeling tender"],
    "Bhayanaka": ["I feel scared and anxious", "something eerie and tense", "I'm terrified and can't relax"],
    "Bibhatsa":  ["I feel disgusted and sick of everything", "I'm repulsed by what I saw today", "I can't stand how corrupt it all is"],
}
ex_pairs = [(r, t) for r in RASAS for t in rasa_examples[r]]
E_ex = model.encode([t for _, t in ex_pairs], normalize_embeddings=True)
ex_rasa = np.array([RASAS.index(r) for r, _ in ex_pairs])


def emb_probs(sents, pool="max", temperature=0.05):
    Q = model.encode(list(sents), normalize_embeddings=True)
    S = Q @ E_ex.T
    scores = np.zeros((len(Q), len(RASAS)))
    for k in range(len(RASAS)):
        cols = S[:, ex_rasa == k]
        scores[:, k] = cols.max(axis=1) if pool == "max" else cols.mean(axis=1)
    z = scores / temperature
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True), S


df = pd.read_csv(sys.argv[1], keep_default_na=False)
P, S = emb_probs(df["user_text"])

# leakage
best = S.max(axis=1)
flag = np.where(best > 0.85)[0]
print(f"LEAKAGE: {len(flag)} of {len(df)} rows have sim > 0.85 to an example phrase")
for i in flag:
    j = S[i].argmax()
    print(f"  {best[i]:.2f} {df.user_text[i]!r} ~ [{ex_pairs[j][0]}] {ex_pairs[j][1]!r}")
print(f"  max sim overall {best.max():.2f}, mean {best.mean():.2f}")

emo = df["primary_rasa"] != "None"
y = df.loc[emo, "primary_rasa"].map(RASAS.index).to_numpy()
Pe = P[emo.to_numpy()]
order = np.argsort(-Pe, axis=1)
top1 = (order[:, 0] == y).mean()
top2 = np.mean([y[i] in order[i, :2] for i in range(len(y))])
print(f"\nPRIMARY (emotional rows, n={len(y)}): top-1 {top1:.3f}   top-2 {top2:.3f}   chance 0.111")

sec = df.loc[emo, "secondary_rasas"].tolist()
hit2 = hit3 = n = 0
for i, s in enumerate(sec):
    if not s:
        continue
    n += 1
    ss = {RASAS.index(x) for x in s.split("|")}
    hit2 += bool(ss & set(order[i, :2]))
    hit3 += bool(ss & set(order[i, :3]))
print(f"SECONDARY (rows with one, n={n}): any in top-2 {hit2/n:.3f}   any in top-3 {hit3/n:.3f}")

cm = np.zeros((9, 9), int)
for t, p in zip(y, order[:, 0]):
    cm[t, p] += 1
print("\nPer-class recall / precision:")
for k, r in enumerate(RASAS):
    rec = cm[k, k] / max(cm[k].sum(), 1)
    prec = cm[k, k] / max(cm[:, k].sum(), 1)
    print(f"  {r:10s} recall {rec:.2f}  precision {prec:.2f}")
print("\nConfusion matrix (rows=true, cols=pred), order:", RASAS)
print(pd.DataFrame(cm, index=RASAS, columns=[r[:4] for r in RASAS]).to_string())

mp = P.max(axis=1)
print(f"\nMax prob: emotional mean {mp[emo.to_numpy()].mean():.2f}, None mean {mp[~emo.to_numpy()].mean():.2f}")
