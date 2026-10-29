# Rasa classifier comparison (synthetic data, 5-fold stratified CV, seed 0)
All numbers are out-of-fold on LLM-written, single-author-labelled data: optimistic, not evidence for real users.

| model | top-1 | top-2 | macro-F1 | secondary hit top-2 / top-3 |
|---|---|---|---|---|
| zero-shot | .589±.065 | .773±.042 | .576±.073 | .34 / .49 |
| LogReg (frozen MiniLM) | .749±.050 | .880±.021 | .745±.052 | .47 / .66 |
| SetFit (1 epoch, 10 iters) | .724±.070 | .856±.026 | .722±.070 | .47 / .62 |

**Winner: LogReg.** It beats zero-shot by +0.17 macro-F1 (paired fold diff std 0.03, wins 5/5), clearly larger than fold variance. Over SetFit it leads by only +0.023 (paired std 0.033, wins 3/5): within noise, so LogReg and SetFit are effectively tied here, and LogReg is ~100x cheaper (seconds vs ~3-5 min/fold). SetFit was run with small settings only; more epochs/iterations were not tried.

**Weak classes (recall, zero-shot -> LogReg / SetFit):** Veera .26 -> .76/.68, Shringara .48 -> .78/.72, Shanta .44 -> .66/.62. Both trained models largely fix Veera and Shringara; Shanta improves but stays the weakest (~.6).

**Still confused:** Shringara <-> Karuna (both directions, in every model), Karuna -> Raudra, Raudra -> Bibhatsa/Veera, Veera -> Raudra; SetFit also Shanta -> Bhayanaka.

**None handling (25 rows):** neither trained model rejects None reliably: LogReg predicts a rasa for 28±23% of None rows, SetFit 48±23% (zero-shot has no None class: 100%). Max-prob gap (emotional minus None): zero-shot .21, LogReg .36, SetFit .28, so a max-prob threshold is a weak but usable signal. Only 5 None rows per fold: high variance.

Saved artifact: models/best (LogReg); `from rasa_classifier import predict` -> (n,10), order Shringara..Shanta, None.
