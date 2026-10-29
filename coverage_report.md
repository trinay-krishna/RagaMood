# Raga coverage in the Saraga dataset

35 mapped ragas: 10 missing, 17 underrepresented (1-2 songs), 8 with 3+ songs. Counts are unique songs across Carnatic + Hindustani, using the paper's names and same-name spelling variants only (add `--equivalents` to include unverified counterparts).

## Songs per rasa

| Rasa | Ragas | Songs | Ragas with 0 songs |
|---|---|---|---|
| Adbhuta | 5 | 5 | 2 |
| Hasya | 4 | 9 | 1 |
| Karuna | 8 | 10 | 3 |
| Shanta | 6 | 7 | 2 |
| Shringara | 8 | 23 | 1 |
| Veera | 4 | 8 | 1 |

## Missing (0 songs)

| Rasa | Raga | Songs | Spellings tried |
|---|---|---|---|
| Adbhuta | Arabhi | 0 | Arabhi |
| Adbhuta | Malayamarutham | 0 | Malayamarutham, Malayamarutam |
| Hasya | Aathana | 0 | Aathana, Athana |
| Karuna | Madhuvanti | 0 | Madhuvanti |
| Karuna | Shivaranjani | 0 | Shivaranjani |
| Karuna | Shubhapanthuvarali | 0 | Shubhapanthuvarali, Shubhapantuvarali, Subhapantuvarali |
| Shanta | Mayamalavagowla | 0 | Mayamalavagowla, Mayamalavagaula |
| Shanta | Yadhukula Kambhodhi | 0 | Yadhukula Kambhodhi, Yadukula Kambhoji, Yadukula Kambodhi |
| Shringara | Brindavani | 0 | Brindavani |
| Veera | Meghamalhaar | 0 | Meghamalhaar, Megh Malhar, Meghamalhar |

## Underrepresented (1-2 songs)

| Rasa | Raga | Songs |
|---|---|---|
| Adbhuta | Hindola | 1 |
| Karuna | Punnagavarali | 1 |
| Shanta | Sama | 1 |
| Shanta | Shuddha Saveri – Durga | 1 |
| Shringara | Pahaadi | 1 |
| Adbhuta | Abheri/Bhimpalasi | 2 |
| Adbhuta | Desh | 2 |
| Hasya | Kunthalavarali | 2 |
| Karuna | Ahibhairav | 2 |
| Karuna | Bageshri | 2 |
| Karuna | Lalith | 2 |
| Shanta | Kalavathi/Valachi | 2 |
| Shringara | Kapi | 2 |
| Shringara | Karaharapriya | 2 |
| Shringara | YamanKalyani | 2 |
| Veera | Kedaragowla | 2 |
| Veera | Madhyamavathi | 2 |

## Adequate (3+ songs)

| Rasa | Raga | Songs |
|---|---|---|
| Hasya | Shankarabharanam | 3 |
| Karuna | Kanada | 3 |
| Shanta | Sindhu Bhairavi | 3 |
| Hasya | Reethigowla | 4 |
| Shringara | Kalyani | 4 |
| Veera | Mohana | 4 |
| Shringara | Behaag | 5 |
| Shringara | Kamas | 7 |

## Notes

- A zero can mean a spelling mismatch rather than true absence; supply alternate spellings and add them to `RAGA_VARIANTS` in `rasa_raga_map.py`.
- The 2-song cutoff is a judgement call, not a statistical threshold.
- Songs are Saraga metadata entries, so one raga can span several concerts or one concert can contribute several songs.
