# How ragas from the paper are matched to Saraga

The rasa→raga lists come from the paper and are not altered. Saraga spells ragas differently, so each
raga is searched under several names. Policy: **the paper's own name plus spelling variants of the same
name only.** Counterparts under a different name or tradition are not used unless `--equivalents` is
passed, and they are unverified (the web check below used Wikipedia-derived pages and blogs, not a
musicology reference).

Names from the paper that contain "/" or "–" are split into the paper's own parts
(Abheri/Bhimpalasi, Kalavathi/Valachi, Shuddha Saveri – Durga).

## Spelling variants (always searched)
| Paper name | Also searched |
|---|---|
| Abheri/Bhimpalasi | Abheri, Bhimpalasi, Bhimpalas |
| Desh | Des |
| Hindola | Hindolam |
| Malayamarutham | Malayamarutam |
| Aathana | Athana |
| Kunthalavarali | Kuntalavarali |
| Reethigowla | Riti Gaula, Ritigaula |
| Shankarabharanam | Sankarabharanam |
| Ahibhairav | Ahir Bhairav |
| Bageshri | Bageshree |
| Lalith | Lalitha, Lalita |
| Punnagavarali | Punnaga Varali |
| Shubhapanthuvarali | Shubhapantuvarali, Subhapantuvarali |
| Kalavathi/Valachi | Kalavathi, Valachi, Kalavati |
| Mayamalavagowla | Mayamalavagaula |
| Shuddha Saveri – Durga | Shuddha Saveri, Durga |
| Sindhu Bhairavi | Sindhubhairavi |
| Yadhukula Kambhodhi | Yadukula Kambhoji, Yadukula Kambodhi |
| Behaag | Behag |
| Pahaadi | Pahadi |
| YamanKalyani | Yaman Kalyan, Yaman Kalyani |
| Kedaragowla | Kedaragaula |
| Karaharapriya | Kharaharapriya |
| Madhyamavathi | Madyamavati, Madhyamavati |
| Meghamalhaar | Megh Malhar, Meghamalhar |
| Mohana | Mohanam |

Even these are transliteration judgements I made, not checked against a reference. "Hindolam" and
"Lalitha/Lalita" assume the paper means the Carnatic raga of that name.

## Counterparts (only with `--equivalents`, unverified)
| Paper name | Extra names | Evidence found |
|---|---|---|
| Ahibhairav | Chakravakam | Commonly cited Carnatic counterpart of Ahir Bhairav (melakarta 16), but a counterpart, not identical |
| Hindola | Hindol | Carnatic Hindolam is usually paired with Hindustani Malkauns, not Hindol, so this is doubtful |
| Lalith | Lalit, Lalat | Hindustani Lalit is often paired with Carnatic Lalitha; not checked |
| Behaag | Bihag | Listed as a north-to-south borrowing; in Hindustani, Behag and Bihag are separate ragas |
| YamanKalyani | Yaman | Yaman is Hindustani for Kalyan/Kalyani-type scale; usually treated as the Carnatic Kalyani counterpart |
| Meghamalhaar | Megh | Megh and Megh Malhar are related but not the same raga |

## Removed (were in an earlier version, now not searched at all)
| Name | Why |
|---|---|
| Ahiri (for Ahibhairav) | A different Carnatic raga |
| Margahindolam, Hindol Pancham (for Hindola) | Distinct ragas |
| Brindavana Saranga (for Brindavani) | Distinct from Brindavani |
| Durga alone, Yaman alone, Megh alone | Only the paper's own pairs and the variants above are kept |

## Effect on coverage
Strict matching gives zero Saraga songs for: Arabhi, Malayamarutham, Aathana, Madhuvanti, Shivaranjani,
Shubhapanthuvarali, Mayamalavagowla, Yadhukula Kambhodhi, Brindavani, Meghamalhaar. See `coverage_report.md`.

## To settle against the paper
Its exact raga spellings, which tradition each raga belongs to, and whether it states any
Carnatic/Hindustani equivalents. Items most affected: Lalith, YamanKalyani, Meghamalhaar, Hindola,
Brindavani, Ahibhairav.
