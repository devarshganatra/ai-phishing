# Day-1 data findings (verified in-repo, 2026-11-03)

Run `python check_dataset.py` to reproduce the first two.

### 1. The groq "collapse" is a refusal artifact
`ai_heldout_test.csv` has 513 rows. The curly apostrophe in "I'm sorry, but I can't help with
that." slipped past the old straight-apostrophe filter in `check_dataset.py`.

| generator | n | unique texts | refusals | with placeholders | valid |
|---|---|---|---|---|---|
| allam-2-7b | 367 | 367 | 0 | 358 | 367 |
| groq/compound | 144 | 29 | **116** | 25 | **28** |
| qwen/qwen3.8-27b | 2 | 2 | 0 | 2 | 2 |

So "groq recall = 6.25%" is mostly the detector correctly **not** flagging a refusal. The usable
held-out set is **397 emails**, effectively one generator (allam). The fixed filter writes
`ai_heldout_test_valid.csv` (397 rows) and `ai_heldout_test_valid_features.csv`. **Re-run every
per-generator number on the valid set and exclude qwen (n=2).**

### 2. allam emits template placeholders everywhere
358/367 allam emails contain `[slot]` placeholders — in phishing and legitimate twins alike.
They are a generator signature, not a phishing cue, so the MPG harness **flags** them
(`n_placeholders` column) rather than dropping them; the matched-pair design controls for generator.

### 3. The Kaggle corpus largely encodes provenance, not intent
Re-fit TF-IDF (1–2 gram, logistic regression) on `data/train.csv`:
- Kaggle test AUC ≈ 0.999, recall ≈ 0.985.
- Top **safe** tokens: `enron, vince, louise, 2001, 2002, edu`.
- Top **phishing** tokens: `viagra, 2004, 2005, remove, sightings, click here`.
- On the 513-row held-out set the refusals score low (mean prob ≈ 0.12–0.49 by n-gram), so a
  reported 95.71% recall that includes refusals as "detected" would be partly false alarms.

This is why the study reframes around **intent vs register** and reports AUC + threshold
recalibration, not raw same-corpus accuracy.

### Smaller notes
- The grammar-error feature is internally consistent (numerator and denominator both use the first
  1,500 chars); only the paper's equation text needs fixing.
- `train.py` now seeds all RNGs and saves per-seed checkpoints; evaluation now dumps per-email
  predictions for the paired tests.
