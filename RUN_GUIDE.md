# Run guide

Two tracks that **run in parallel** — nothing in Track B blocks Track A:

- **Track A — Kaggle T4 (GPU):** train the DistilBERT variants. Needs only the *Kaggle* features
  (`data/{train,val,test}_features.csv`). Independent of the new matched-pair emails.
- **Track B — local CPU:** generate the matched-pair emails, then extract their features.

They only converge at the **scoring** step (score the matched-pair features with the trained
checkpoints), which needs both tracks finished.

```
Track A (T4):   train_features ─► train.py (35 runs) ─────────────┐
                                                                  ├─► evaluate_predictions.py + tfidf_baseline.py
Track B (CPU):  generate_mpg ─► extract_features (MPG) ───────────┘
```

---

## Prerequisite that does NOT exist yet: `data/train_features.csv`

The repo ships `data/val_features.csv` and `data/test_features.csv`, but **not**
`data/train_features.csv` — `train.py` needs it and it has to be generated once from
`data/train.csv` (17.5k rows). This is the slow step (LanguageTool + GPT-2 perplexity, ~60–90 min
on CPU). Do it once, anywhere, then reuse:

```bash
python extract_features.py --input data/train.csv     # -> data/train_features.csv  (~60-90 min)
```

Run it on the Kaggle notebook at the start of Track A, or locally and upload the result. (A fast
check is `--max_rows 2000`, but the full corpus is what the reviewer's E2 needs.)

---

## Track B — local (CPU): generation and feature extraction

```bash
# one-time setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt openai
python -m spacy download en_core_web_sm
# .env needs GROQ_API_KEY=...  (add more providers below to get more generators)

python check_dataset.py          # verify the Day-1 data findings

# smoke test (tiny)
python generate_mpg.py --stage generate --generators groq:openai/gpt-oss-20b --per_cell 1

# full grid: all 4 generators run CONCURRENTLY, each paced by its own rate limiter
python generate_mpg.py --stage generate --per_cell 15
python generate_mpg.py --stage status      # accepted counts + refusal rate per generator
python generate_mpg.py --stage rewrite      # register variants (P_R / B_hard_R)
python generate_mpg.py --stage audit_sheet --n 60   # blinded sheet for the kappa rating step

# features on the generated emails (slow part = LanguageTool, ~15-30 min for the grid)
python extract_features.py --input mpg_emails.csv     # -> mpg_emails_features.csv
```

### Rate limiting (`rate_limiter.py`)
Each model gets its own limiter, so the four generators run at the same time without starving
each other. The limiter combines a per-model minimum gap between calls (`RATE_LIMITS`) with
header-driven throttling: it reads Groq's `x-ratelimit-remaining/reset-*` headers and, when the
per-minute **token** budget runs low, waits only long enough to refill what the next call needs
(not the full bucket). Override the floor for every model with `--min_interval 5` if you hit
limits; raise a single model's `min_interval` in `RATE_LIMITS` to slow just that one.

Observed free-tier limits (check yours on Day 1 — the limiter adapts either way):

| model | requests/day | tokens/min | notes |
|---|---|---|---|
| allam-2-7b | ~7000 | ~6000 | token/min is the bottleneck (~11 emails/min); 0% refusals |
| qwen/qwen3.8-27b | ~1000 | ~8000 | slowest per call (reasoning); 0% refusals |
| openai/gpt-oss-20b | ~1000 | ~8000 | ~21% refusal on phishing prompts |
| openai/gpt-oss-120b | ~1000 | ~8000 | **refuses ~62% of phishing** — see note below |

### gpt-oss-120b refuses most phishing prompts
In testing, `gpt-oss-120b` returns "I can't help with that" for ~62% of the phishing (P) prompts,
so it yields few P emails (it's fine for the benign B_hard/B_easy conditions). Its refusal rate is
itself a reportable E1 result. If you want a 4th *reliable* phishing generator, add another
provider (below) and drop gpt-oss-120b from the P condition, or keep it only for benign + refusal
stats. allam, gpt-oss-20b and qwen all comply.

### Adding more generators / providers
`generate_mpg.py` and `rate_limiter.py` take any `provider:model`. Providers live in
`PROVIDERS` in `generate_mpg.py` (groq, gemini, openrouter, cerebras, mistral) — add the matching
key to `.env` and pass e.g. `--generators groq:allam-2-7b,openrouter:meta-llama/llama-3.1-70b-instruct`.

### Rough timing (full grid, `--per_cell 15`, ~1800 target emails)
Concurrent wall time is bounded by the slowest model, not the sum. Expect **~45 min** for
generation (allam is token/min-bound at ~11/min; qwen is per-call-slow), plus **~15–30 min** for
feature extraction. Generation resumes from the CSVs if a quota is hit, so a daily-limited model
can be finished the next day without redoing anything.

---

## Track A — Kaggle T4: training and GPU scoring

Upload `model.py`, `train.py`, `evaluate_predictions.py`, `tfidf_baseline.py`, and the feature
CSVs: `data/train_features.csv` (generate it first, see above), `data/val_features.csv`,
`data/test_features.csv`, and later `mpg_emails_features.csv`. Settings → Accelerator → GPU T4.

```bash
pip install -q transformers

# time one run first (fp16/AMP is on automatically on a T4, ~2x faster than before)
python train.py --train /kaggle/working/train_features.csv --val /kaggle/input/initial/val_features.csv \
    --variants baseline,behavioral --seeds 0
```

**Use BOTH T4s (Kaggle "GPU T4 x2").** The 35 runs (7 variants x 5 seeds) are independent, so
instead of DataParallel (which gives ~no gain on a 66M-param model and muddies the seed-variance
analysis) each GPU runs its own process over half the grid. In a single `!`-shell cell:

```bash
TRAIN=/kaggle/working/train_features.csv
VAL=/kaggle/input/initial/val_features.csv
VARS=baseline,perplexity,grammar,length,behavioral,psychological,full

CUDA_VISIBLE_DEVICES=0 python train.py --shard 0 --num_shards 2 --train $TRAIN --val $VAL --variants $VARS --seeds 0,1,2,3,4 &
CUDA_VISIBLE_DEVICES=1 python train.py --shard 1 --num_shards 2 --train $TRAIN --val $VAL --variants $VARS --seeds 0,1,2,3,4 &
wait

# merge the two per-shard summaries into one
python -c "import pandas as pd, glob; pd.concat([pd.read_csv(f) for f in glob.glob('train_summary_shard*.csv')]).to_csv('train_summary.csv', index=False)"
```

Each shard writes its own `train_summary_shard{0,1}.csv` (no clobbering); checkpoints go to the
shared `saved_models/`. AMP + both GPUs brings the full matrix from ~2 h down to roughly ~30 min.

Single-GPU fallback (one card, or a different accelerator):

```bash
python train.py --variants baseline,perplexity,grammar,length,behavioral,psychological,full --seeds 0,1,2,3,4

# per-email predictions for the paired tests (Kaggle test + matched-pair set)
python evaluate_predictions.py --inputs data/test_features.csv mpg_emails_features.csv --out predictions_long.csv

# matched TF-IDF baseline (same train split) + learning curve  (CPU, fast)
python tfidf_baseline.py --inputs data/test_features.csv mpg_emails_features.csv --learning_curve
```

Download `predictions_long.csv`, `tfidf_predictions_long.csv`, `train_summary.csv`,
`mpg_emails_features.csv` and `tfidf_learning_curve.csv` back to the repo root.

Checkpoints are named `saved_models/model_<variant>_seed<seed>.pt` and auto-discovered by
`evaluate_predictions.py`.

---

## Analysis — tables and figures (local, runs on the downloaded dumps)

Once the five files above are in the repo root, everything below runs in ~1 min on CPU. No model
is re-run; the scripts read only the per-email dumps plus the matched-pair feature file.

```bash
# all hypothesis tests + result tables -> analysis_out/*.csv
python analysis/paper_stats.py \
    --preds predictions_long.csv --tfidf tfidf_predictions_long.csv \
    --features mpg_emails_features.csv --out analysis_out

# all publication figures -> analysis_out/*.png
python analysis/paper_figures.py \
    --preds predictions_long.csv --tfidf tfidf_predictions_long.csv \
    --features mpg_emails_features.csv --train data/train.csv \
    --learning_curve tfidf_learning_curve.csv --out analysis_out
```

What comes out, mapped to the reviewer's required items and the pre-registered hypotheses:

| Output | Reviewer item | Hypothesis |
|---|---|---|
| `table_mpg_summary.csv`, `fig_H1_intent_auc.png` | matched counterfactual | H1 intent-blindness |
| `table_register_effect.csv`, `fig_H2_register.png` | authorship vs dataset shift | H2 register-sensitivity |
| `table_feature_authorship.csv`, `fig_H3_features.png`, `fig_H3_by_generator.png` | feature distributions | H3 authorship-not-intent |
| `fig_H4_false_alarm.png`, `fig_tfidf_coef.png`, `table_tfidf_top_tokens.csv` | TF-IDF coefficient plot | H4 lexical false alarm |
| `table_ablation_vs_baseline.csv`, `fig_ablation.png`, `fig_forest.png` | single-feature ablation + paired tests | — |
| `table_recall_by_generator.csv`, `fig_heatmap.png` | per-generator TF-IDF/DistilBERT | — |
| `table_conventional_by_seed.csv`, `table_seed_variance.csv` | ≥5 seeds, no-cost check | — |
| `fig_collapse.png`, `fig_score_hist.png`, `fig_learning_curve.png` | learning curve, error diagnostics | — |

Paired tests use **exact McNemar** (`scipy.stats.binomtest` on discordant pairs) and a **paired
bootstrap** (2000 resamples) for recall differences with 95% CIs; the P↔P_R register pairs are
reconstructed from `source_sample_id` in the feature file.
