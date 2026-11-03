# Run guide

Two kinds of work: **light** (generation, audits, TF-IDF) runs locally on CPU; **heavy**
(DistilBERT training, GPU scoring) runs on a Kaggle T4 notebook.

## Local (CPU) — generation and audits

```bash
# one-time
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt openai
python -m spacy download en_core_web_sm

# .env needs GROQ_API_KEY=... (add GEMINI_API_KEY / OPENROUTER_API_KEY to add providers)

# 0. verify the Day-1 data findings
python check_dataset.py

# 1. smoke-test the matched-pair harness (tiny, ~1 min)
python generate_mpg.py --stage generate --generators groq:openai/gpt-oss-20b --per_cell 1

# 2. full generation grid (resumes if interrupted / quota hit; mind free-tier limits)
python generate_mpg.py --stage generate --per_cell 15
python generate_mpg.py --stage status            # accepted counts + refusal rate per generator
python generate_mpg.py --stage rewrite           # register variants P_R / B_hard_R
python generate_mpg.py --stage audit_sheet --n 60   # blinded sheet for the κ rating step

# 3. feature extraction on the generated emails (slow part = LanguageTool, ~1-2 h for all)
python extract_features.py --input mpg_emails.csv     # -> mpg_emails_features.csv
```

Groq free-tier limits seen on this key (check yours on Day 1 with `python list_models.py` and the
response headers): allam-2-7b ~7000 req/day; qwen / gpt-oss ~1000 req/day, ~8000 tok/min. Spread
heavy generation across providers, or run over two days — the harness resumes from the CSVs.

## Kaggle T4 — training and GPU scoring

Upload `model.py`, `train.py`, `evaluate_predictions.py`, `tfidf_baseline.py` and the feature CSVs
(`data/train_features.csv`, `data/val_features.csv`, `data/test_features.csv`,
`mpg_emails_features.csv`). Enable the T4 GPU (Settings → Accelerator → GPU T4).

```bash
pip install -q transformers

# 7 variants x 5 seeds = 35 runs, ~3-4 min each on a T4 (~2 h total). Start small to time one run:
python train.py --variants baseline,behavioral --seeds 0

# full matrix
python train.py --variants baseline,perplexity,grammar,length,behavioral,psychological,full --seeds 0,1,2,3,4

# per-email predictions for the paired tests (Kaggle test + MPG)
python evaluate_predictions.py --inputs data/test_features.csv mpg_emails_features.csv --out predictions_long.csv

# matched TF-IDF baseline (same train split) + learning curve
python tfidf_baseline.py --inputs data/test_features.csv mpg_emails_features.csv --learning_curve
```

Download `predictions_long.csv`, `tfidf_predictions_long.csv`, `train_summary.csv` back to the repo;
the statistics (McNemar, paired bootstrap, cluster bootstrap) run locally on those dumps.

## Checkpoint naming
`saved_models/model_<variant>_seed<seed>.pt`. `evaluate_predictions.py` auto-discovers them.
