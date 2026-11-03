# Pre-registration — Intent or Register?

**Study:** *Intent or Register? A Matched-Pair Audit of Phishing Detectors and AI-Text Features under LLM Authorship*
**Frozen:** 2026-11-03 · **Prompt version:** `mpg-v1`

These are predictions written **before** any model is trained or evaluated on the matched-pair
grid (MPG). They are not results. Every outcome — including the failure of any hypothesis — is
reportable; a failed H1–H4 means the detectors *do* track intent and the earlier "collapse" was
calibration plus refusal artifacts, which corrects the record.

## Background facts already verified in the repo (see DATA_FINDINGS.md)
- 116 / 144 `groq/compound` "phishing" emails are the identical refusal "I'm sorry, but I can't
  help with that." (curly apostrophe slipped past the old filter). The usable held-out set is 397
  emails, effectively one generator (allam, n=367).
- `allam-2-7b` emits `[slot]` placeholders in ~95% of emails, in phishing and legitimate alike.
- The Kaggle corpus largely encodes provenance: TF-IDF's top tokens are `enron/vince/2001` (safe)
  and `viagra/2004/2005` (phishing), not intent cues.

## Design
Matched-pair grid: for each scenario × generator, three conditions holding surface cues constant
while varying only intent — **P** (phishing, label 1), **B_hard** (legitimate twin with the same
sender/deadline/link/authority cues, label 0), **B_easy** (plain legitimate, label 0). A rewrite
stage produces intent- and fact-preserving register variants (**P_R**, **B_hard_R**).

Generators (≥4, target 4–6, Groq free tier): `allam-2-7b`, `qwen/qwen3.8-27b`,
`openai/gpt-oss-20b`, `openai/gpt-oss-120b`. Every attempt (success/refusal/reject/error) is logged.

## Hypotheses
- **H1 — intent-blindness.** Kaggle-trained detectors score AUC ≤ 0.75 separating P from B_hard,
  despite ≥ 0.98 on the Kaggle test split.
- **H2 — register-sensitivity.** An intent-preserving register rewrite moves DistilBERT's score
  more than flipping intent (P ↔ B_hard) does.
- **H3 — features encode authorship, not intent.** Perplexity, grammar-error rate and length
  separate human (Kaggle) from LLM text (AUC > 0.9) but not P from B_hard within LLM text
  (AUC < 0.65); adding them raises recall and B_hard false positives by similar amounts.
- **H4 — lexical trigger.** TF-IDF has high P recall but ≥ 40% false-positive rate on B_hard, so
  its apparent robustness is partly false alarms.
- **H5 — fix (optional).** Pair-contrastive fine-tuning on pairs from 2 generators raises
  P-vs-B_hard AUC to ≥ 0.9 on unseen generators and held-out scenarios, costs ≤ 1 F1 point on
  Kaggle, and beats both threshold recalibration and naive data mixing.

## Analysis plan (all from per-email prediction dumps)
- Primary metric on MPG: AUC and FPR@threshold (threshold fixed on the Kaggle validation split).
  **Drop precision/F1 on any positive-only set.** Qwen excluded from inferential analysis if n is small.
- Model comparisons: exact McNemar + paired bootstrap; Holm correction across planned contrasts;
  cluster bootstrap over generator × scenario for MPG.
- Variance: 5 seeds per DistilBERT variant; report mean, SD, 95% CI. Separate training-run variance
  from test-item variance.
- Feature ablation: single-feature variants (`perplexity`, `grammar`, `length`) in addition to the
  behavioral block, so any gain is attributed to a specific feature.

## Cut lines
- Drop H5/E6 if not clearly working by Day 8; the paper stands on the audit (E1–E5).
- 4 generators is the minimum; drop to 4 if quotas block 6.
- If the Day-3 twin-prompt human audit scores κ < 0.7, fix prompts before generating more.
