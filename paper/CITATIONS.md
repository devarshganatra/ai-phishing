# Citation map for the revised paper

`references.bib` holds every entry. This file says **where each one goes and what it supports**,
so the writeup can cite against the actual results in `analysis_out/`.

Status: **[V]** checked against arXiv / publisher record (Nov 2026) · **[S]** standard reference ·
**[C]** confirm volume/pages/authors before submission.

---

## 1. Section-by-section placement

### 1 Introduction — threat and motivation
| Key | Use it for |
|---|---|
| `hazell2023spear`, `heiding2024devising`, `heiding2024automated` | LLMs make phishing cheap and as effective as human experts (54% vs 12% click-through) |
| `bethany2024lateral` | LLM lateral phishing matches professional comms staff in a real 9,000-employee org |
| `qi2024spearbot` | BERT/RoBERTa/GPT-2 defenders trained on legacy corpora collapse on LLM spear-phishing |
| `pajola2025ephishgen` | Same-dataset near-perfect scores mislead; detectors lose >20 pts on E-PhishLLM |
| `arp2022dos`, `koh2021wilds` | In-distribution accuracy is not evidence of robustness under shift |

### 2.1 Feature-based and psychological detection
| Key | Use it for |
|---|---|
| `fette2007learning`, `verma2012detecting` | Classical feature / NLP detectors (kept from v2) |
| `ferreira2015analysis`, `cialdini2006influence` | Persuasion principles (authority, urgency) behind the psychological features |
| `shahriar2022psychological` | **Closest prior work** — urgency/fear trait scores fused with BERT/SBERT. *Must cite and differentiate* (see §3) |
| `toth2025constructing` | Human vs LLM email dataset annotated for urgency/fear/authority. ⚠ arXiv shows a withdrawn version — check status first |

### 2.2 Transformer and LLM-based detectors
| Key | Use it for |
|---|---|
| `vaswani2017attention`, `devlin2019bert`, `sanh2019distilbert` | Backbone (now canonical venues instead of arXiv) |
| `lee2020catbert`, `jamal2023improved` | Prior Transformer email classifiers (kept from v2) |
| `mahendru2024securenet` | DeBERTa vs GPT-4/Gemini for phishing |
| `koide2024chatspamdetector`, `liu2025pimref` | LLM-as-detector and reference/fact-checking detectors — the direction the field is moving |
| `kulal2025robust` | Conventional + LLM-generated + adversarial phishing evaluation (88 LLM emails, no benign LLM class) |
| `liu2019roberta`, `he2021deberta` | Limitations: stronger encoders not evaluated |

### 2.3 Dataset artifacts and corpus provenance (supports H4 + the TF-IDF result)
| Key | Use it for |
|---|---|
| `klimt2004enron`, `alsubaiey2024novel`, `champa2024curated`, `mendes2025meajor` | Corpus provenance: the Kaggle collection is a merge of Enron + legacy spam/phishing sources |
| `verma2019dataquality` | Data-quality pitfalls in phishing datasets |
| `bhuiyan2026fragility` | **Closest overlap with our provenance finding** — TF-IDF models learn corpus identity (97–98%). *Must cite and differentiate* (see §3) |
| `pendlebury2019tesseract`, `sommer2010outside` | Spatial/temporal bias; the "closed world" problem in security ML |

### 2.4 Shortcut learning and contrast sets (methodological backbone)
| Key | Use it for |
|---|---|
| `geirhos2020shortcut` | Definition of shortcut learning — frames "register as a shortcut for phishing" |
| `gururangan2018annotation`, `mccoy2019right`, `niven2019probing` | High benchmark scores driven by spurious cues |
| `gardner2020contrast` | **Direct precedent for P vs B_hard** — minimally different pairs with opposite labels probe the decision boundary |
| `kaushik2020learning` | **Direct precedent for P vs P_R** — counterfactual edits that change one factor and hold the rest |
| `ribeiro2020checklist` | Behavioral testing beyond aggregate accuracy |

### 2.5 Machine-generated text signals and register
| Key | Use it for |
|---|---|
| `gehrmann2019gltr`, `ippolito2020automatic`, `mitchell2023detectgpt`, `radford2019language` | Likelihood statistics as LLM-text signals (motivation for perplexity) |
| `sadasivan2023can`, `krishna2023paraphrasing` | Paraphrase breaks those signals (DetectGPT 70.3% → 4.6%) — motivates the register rewrite P → P_R |
| `liang2023gpt` | **Supports H3**: perplexity tracks writing style/fluency, not authorship or intent |
| `biber1988variation`, `biber2009register` | Formal definition of "register" — cite at first use of the term |

### 3 Methodology
| Key | Use it for |
|---|---|
| `bari2025allam`, `yang2025qwen3`, `openai2025gptoss`, `groq_allam`, `groq_qwen` | The three generators (+ gpt-oss-20b as the fixed rewriter) |
| `languagetool`, `honnibal2020spacy`, `radford2019language` | Feature extraction tools |
| `wolf2020transformers`, `paszke2019pytorch`, `pedregosa2011scikit`, `loshchilov2019decoupled` | Implementation and optimizer |

### 3.x Statistical analysis plan
| Key | Use it for |
|---|---|
| `mcnemar1947note`, `dietterich1998approximate` | Exact paired McNemar on per-email decisions |
| `efron1993bootstrap` | Paired bootstrap CIs for Δrecall |
| `holm1979simple` | Correction across the 7 planned ablation contrasts |
| `dror2018hitchhiker` | Significance testing practice in NLP |
| `bouthillier2021accounting` | Why 5 seeds — and to frame the large OOD seed variance |
| `cohen1988statistical` | Cohen's d in the H3 feature table |
| `hanley1982meaning` | AUC interpretation for the H1 intent test |
| `lakens2017equivalence` | TOST if you claim "no cost on conventional data" |
| `wilson1927probable` | Only if Wilson intervals are kept anywhere |

---

## 2. Figure → citation map

| Figure / table | Finding (real data) | Cite alongside |
|---|---|---|
| `fig_collapse` | In-distribution F1 0.987 → matched-pair recall 0.53 | `qi2024spearbot`, `pajola2025ephishgen`, `arp2022dos`, `koh2021wilds` |
| `fig_H1_intent_auc`, `table_mpg_summary` | AUC(P vs B_hard) ≈ 0.62 vs AUC(P vs B_easy) ≈ 0.79 | `gardner2020contrast`, `geirhos2020shortcut`, `mccoy2019right`, `hanley1982meaning` |
| `fig_H2_register`, `table_register_effect` | Recall drops P → P_R; paired test n.s. at n = 48 | `kaushik2020learning`, `krishna2023paraphrasing`, `sadasivan2023can`, `biber2009register` |
| `fig_H3_features`, `fig_H3_by_generator`, `table_feature_authorship` | PPL/grammar/length track generator (spread 1.3–1.6 SD); urgency/fear track intent (d = 1.08 / 0.84) | `liang2023gpt`, `shahriar2022psychological`, `ferreira2015analysis`, `cohen1988statistical` |
| `fig_H4_false_alarm` | FPR 0.37 on benign twins vs 0.12 on plain benign | `gururangan2018annotation`, `ribeiro2020checklist` |
| `fig_tfidf_coef`, `table_tfidf_top_tokens` | Safe = enron/vince/2002; phish = viagra/click here/2004 | `bhuiyan2026fragility`, `verma2019dataquality`, `klimt2004enron` |
| `fig_learning_curve` | TF-IDF recall 0.98 already at n = 1,600 | `pajola2025ephishgen` (lexical baselines can transfer well) |
| `fig_ablation`, `fig_forest`, `table_ablation_vs_baseline` | Behavioral +0.013 (p = 0.24, n.s.); TF-IDF +0.23 (p < 0.001) | `mcnemar1947note`, `dietterich1998approximate`, `efron1993bootstrap`, `holm1979simple` |
| `table_seed_variance` | OOD recall SD 0.06–0.14 vs in-distribution SD 0.001 | `bouthillier2021accounting` |
| `fig_heatmap`, `table_recall_by_generator` | allam 0.74 vs gpt-oss 0.41 / qwen 0.43 (baseline) | `qi2024spearbot` (attack strategy changes defender behavior) |

---

## 3. Differentiation sentences the reviewer asked for

Draft wording; adapt it in the writeup.

- **vs Shahriar et al. (2022):** They show learned urgency/fear trait scores *improve* BERT on a
  conventional corpus. We test whether simple cue counts change behavior on *matched* LLM pairs, and
  find these cues carry intent signal (d ≈ 1) that the fused DistilBERT still does not use to separate
  P from B_hard (AUC unchanged at ≈ 0.62).
- **vs SpearBot (Qi et al., 2024):** SpearBot shows defenders collapse on adversarially refined LLM
  phishing. We show the collapse persists without adversarial refinement, and we isolate *why*: on
  intent-matched benign twins the detector is near chance.
- **vs E-PhishGen (Pajola et al., 2025):** E-PhishGen documents cross-dataset degradation on a large
  synthetic benchmark. We contribute a controlled factorial (intent × register × generator) that
  attributes the degradation to cue/register reliance, not dataset shift alone.
- **vs Bhuiyan & Bhuiyan (2026):** They show TF-IDF models learn corpus identity across six classical
  corpora. We confirm the same provenance tokens on our training corpus, and add the opposite twist
  for Transformers: the lexical model reads intent *better* (AUC 0.75 vs 0.62) on matched LLM pairs.
- **vs Gardner et al. (2020) / Kaushik et al. (2020):** We bring contrast-set and counterfactual
  evaluation from NLI/sentiment to phishing detection, where the "label-flipping minimal edit" is the
  benign twin.

---

## 4. Changes from the v2 reference list

**Kept (17):** Ferreira & Lenzini, Fette, Verma 2012, CATBERT, Jamal & Wimmer, Hazell, Heiding 2024,
GLTR, Ippolito, DetectGPT, GPT-2, DistilBERT, Sadasivan, Enron, AdamW, Wilson, HF Transformers.
Vaswani, BERT and AdamW now cite the canonical venue instead of arXiv (the reviewer asked for this).

**Dropped:**
- `[16] Garcez & Lamb (neurosymbolic)` — only existed to justify "dual-stream", which the reviewer
  asked you to drop.
- `[21] Groq Compound` — the generator is no longer in the study.
- `[18] imbalanced-learn` — `data/train.csv` is already balanced (8,783 / 8,783). Re-add it only if
  the methodology still mentions oversampling.

**Not citable:**
- **Park et al. 2026 ("A Large-Scale Empirical Study of Modern Phishing Email Content")**, recommended
  by the reviewer, **could not be found** on arXiv or anywhere else. Do not cite it unless you have
  the PDF or a working link.

**Before submission, confirm:** E-PhishGen full author list (v1/v2 order differs) · Bethany et al.
author list and IEEE Access volume · SpearBot 5th author · Al-Subaiey volume · BDCC volume/issue ·
Tóth et al. withdrawal status · Heiding et al. later *Expert Systems with Applications* version.
