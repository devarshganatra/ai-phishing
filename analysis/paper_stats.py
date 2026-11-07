"""Hypothesis tests and result tables for the Intent-vs-Register study.

Reads the per-email prediction dumps (evaluate_predictions.py + tfidf_baseline.py) and the
matched-pair feature file, and writes the tables the paper needs. Nothing here re-runs a model:
it all operates on the dumps, so it is fast and fully reproducible.

Hypotheses (see PREREG.md):
  H1 intent-blindness   : a detector barely separates P from its matched benign twin B_hard
                          (same surface cues, opposite intent) even though it separates P from
                          B_easy. Metric: AUC(P vs B_hard) and the gap to AUC(P vs B_easy).
  H2 register-sensitivity: rewriting a phishing email into a neutral register (P -> P_R, intent
                          held constant) changes detection. Metric: paired recall drop + McNemar.
  H3 authorship-not-intent: handcrafted features track the GENERATOR more than the intent
                          (P vs B_hard). Metric: Cohen's d(intent) vs spread across generators.
  H4 lexical false alarm : TF-IDF fires on B_hard (shared lexical cues) -> false positives.
                          Metric: FPR on B_hard vs B_easy.

Usage (after downloading the Kaggle dumps into this folder):
  python analysis/paper_stats.py \
      --preds predictions_long.csv \
      --tfidf tfidf_predictions_long.csv \
      --features mpg_emails_features.csv \
      --out analysis_out
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

MPG = 'mpg_emails'          # dataset name produced from mpg_emails_features.csv
THRESH = 0.5
FEATURES = ['perplexity', 'grammar_errors', 'word_count', 'urgency_score', 'fear_score', 'authority_score']
rng = np.random.default_rng(0)


# ----------------------------------------------------------------------------- loading
def load_preds(preds_path, tfidf_path):
    """Concatenate DistilBERT + TF-IDF long dumps into one frame (same schema)."""
    frames = []
    for p in (preds_path, tfidf_path):
        if p and os.path.exists(p):
            frames.append(pd.read_csv(p))
        else:
            print(f"  (note) missing {p}, skipping")
    if not frames:
        raise SystemExit("No prediction dumps found.")
    df = pd.concat(frames, ignore_index=True)
    df['pred'] = (df['prob'] >= THRESH).astype(int)
    return df


def collapse_seeds(df):
    """One row per (dataset, variant, email_idx): mean prob across seeds, carried cols kept."""
    carry = [c for c in ['label', 'condition', 'generator_model', 'scenario_id', 'register', 'sample_id']
             if c in df.columns]
    agg = {'prob': 'mean'}
    g = df.groupby(['dataset', 'variant', 'email_idx'], as_index=False).agg(
        prob=('prob', 'mean'), **{c: (c, 'first') for c in carry})
    g['pred'] = (g['prob'] >= THRESH).astype(int)
    return g


def attach_pairing(mpg_rows, features_path):
    """Add source_sample_id to the mpg rows via sample_id -> source_sample_id from the feature file."""
    if not features_path or not os.path.exists(features_path):
        return mpg_rows
    feat = pd.read_csv(features_path)
    if 'sample_id' in feat.columns and 'source_sample_id' in feat.columns:
        m = dict(zip(feat['sample_id'], feat['source_sample_id']))
        mpg_rows = mpg_rows.copy()
        mpg_rows['source_sample_id'] = mpg_rows['sample_id'].map(m)
    return mpg_rows


# ----------------------------------------------------------------------------- helpers
def recall(sub):
    pos = sub[sub['label'] == 1]
    return pos['pred'].mean() if len(pos) else np.nan


def fpr(sub):
    neg = sub[sub['label'] == 0]
    return neg['pred'].mean() if len(neg) else np.nan


def safe_auc(pos_probs, neg_probs):
    if len(pos_probs) == 0 or len(neg_probs) == 0:
        return np.nan
    y = np.r_[np.ones(len(pos_probs)), np.zeros(len(neg_probs))]
    s = np.r_[pos_probs, neg_probs]
    try:
        return roc_auc_score(y, s)
    except ValueError:
        return np.nan


def mcnemar_exact(detect_a, detect_b):
    """Exact paired McNemar on two boolean arrays (same items). Returns (b, c, p_value)."""
    a = np.asarray(detect_a).astype(bool)
    b = np.asarray(detect_b).astype(bool)
    only_a = int(np.sum(a & ~b))   # a detects, b misses
    only_b = int(np.sum(~a & b))   # b detects, a misses
    n = only_a + only_b
    if n == 0:
        return only_a, only_b, 1.0
    p = binomtest(min(only_a, only_b), n, 0.5, alternative='two-sided').pvalue
    return only_a, only_b, p


def boot_diff(items_a, items_b, n_boot=2000):
    """Paired bootstrap of mean(a)-mean(b) over matched items; returns (diff, lo, hi)."""
    a = np.asarray(items_a, float)
    b = np.asarray(items_b, float)
    if len(a) == 0 or len(a) != len(b):
        return np.nan, np.nan, np.nan
    idx = np.arange(len(a))
    diffs = [a[s].mean() - b[s].mean() for s in (rng.choice(idx, len(idx), replace=True) for _ in range(n_boot))]
    return float(a.mean() - b.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def cohens_d(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or len(y) < 2:
        return np.nan
    nx, ny = len(x), len(y)
    sp = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / max(1, nx + ny - 2))
    return (x.mean() - y.mean()) / sp if sp > 0 else np.nan


# ----------------------------------------------------------------------------- tables
def table_mpg_summary(mpg, out):
    """Per-variant recall/FPR/AUC on the matched-pair set (seed-averaged probabilities)."""
    rows = []
    for variant, sub in mpg.groupby('variant'):
        by = {c: sub[sub['condition'] == c] for c in ['P', 'B_hard', 'B_easy', 'P_R', 'B_hard_R']}
        rows.append({
            'variant': variant,
            'recall_P': recall(by['P']),
            'recall_P_R': recall(by['P_R']),
            'FPR_B_hard': fpr(by['B_hard']),
            'FPR_B_easy': fpr(by['B_easy']),
            'AUC_P_vs_Bhard': safe_auc(by['P']['prob'].values, by['B_hard']['prob'].values),   # H1
            'AUC_P_vs_Beasy': safe_auc(by['P']['prob'].values, by['B_easy']['prob'].values),
            'n_P': len(by['P']), 'n_Bhard': len(by['B_hard']), 'n_Beasy': len(by['B_easy']),
        })
    t = pd.DataFrame(rows).sort_values('variant')
    t['intent_gap'] = t['AUC_P_vs_Beasy'] - t['AUC_P_vs_Bhard']   # H1: big gap = reads cues, not intent
    t.round(4).to_csv(os.path.join(out, 'table_mpg_summary.csv'), index=False)
    return t


def table_per_generator(mpg, out):
    """Recall on P, per variant x generator (the disaggregation the reviewer required)."""
    p = mpg[mpg['condition'] == 'P']
    if 'generator_model' not in p.columns:
        return None
    t = p.groupby(['variant', 'generator_model']).apply(
        lambda s: pd.Series({'recall_P': s['pred'].mean(), 'n': len(s)}), include_groups=False
    ).reset_index()
    t.round(4).to_csv(os.path.join(out, 'table_recall_by_generator.csv'), index=False)
    return t


def table_ablation_vs_baseline(mpg, out):
    """Paired McNemar + bootstrap: each variant vs baseline on P emails (matched by sample_id)."""
    p = mpg[mpg['condition'] == 'P']
    if 'baseline' not in set(p['variant']):
        return None
    base = p[p['variant'] == 'baseline'].set_index('sample_id')
    rows = []
    for variant, sub in p.groupby('variant'):
        if variant == 'baseline':
            continue
        sub = sub.set_index('sample_id')
        common = base.index.intersection(sub.index)
        if len(common) == 0:
            continue
        a = sub.loc[common, 'pred'].values      # variant
        b = base.loc[common, 'pred'].values     # baseline
        only_v, only_b, p_val = mcnemar_exact(a, b)
        d, lo, hi = boot_diff(a, b)
        rows.append({'variant': variant, 'n_pairs': len(common),
                     'recall_variant': a.mean(), 'recall_baseline': b.mean(),
                     'delta_recall': d, 'ci_lo': lo, 'ci_hi': hi,
                     'mcnemar_only_variant': only_v, 'mcnemar_only_baseline': only_b,
                     'mcnemar_p': p_val})
    t = pd.DataFrame(rows).sort_values('delta_recall', ascending=False)
    t.round(4).to_csv(os.path.join(out, 'table_ablation_vs_baseline.csv'), index=False)
    return t


def table_register_effect(mpg, out):
    """H2: paired P vs P_R (intent constant, register changed), matched via source_sample_id."""
    if 'source_sample_id' not in mpg.columns:
        print("  (H2) no source_sample_id available; reporting unpaired recall only.")
    rows = []
    for variant, sub in mpg.groupby('variant'):
        P = sub[sub['condition'] == 'P']
        PR = sub[sub['condition'] == 'P_R']
        rec_P, rec_PR = recall(P), recall(PR)
        entry = {'variant': variant, 'recall_P': rec_P, 'recall_P_R': rec_PR,
                 'unpaired_drop': (rec_P - rec_PR) if pd.notna(rec_P) and pd.notna(rec_PR) else np.nan}
        if 'source_sample_id' in sub.columns and len(PR):
            # pair each P_R to its source P (by that P's sample_id)
            p_by_id = P.set_index('sample_id')['pred']
            pr = PR.dropna(subset=['source_sample_id'])
            pr = pr[pr['source_sample_id'].isin(p_by_id.index)]
            if len(pr):
                a = p_by_id.loc[pr['source_sample_id']].values   # original P detection
                b = pr['pred'].values                            # rewritten P_R detection
                only_P, only_PR, p_val = mcnemar_exact(a, b)
                entry.update({'n_pairs': len(pr), 'paired_P_detected': a.mean(),
                              'paired_PR_detected': b.mean(), 'mcnemar_p': p_val})
        rows.append(entry)
    t = pd.DataFrame(rows).sort_values('variant')
    t.round(4).to_csv(os.path.join(out, 'table_register_effect.csv'), index=False)
    return t


def table_feature_authorship(features_path, out):
    """H3: for each feature, Cohen's d(P vs B_hard) vs spread across generators."""
    if not features_path or not os.path.exists(features_path):
        return None
    f = pd.read_csv(features_path)
    if 'condition' not in f.columns:
        return None
    rows = []
    for feat in [c for c in FEATURES if c in f.columns]:
        P = f[f['condition'] == 'P'][feat].dropna()
        BH = f[f['condition'] == 'B_hard'][feat].dropna()
        d_intent = cohens_d(P, BH)
        # spread across generators within P: (max-min generator mean) / pooled sd
        gen_spread = np.nan
        if 'generator_model' in f.columns:
            gmeans = f[f['condition'] == 'P'].groupby('generator_model')[feat].mean()
            sd = f[f['condition'] == 'P'][feat].std(ddof=1)
            gen_spread = (gmeans.max() - gmeans.min()) / sd if sd and sd > 0 else np.nan
        rows.append({'feature': feat, 'cohens_d_P_vs_Bhard': d_intent,
                     'generator_spread_sd': gen_spread})
    t = pd.DataFrame(rows)
    t.round(4).to_csv(os.path.join(out, 'table_feature_authorship.csv'), index=False)
    return t


def table_conventional(preds, out):
    """In-distribution performance per variant on the 'test' set, seed mean +/- SD (no-cost check)."""
    test = preds[preds['dataset'] == 'test']
    if len(test) == 0:
        return None
    rows = []
    for (variant, seed), sub in test.groupby(['variant', 'seed']):
        pos = sub[sub['label'] == 1]
        tp = ((pos['prob'] >= THRESH)).sum()
        rec = tp / max(1, len(pos))
        pred = (sub['prob'] >= THRESH).astype(int)
        tpf = ((pred == 1) & (sub['label'] == 1)).sum()
        prec = tpf / max(1, (pred == 1).sum())
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        rows.append({'variant': variant, 'seed': seed, 'recall': rec, 'precision': prec, 'f1': f1})
    per_seed = pd.DataFrame(rows)
    summ = per_seed.groupby('variant').agg(
        f1_mean=('f1', 'mean'), f1_sd=('f1', 'std'),
        recall_mean=('recall', 'mean'), recall_sd=('recall', 'std')).reset_index()
    summ.round(4).to_csv(os.path.join(out, 'table_conventional_by_seed.csv'), index=False)
    return summ


def table_seed_variance(mpg, out):
    """Recall on P across seeds per variant (needs the un-collapsed dump)."""
    # mpg here is seed-collapsed; recompute from raw is done in main via a separate pass.
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds', default='predictions_long.csv')
    ap.add_argument('--tfidf', default='tfidf_predictions_long.csv')
    ap.add_argument('--features', default='mpg_emails_features.csv')
    ap.add_argument('--out', default='analysis_out')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    raw = load_preds(args.preds, args.tfidf)
    collapsed = collapse_seeds(raw)
    mpg = collapsed[collapsed['dataset'] == MPG].copy()
    if len(mpg) == 0:
        print(f"  WARNING: no rows with dataset=='{MPG}'. Datasets present: {sorted(collapsed['dataset'].unique())}")
    mpg = attach_pairing(mpg, args.features)

    print("H1/overview ->", end=' ')
    print(table_mpg_summary(mpg, args.out).to_string(index=False) if len(mpg) else "skipped")
    table_per_generator(mpg, args.out)
    table_ablation_vs_baseline(mpg, args.out)
    table_register_effect(mpg, args.out)
    table_feature_authorship(args.features, args.out)
    table_conventional(raw, args.out)

    # Seed variance on P recall, per variant (from the raw DistilBERT dump).
    p_raw = raw[(raw['dataset'] == MPG) & (raw['condition'] == 'P')]
    if len(p_raw):
        sv = p_raw.groupby(['variant', 'seed']).apply(
            lambda s: (s['prob'] >= THRESH).mean(), include_groups=False).reset_index(name='recall_P')
        svs = sv.groupby('variant').agg(recall_mean=('recall_P', 'mean'),
                                        recall_sd=('recall_P', 'std'), n_seeds=('recall_P', 'size')).reset_index()
        svs.round(4).to_csv(os.path.join(args.out, 'table_seed_variance.csv'), index=False)

    print(f"\nAll tables written to {args.out}/")


if __name__ == '__main__':
    main()
