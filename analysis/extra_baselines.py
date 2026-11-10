"""Review-2 experiments: a stronger neural baseline (RoBERTa-base) and an external benchmark
(E-PhishLLM, English subset), analysed with the same rules as paper_stats.py.

Inputs (results_kaggle/ holds the Kaggle outputs of kaggle/train_backbone.py and tfidf_baseline.py):
  predictions_long.csv, tfidf_predictions_long.csv        original paper runs (DistilBERT variants, TF-IDF)
  results_kaggle/predictions_{roberta,distilbert}_shard*.csv   new runs, 5 seeds each
  results_kaggle/tfidf_ephishllm.csv                        TF-IDF scored on E-PhishLLM
  mpg_emails_features.csv                                   pairing (sample_id -> source_sample_id)

Every model is scored on the seed-averaged probability at threshold 0.5, as in the paper.
'distilbert' is a fresh re-run of the paper's text-only baseline with train_backbone.py; it is a
reproducibility check for 'baseline' and the like-for-like comparator for RoBERTa.

  python analysis/extra_baselines.py      -> analysis_out/review2/*.csv
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_stats import THRESH, attach_pairing, boot_diff, cohens_d, mcnemar_exact  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RK = os.path.join(ROOT, 'results_kaggle')
OUT = os.path.join(ROOT, 'analysis_out', 'review2')
MODELS = ['baseline', 'distilbert', 'roberta', 'tfidf']
NAME = {'baseline': 'DistilBERT (paper run)', 'distilbert': 'DistilBERT (re-run)',
        'roberta': 'RoBERTa-base', 'tfidf': 'TF-IDF + LR'}
N_BOOT = 2000


def load():
    frames = [pd.read_csv(os.path.join(ROOT, 'predictions_long.csv')),
              pd.read_csv(os.path.join(ROOT, 'tfidf_predictions_long.csv'))]
    frames += [pd.read_csv(p) for p in sorted(glob.glob(os.path.join(RK, 'predictions_*_shard*.csv')))]
    frames.append(pd.read_csv(os.path.join(RK, 'tfidf_ephishllm.csv')))
    raw = pd.concat(frames, ignore_index=True)
    raw['dataset'] = raw['dataset'].replace({'ephishllm_en': 'ephishllm'})
    raw = raw[raw['variant'].isin(MODELS)].copy()
    raw['pred'] = (raw['prob'] >= THRESH).astype(int)
    return raw


def collapse(raw):
    carry = [c for c in ['label', 'condition', 'generator_model', 'scenario_id', 'register', 'sample_id']
             if c in raw.columns]
    g = raw.groupby(['dataset', 'variant', 'email_idx'], as_index=False).agg(
        prob=('prob', 'mean'), n_seeds=('seed', 'nunique'), **{c: (c, 'first') for c in carry})
    g['pred'] = (g['prob'] >= THRESH).astype(int)
    return g


def metrics(y, prob):
    y, prob = np.asarray(y), np.asarray(prob)
    pred = (prob >= THRESH).astype(int)
    tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
    fn, tn = int(((pred == 0) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return {'accuracy': (tp + tn) / len(y), 'precision': prec, 'recall': rec,
            'f1': 2 * prec * rec / max(1e-12, prec + rec), 'fpr': fp / max(1, fp + tn),
            'auc': roc_auc_score(y, prob)}


def auc_pair(s, a, b):
    x = s[s['condition'].isin([a, b])]
    return roc_auc_score((x['condition'] == a).astype(int), x['prob'])


def cluster_ci(s, a, b, rng):
    """95% CI for AUC(a vs b), resampling the 30 generator x scenario cells (as in the paper)."""
    x = s[s['condition'].isin([a, b])].reset_index(drop=True)
    cells = list(x.groupby(['generator_model', 'scenario_id']).groups.values())
    vals = []
    for _ in range(N_BOOT):
        idx = np.concatenate([cells[i] for i in rng.integers(0, len(cells), len(cells))])
        t = x.loc[idx]
        vals.append(roc_auc_score((t['condition'] == a).astype(int), t['prob']))
    return np.percentile(vals, [2.5, 97.5])


def holm(p):
    p = np.asarray(p, float)
    order, out, run = np.argsort(p), np.empty(len(p)), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - r) * p[i]))
        out[i] = run
    return out


# ----------------------------------------------------------------------------- tables
def t_conventional(raw):
    """Kaggle/Enron held-out test split, per-seed mean +/- SD."""
    rows = []
    for (v, seed), s in raw[raw['dataset'] == 'test'].groupby(['variant', 'seed']):
        rows.append({'model': v, 'seed': seed, **metrics(s['label'], s['prob'])})
    d = pd.DataFrame(rows)
    out = d.groupby('model').agg(n_seeds=('seed', 'size'), f1_mean=('f1', 'mean'), f1_sd=('f1', 'std'),
                                 recall_mean=('recall', 'mean'), recall_sd=('recall', 'std'),
                                 auc_mean=('auc', 'mean')).reindex(MODELS).reset_index()
    return out


def t_matched_pairs(mpg):
    """H1 and H4 for every model: recall, FPR and AUCs on the matched-pair set."""
    rng = np.random.default_rng(0)
    rows = []
    for v in MODELS:
        s = mpg[mpg['variant'] == v]
        c = {k: s[s['condition'] == k] for k in ['P', 'B_hard', 'B_easy', 'P_R', 'B_hard_R']}
        lo, hi = cluster_ci(s, 'P', 'B_hard', rng)
        rows.append({'model': v, 'recall_P': c['P']['pred'].mean(), 'recall_P_R': c['P_R']['pred'].mean(),
                     'FPR_B_hard': c['B_hard']['pred'].mean(), 'FPR_B_easy': c['B_easy']['pred'].mean(),
                     'FPR_B_hard_R': c['B_hard_R']['pred'].mean(),
                     'AUC_P_vs_Bhard': auc_pair(s, 'P', 'B_hard'), 'AUC_P_vs_Bhard_lo': lo,
                     'AUC_P_vs_Bhard_hi': hi, 'AUC_P_vs_Beasy': auc_pair(s, 'P', 'B_easy'),
                     'H1_supported (AUC<=0.75)': auc_pair(s, 'P', 'B_hard') <= 0.75,
                     'H4_supported (FPR_Bhard>=0.40)': c['B_hard']['pred'].mean() >= 0.40})
    return pd.DataFrame(rows)


def t_vs_baseline(mpg):
    """Paired comparisons with the paper's DistilBERT baseline on the same emails:
    recall on P and false alarms on B_hard (exact McNemar + paired bootstrap, Holm across all)."""
    rows = []
    for cond, metric in [('P', 'recall_P'), ('B_hard', 'FPR_B_hard')]:
        sub = mpg[mpg['condition'] == cond]
        base = sub[sub['variant'] == 'baseline'].set_index('sample_id')['pred']
        for v in ['distilbert', 'roberta', 'tfidf']:
            other = sub[sub['variant'] == v].set_index('sample_id')['pred']
            ids = base.index.intersection(other.index)
            a, b = other.loc[ids].values, base.loc[ids].values
            only_v, only_b, p = mcnemar_exact(a, b)
            d, lo, hi = boot_diff(a, b)
            rows.append({'metric': metric, 'model': v, 'n': len(ids), 'model_value': a.mean(),
                         'baseline_value': b.mean(), 'delta': d, 'ci_lo': lo, 'ci_hi': hi,
                         'only_model': only_v, 'only_baseline': only_b, 'mcnemar_p': p})
    t = pd.DataFrame(rows)
    t['holm_p'] = holm(t['mcnemar_p'])
    return t


def t_register(mpg):
    """H2: P vs its own register rewrite P_R (paired via source_sample_id)."""
    rows = []
    for v in MODELS:
        s = mpg[mpg['variant'] == v]
        p = s[s['condition'] == 'P'].set_index('sample_id')['pred']
        pr = s[(s['condition'] == 'P_R') & s['source_sample_id'].isin(p.index)]
        a, b = p.loc[pr['source_sample_id']].values, pr['pred'].values
        only_p, only_pr, pval = mcnemar_exact(a, b)
        rows.append({'model': v, 'n_pairs': len(pr), 'P_detected': a.mean(), 'P_R_detected': b.mean(),
                     'only_P': only_p, 'only_P_R': only_pr, 'mcnemar_p': pval})
    return pd.DataFrame(rows)


def t_generator(mpg):
    """Recall on P and FPR on B_hard per generator (shows whether a stronger encoder helps everywhere)."""
    rows = []
    for (v, g), s in mpg[mpg['condition'].isin(['P', 'B_hard'])].groupby(['variant', 'generator_model']):
        rows.append({'model': v, 'generator': g,
                     'recall_P': s[s['condition'] == 'P']['pred'].mean(),
                     'FPR_B_hard': s[s['condition'] == 'B_hard']['pred'].mean(),
                     'AUC_P_vs_Bhard': auc_pair(s, 'P', 'B_hard')})
    return pd.DataFrame(rows)


def t_score_shift(mpg):
    """How far apart P and B_hard probabilities are (Cohen's d on seed-averaged probability)."""
    rows = []
    for v in MODELS:
        s = mpg[mpg['variant'] == v]
        rows.append({'model': v, 'mean_prob_P': s[s['condition'] == 'P']['prob'].mean(),
                     'mean_prob_B_hard': s[s['condition'] == 'B_hard']['prob'].mean(),
                     'mean_prob_B_easy': s[s['condition'] == 'B_easy']['prob'].mean(),
                     'cohens_d_P_vs_Bhard': cohens_d(s[s['condition'] == 'P']['prob'],
                                                     s[s['condition'] == 'B_hard']['prob'])})
    return pd.DataFrame(rows)


def t_ephishllm(raw, col):
    """External benchmark: seed-averaged ensemble, per-seed spread, and a bootstrap CI on F1."""
    rng = np.random.default_rng(0)
    eph = col[col['dataset'] == 'ephishllm']
    raw_e = raw[raw['dataset'] == 'ephishllm']
    rows = []
    for v in ['distilbert', 'roberta', 'tfidf']:
        s = eph[eph['variant'] == v].sort_values('email_idx')
        if s.empty:
            continue
        m = metrics(s['label'], s['prob'])
        y, pr = s['label'].values, s['prob'].values
        f1s = []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(y), len(y))
            f1s.append(metrics(y[i], pr[i])['f1'] if len(set(y[i])) == 2 else np.nan)
        per_seed = [metrics(g['label'], g['prob'])['f1'] for _, g in raw_e[raw_e['variant'] == v].groupby('seed')]
        rows.append({'model': v, 'n': len(s), 'n_phish': int((y == 1).sum()), **m,
                     'f1_lo': np.nanpercentile(f1s, 2.5), 'f1_hi': np.nanpercentile(f1s, 97.5),
                     'f1_seed_min': min(per_seed), 'f1_seed_max': max(per_seed), 'n_seeds': len(per_seed)})
    return pd.DataFrame(rows)


def t_ephish_threshold(col):
    """Is the drop on E-PhishLLM a calibration shift or a ranking failure? Best-F1 threshold vs 0.5."""
    rows = []
    eph = col[col['dataset'] == 'ephishllm']
    for v in ['distilbert', 'roberta', 'tfidf']:
        s = eph[eph['variant'] == v]
        if s.empty:
            continue
        best = max(((t, metrics(s['label'], (s['prob'] >= t).astype(float))['f1'])
                    for t in np.r_[0.001, 0.01, np.arange(0.05, 1.0, 0.05), 0.99, 0.999]), key=lambda z: z[1])
        rows.append({'model': v, 'f1_at_0.5': metrics(s['label'], s['prob'])['f1'],
                     'oracle_threshold': best[0], 'f1_at_oracle_threshold': best[1],
                     'median_prob_phish': s[s['label'] == 1]['prob'].median(),
                     'median_prob_legit': s[s['label'] == 0]['prob'].median()})
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    raw = load()
    col = collapse(raw)
    mpg = attach_pairing(col[col['dataset'] == 'mpg_emails'].copy(), os.path.join(ROOT, 'mpg_emails_features.csv'))
    print('seeds per model/dataset:\n', raw.groupby(['dataset', 'variant'])['seed'].nunique().unstack(), '\n')

    tables = {'conventional_test': t_conventional(raw), 'matched_pairs': t_matched_pairs(mpg),
              'vs_baseline_paired': t_vs_baseline(mpg), 'register_H2': t_register(mpg),
              'by_generator': t_generator(mpg), 'score_shift': t_score_shift(mpg),
              'ephishllm': t_ephishllm(raw, col), 'ephishllm_threshold': t_ephish_threshold(col)}
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 30)
    for name, t in tables.items():
        t.round(4).to_csv(os.path.join(OUT, f'table_{name}.csv'), index=False)
        print(f'== {name}\n{t.round(3).to_string(index=False)}\n')
    print('written to', OUT)


if __name__ == '__main__':
    main()
