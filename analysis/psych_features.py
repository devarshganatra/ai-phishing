"""Review 2 (High): psychological features tested one by one, raw and length-normalised.

The paper's "psychological" group sums three keyword counts (urgency, fear, authority). Here each
is examined on its own, as a raw count, per 100 words, and as presence (>= 1), and the authority
score is split into its two parts (job-title keywords and spaCy ORG entities), because the ORG
part grows with length and with company names rather than with persuasion.

For every feature:
  train  AUC(phishing vs legitimate) on the Kaggle/Enron training split: is it a cue the
         detectors could have learned?
  mpg    AUC and Cohen's d for P vs B_hard (intent at fixed cues), P vs B_easy, and B_hard vs
         B_easy (design check: B_hard should carry the cues too)
  spread range of P means across the three generators, in pooled SD units
Mann-Whitney p-values for P vs B_hard are Holm-adjusted across all rows.

  python analysis/psych_features.py   -> analysis_out/review2/table_psych_features.csv
"""
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'analysis_out', 'review2')
TITLES = [r'\bceo\b', r'\bcfo\b', r'\bhr\b', r'\bit department\b', r'\badmin\b', r'\bsupport\b',
          r'\bmanager\b', r'\bdirector\b', r'\bpresident\b', r'\bhelpdesk\b']


def add_parts(df):
    low = df['text'].fillna('').str.lower()
    df['authority_titles'] = sum(low.str.count(p) for p in TITLES)
    df['authority_org'] = (df['authority_score'] - df['authority_titles']).clip(lower=0)
    base = ['urgency_score', 'fear_score', 'authority_score', 'authority_titles', 'authority_org']
    df['psych_total'] = df[['urgency_score', 'fear_score', 'authority_score']].sum(axis=1)
    wc = df['word_count'].clip(lower=1)
    for c in base + ['psych_total']:
        df[c + '_per100w'] = 100 * df[c] / wc
        df[c + '_present'] = (df[c] > 0).astype(int)
    return df


def d(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    sp = np.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) / (len(x) + len(y) - 2))
    return (x.mean() - y.mean()) / sp if sp > 0 else np.nan


def auc(pos, neg):
    if np.ptp(np.r_[pos, neg]) == 0:
        return 0.5
    return roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])


def holm(p):
    p = np.asarray(p, float)
    o, out, run = np.argsort(p), np.empty(len(p)), 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(p) - r) * p[i]))
        out[i] = run
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    tr = add_parts(pd.read_csv(os.path.join(ROOT, 'data', 'train_features.csv')))
    mp = add_parts(pd.read_csv(os.path.join(ROOT, 'mpg_emails_features.csv')))
    feats = []
    for base in ['urgency_score', 'fear_score', 'authority_score', 'authority_titles', 'authority_org', 'psych_total']:
        feats += [(base, 'count', base), (base, 'per 100 words', base + '_per100w'), (base, 'present', base + '_present')]
    feats.append(('word_count', 'count', 'word_count'))

    rows = []
    for base, form, col in feats:
        P, BH, BE = (mp[mp.condition == c][col].values for c in ['P', 'B_hard', 'B_easy'])
        gm = mp[mp.condition == 'P'].groupby('generator_model')[col].mean()
        sd = mp[mp.condition == 'P'][col].std(ddof=1)
        rows.append({
            'feature': base, 'form': form,
            'train_mean_phish': tr[tr.label == 1][col].mean(), 'train_mean_legit': tr[tr.label == 0][col].mean(),
            'train_AUC': auc(tr[tr.label == 1][col].values, tr[tr.label == 0][col].values),
            'mean_P': P.mean(), 'mean_B_hard': BH.mean(), 'mean_B_easy': BE.mean(),
            'AUC_P_vs_Bhard': auc(P, BH), 'd_P_vs_Bhard': d(P, BH),
            'AUC_P_vs_Beasy': auc(P, BE), 'AUC_Bhard_vs_Beasy': auc(BH, BE),
            'p_P_vs_Bhard': mannwhitneyu(P, BH).pvalue if np.ptp(np.r_[P, BH]) else 1.0,
            'generator_spread_sd': (gm.max() - gm.min()) / sd if sd > 0 else np.nan})
    t = pd.DataFrame(rows)
    t['holm_P_vs_Bhard'] = holm(t['p_P_vs_Bhard'])
    t.round(4).to_csv(os.path.join(OUT, 'table_psych_features.csv'), index=False)
    pd.set_option('display.width', 250)
    cols = ['feature', 'form', 'train_AUC', 'mean_P', 'mean_B_hard', 'mean_B_easy', 'AUC_P_vs_Bhard',
            'd_P_vs_Bhard', 'holm_P_vs_Bhard', 'AUC_Bhard_vs_Beasy', 'generator_spread_sd']
    print(t[cols].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
