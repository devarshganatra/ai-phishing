"""Publication figures for the Intent-vs-Register study.

Reads the same dumps as paper_stats.py and writes PNGs mapped to the reviewer's required
visuals. Each figure is self-contained and skips gracefully if its inputs are absent, so you
can run it the moment the Kaggle dumps land.

Figures:
  fig_collapse            conventional (test) recall vs matched-pair P recall, per variant
  fig_H1_intent_auc       AUC(P vs B_hard) vs AUC(P vs B_easy) -> intent-blindness gap
  fig_H2_register         recall on P vs P_R (intent constant, register changed)
  fig_H3_features         six handcrafted features, violins by condition (P/B_hard/B_easy)
  fig_H3_by_generator     one feature's distribution split by generator (authorship signal)
  fig_H4_false_alarm      FPR on B_hard vs B_easy per model (lexical false alarms)
  fig_tfidf_coef          top TF-IDF safe/phishing tokens (provenance, not intent)
  fig_ablation            per-feature recall on P with seed error bars
  fig_forest              paired delta-recall (variant - baseline) with bootstrap 95% CI
  fig_heatmap             generator x condition recall heatmap (baseline model)
  fig_learning_curve      TF-IDF recall vs training size
  fig_score_hist          P vs B_hard probability histograms (shortcut-learning diagnostic)

Usage:
  python analysis/paper_figures.py \
      --preds predictions_long.csv --tfidf tfidf_predictions_long.csv \
      --features mpg_emails_features.csv --train data/train.csv \
      --learning_curve tfidf_learning_curve.csv --out analysis_out
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paper_stats import (load_preds, collapse_seeds, attach_pairing, recall, fpr, safe_auc,
                         boot_diff, FEATURES, MPG, THRESH)

plt.rcParams.update({'figure.dpi': 130, 'savefig.dpi': 150, 'font.size': 10,
                     'axes.grid': True, 'grid.alpha': 0.25, 'axes.axisbelow': True})
BLUE, ORANGE, GREEN, RED, GREY = '#2d6cdf', '#e8821a', '#2a9d4a', '#d1495b', '#8a8f98'
VARIANT_ORDER = ['baseline', 'perplexity', 'grammar', 'length', 'behavioral', 'psychological', 'full', 'tfidf']


def order_variants(vs):
    vs = list(vs)
    return [v for v in VARIANT_ORDER if v in vs] + [v for v in vs if v not in VARIANT_ORDER]


def save(fig, out, name):
    path = os.path.join(out, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"  wrote {path}")


# ----------------------------------------------------------------------------- figures
def fig_collapse(mpg, preds, out):
    test = preds[preds['dataset'] == 'test']
    variants = order_variants(set(mpg['variant']))
    conv, mpgp = [], []
    for v in variants:
        tv = test[(test['variant'] == v) & (test['label'] == 1)]
        conv.append((tv['prob'] >= THRESH).mean() if len(tv) else np.nan)
        pv = mpg[(mpg['variant'] == v) & (mpg['condition'] == 'P')]
        mpgp.append(recall(pv))
    x = np.arange(len(variants)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(x - w/2, conv, w, label='Conventional test (P)', color=BLUE)
    ax.bar(x + w/2, mpgp, w, label='Matched-pair P', color=ORANGE)
    ax.set_xticks(x); ax.set_xticklabels(variants, rotation=30, ha='right')
    ax.set_ylabel('Recall'); ax.set_ylim(0, 1.02)
    ax.set_title('In-distribution vs matched-pair recall (the collapse, per variant)')
    ax.legend()
    save(fig, out, 'fig_collapse.png')


def fig_H1_intent_auc(mpg, out):
    variants = order_variants(set(mpg['variant']))
    a_bh, a_be = [], []
    for v in variants:
        s = mpg[mpg['variant'] == v]
        P = s[s['condition'] == 'P']['prob'].values
        a_bh.append(safe_auc(P, s[s['condition'] == 'B_hard']['prob'].values))
        a_be.append(safe_auc(P, s[s['condition'] == 'B_easy']['prob'].values))
    x = np.arange(len(variants)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(x - w/2, a_be, w, label='AUC(P vs B_easy)', color=GREEN)
    ax.bar(x + w/2, a_bh, w, label='AUC(P vs B_hard)  [intent]', color=RED)
    ax.axhline(0.5, color=GREY, ls='--', lw=1, label='chance')
    ax.set_xticks(x); ax.set_xticklabels(variants, rotation=30, ha='right')
    ax.set_ylabel('AUC'); ax.set_ylim(0, 1.02)
    ax.set_title('H1: can the detector read intent? (P vs matched benign twin)')
    ax.legend()
    save(fig, out, 'fig_H1_intent_auc.png')


def fig_H2_register(mpg, out):
    variants = order_variants(set(mpg['variant']))
    rp, rpr = [], []
    for v in variants:
        s = mpg[mpg['variant'] == v]
        rp.append(recall(s[s['condition'] == 'P']))
        rpr.append(recall(s[s['condition'] == 'P_R']))
    if all(np.isnan(rpr)):
        print("  (skip fig_H2_register: no P_R rows)")
        return
    x = np.arange(len(variants)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(x - w/2, rp, w, label='recall(P)', color=BLUE)
    ax.bar(x + w/2, rpr, w, label='recall(P_R)  [register changed]', color=ORANGE)
    ax.set_xticks(x); ax.set_xticklabels(variants, rotation=30, ha='right')
    ax.set_ylabel('Recall'); ax.set_ylim(0, 1.02)
    ax.set_title('H2: does rewriting the register (intent held constant) change detection?')
    ax.legend()
    save(fig, out, 'fig_H2_register.png')


def fig_H3_features(features_path, out):
    if not features_path or not os.path.exists(features_path):
        return
    f = pd.read_csv(features_path)
    if 'condition' not in f.columns:
        return
    conds = [c for c in ['P', 'B_hard', 'B_easy'] if c in set(f['condition'])]
    feats = [c for c in FEATURES if c in f.columns]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, feat in zip(axes.ravel(), feats):
        data = [f[f['condition'] == c][feat].dropna().values for c in conds]
        parts = ax.violinplot(data, showmedians=True, showextrema=False)
        for pc, col in zip(parts['bodies'], [BLUE, RED, GREEN]):
            pc.set_facecolor(col); pc.set_alpha(0.6)
        ax.set_xticks(range(1, len(conds) + 1)); ax.set_xticklabels(conds)
        ax.set_title(feat)
    for ax in axes.ravel()[len(feats):]:
        ax.axis('off')
    fig.suptitle('H3: handcrafted feature distributions by condition '
                 '(P vs B_hard overlap = features do not encode intent)', y=1.0)
    save(fig, out, 'fig_H3_features.png')


def fig_H3_by_generator(features_path, out, feat='perplexity'):
    if not features_path or not os.path.exists(features_path):
        return
    f = pd.read_csv(features_path)
    if 'generator_model' not in f.columns or feat not in f.columns:
        return
    P = f[f['condition'] == 'P'] if 'condition' in f.columns else f
    gens = sorted(P['generator_model'].dropna().unique())
    data = [P[P['generator_model'] == g][feat].dropna().values for g in gens]
    if not any(len(d) for d in data):
        return
    fig, ax = plt.subplots(figsize=(8, 4.2))
    parts = ax.violinplot(data, showmedians=True, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor(ORANGE); pc.set_alpha(0.6)
    ax.set_xticks(range(1, len(gens) + 1))
    ax.set_xticklabels([g.split(':')[-1] for g in gens], rotation=20, ha='right')
    ax.set_ylabel(feat)
    ax.set_title(f'H3: {feat} of phishing (P) by generator — a generator signature, not intent')
    save(fig, out, 'fig_H3_by_generator.png')


def fig_H4_false_alarm(mpg, out):
    variants = order_variants(set(mpg['variant']))
    bh, be = [], []
    for v in variants:
        s = mpg[mpg['variant'] == v]
        bh.append(fpr(s[s['condition'] == 'B_hard']))
        be.append(fpr(s[s['condition'] == 'B_easy']))
    x = np.arange(len(variants)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(x - w/2, be, w, label='FPR on B_easy', color=GREEN)
    ax.bar(x + w/2, bh, w, label='FPR on B_hard  [shared cues]', color=RED)
    ax.set_xticks(x); ax.set_xticklabels(variants, rotation=30, ha='right')
    ax.set_ylabel('False-positive rate'); ax.set_ylim(0, 1.02)
    ax.set_title('H4: do benign twins with phishing-like cues trigger false alarms?')
    ax.legend()
    save(fig, out, 'fig_H4_false_alarm.png')


def fig_tfidf_coef(train_path, out, topn=15):
    if not train_path or not os.path.exists(train_path):
        print("  (skip fig_tfidf_coef: no --train for refit)")
        return
    try:
        from tfidf_baseline import build_pipeline
    except Exception as e:
        print(f"  (skip fig_tfidf_coef: {e})")
        return
    tr = pd.read_csv(train_path).dropna(subset=['text', 'label'])
    pipe = build_pipeline().fit(tr['text'].astype(str), tr['label'].astype(int))
    names = np.array(pipe.named_steps['tfidf'].get_feature_names_out())
    coef = pipe.named_steps['clf'].coef_[0]
    order = np.argsort(coef)
    topn = min(topn, len(names) // 2)   # robust to tiny vocabularies
    if topn < 1:
        print("  (skip fig_tfidf_coef: vocabulary too small)")
        return
    safe_i, phish_i = order[:topn], order[-topn:]
    toks = list(names[safe_i]) + list(names[phish_i])
    vals = list(coef[safe_i]) + list(coef[phish_i])
    cols = [GREEN] * len(safe_i) + [RED] * len(phish_i)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(toks)), vals, color=cols)
    ax.set_yticks(range(len(toks))); ax.set_yticklabels(toks)
    ax.axvline(0, color='k', lw=0.8)
    ax.set_xlabel('logistic-regression coefficient')
    ax.set_title('TF-IDF top tokens: provenance (enron/years) vs phishing cues')
    save(fig, out, 'fig_tfidf_coef.png')
    pd.DataFrame({'token': toks, 'coef': vals,
                  'side': ['safe'] * len(safe_i) + ['phish'] * len(phish_i)}).to_csv(
        os.path.join(out, 'table_tfidf_top_tokens.csv'), index=False)


def fig_ablation(raw, out):
    p = raw[(raw['dataset'] == MPG) & (raw['condition'] == 'P')]
    if len(p) == 0:
        return
    per = p.groupby(['variant', 'seed']).apply(
        lambda s: (s['prob'] >= THRESH).mean(), include_groups=False).reset_index(name='rec')
    stat = per.groupby('variant').agg(mean=('rec', 'mean'), sd=('rec', 'std')).reset_index()
    stat['sd'] = stat['sd'].fillna(0)
    variants = order_variants(stat['variant'])
    stat = stat.set_index('variant').loc[variants].reset_index()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    xs = np.arange(len(stat))
    ax.bar(xs, stat['mean'], yerr=stat['sd'], capsize=4, color=BLUE)
    ax.set_ylabel('Recall on P (mean ± SD across seeds)'); ax.set_ylim(0, 1.02)
    ax.set_xticks(xs); ax.set_xticklabels(stat['variant'], rotation=30, ha='right')
    ax.set_title('Per-feature ablation: recall on matched-pair phishing')
    save(fig, out, 'fig_ablation.png')


def fig_forest(mpg, out):
    p = mpg[mpg['condition'] == 'P']
    if 'baseline' not in set(p['variant']):
        return
    base = p[p['variant'] == 'baseline'].set_index('sample_id')['pred']
    rows = []
    for v in order_variants(set(p['variant'])):
        if v == 'baseline':
            continue
        sv = p[p['variant'] == v].set_index('sample_id')['pred']
        common = base.index.intersection(sv.index)
        if len(common) == 0:
            continue
        d, lo, hi = boot_diff(sv.loc[common].values, base.loc[common].values)
        rows.append((v, d, lo, hi))
    if not rows:
        return
    labels = [r[0] for r in rows]; y = np.arange(len(rows))
    d = [r[1] for r in rows]; lo = [r[2] for r in rows]; hi = [r[3] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(rows) + 1.5))
    ax.errorbar(d, y, xerr=[np.array(d) - np.array(lo), np.array(hi) - np.array(d)],
                fmt='o', color=BLUE, capsize=4)
    ax.axvline(0, color=GREY, ls='--')
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel('Δ recall vs baseline (paired bootstrap 95% CI)')
    ax.set_title('Feature-variant effect on matched-pair recall')
    save(fig, out, 'fig_forest.png')


def fig_heatmap(mpg, out):
    if 'generator_model' not in mpg.columns:
        return
    base = mpg[mpg['variant'] == 'baseline']
    if len(base) == 0:
        base = mpg[mpg['variant'] == sorted(mpg['variant'].unique())[0]]
    conds = [c for c in ['P', 'B_hard', 'B_easy', 'P_R', 'B_hard_R'] if c in set(base['condition'])]
    gens = sorted(base['generator_model'].dropna().unique())
    M = np.full((len(gens), len(conds)), np.nan)
    for i, g in enumerate(gens):
        for j, c in enumerate(conds):
            sub = base[(base['generator_model'] == g) & (base['condition'] == c)]
            if len(sub):
                M[i, j] = sub['pred'].mean()
    fig, ax = plt.subplots(figsize=(1.3 * len(conds) + 2, 0.7 * len(gens) + 2))
    im = ax.imshow(M, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(len(conds))); ax.set_xticklabels(conds)
    ax.set_yticks(range(len(gens))); ax.set_yticklabels([g.split(':')[-1] for g in gens])
    for i in range(len(gens)):
        for j in range(len(conds)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=9)
    ax.set_title('Baseline: predicted-phishing rate by generator × condition')
    fig.colorbar(im, ax=ax, shrink=0.8, label='P(pred=phishing)')
    save(fig, out, 'fig_heatmap.png')


def fig_learning_curve(lc_path, out):
    if not lc_path or not os.path.exists(lc_path):
        return
    lc = pd.read_csv(lc_path)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(lc['n_train'], lc['recall'], 'o-', color=BLUE)
    ax.set_xlabel('TF-IDF training size'); ax.set_ylabel('Recall on first input')
    ax.set_title('TF-IDF learning curve (matched-volume comparison)')
    save(fig, out, 'fig_learning_curve.png')


def fig_score_hist(mpg, out):
    base = mpg[mpg['variant'] == 'baseline']
    if len(base) == 0:
        return
    P = base[base['condition'] == 'P']['prob'].values
    BH = base[base['condition'] == 'B_hard']['prob'].values
    if len(P) == 0 or len(BH) == 0:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bins = np.linspace(0, 1, 26)
    ax.hist(P, bins=bins, alpha=0.6, label='P (phishing)', color=RED, density=True)
    ax.hist(BH, bins=bins, alpha=0.6, label='B_hard (benign twin)', color=GREEN, density=True)
    ax.axvline(THRESH, color=GREY, ls='--')
    ax.set_xlabel('baseline P(phishing)'); ax.set_ylabel('density')
    ax.set_title('Score overlap of matched P vs B_hard (shortcut-learning diagnostic)')
    ax.legend()
    save(fig, out, 'fig_score_hist.png')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preds', default='predictions_long.csv')
    ap.add_argument('--tfidf', default='tfidf_predictions_long.csv')
    ap.add_argument('--features', default='mpg_emails_features.csv')
    ap.add_argument('--train', default='data/train.csv', help="for the TF-IDF coefficient figure")
    ap.add_argument('--learning_curve', default='tfidf_learning_curve.csv')
    ap.add_argument('--out', default='analysis_out')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    raw = load_preds(args.preds, args.tfidf)
    collapsed = collapse_seeds(raw)
    mpg = attach_pairing(collapsed[collapsed['dataset'] == MPG].copy(), args.features)
    # carry condition onto raw for the ablation/ per-seed figures
    if 'condition' not in raw.columns and 'sample_id' in raw.columns and os.path.exists(args.features):
        feat = pd.read_csv(args.features)[['sample_id', 'condition']].drop_duplicates()
        raw = raw.merge(feat, on='sample_id', how='left')

    print("Building figures:")
    fig_collapse(mpg, collapsed, args.out)
    fig_H1_intent_auc(mpg, args.out)
    fig_H2_register(mpg, args.out)
    fig_H3_features(args.features, args.out)
    fig_H3_by_generator(args.features, args.out)
    fig_H4_false_alarm(mpg, args.out)
    fig_tfidf_coef(args.train, args.out)
    fig_ablation(raw, args.out)
    fig_forest(mpg, args.out)
    fig_heatmap(mpg, args.out)
    fig_learning_curve(args.learning_curve, args.out)
    fig_score_hist(mpg, args.out)
    print(f"\nAll figures written to {args.out}/")


if __name__ == '__main__':
    main()
