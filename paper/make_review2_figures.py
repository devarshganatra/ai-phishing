"""Figure for the review-2 experiments (stronger baseline + E-PhishLLM), in the paper's style.

Reads the tables written by analysis/extra_baselines.py.

  python analysis/extra_baselines.py
  python paper/make_review2_figures.py   -> paper/figures/fig10_baselines.{svg,pdf,png}
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from make_figures import BE_COL, BLUE, GRID, MUTED, P_COL, ROOT, SLATE, TEXT_W, save

T = os.path.join(ROOT, 'analysis_out', 'review2')
PURPLE = '#6b4c9a'   # RoBERTa
NAME = {'baseline': 'DistilBERT (paper run)', 'distilbert': 'DistilBERT (re-run)',
        'roberta': 'RoBERTa-base', 'tfidf': 'TF-IDF + LR'}


def main():
    mp = pd.read_csv(os.path.join(T, 'table_matched_pairs.csv')).set_index('model')
    ep = pd.read_csv(os.path.join(T, 'table_ephishllm.csv')).set_index('model')

    fig, (a, b) = plt.subplots(1, 2, figsize=(TEXT_W, 2.5), gridspec_kw={'width_ratios': [1.05, 1]})

    order = ['baseline', 'distilbert', 'roberta', 'tfidf']
    ys = np.arange(len(order))[::-1]
    a.axvline(0.5, color=MUTED, ls=(0, (3, 2)), lw=0.8)
    a.axvline(0.75, color=MUTED, ls=':', lw=0.8)
    a.text(0.755, ys[-1] - 0.42, 'H1 bound', fontsize=7.5, color=MUTED, va='center')
    for y, m in zip(ys, order):
        r = mp.loc[m]
        a.plot([r.AUC_P_vs_Bhard_lo, r.AUC_P_vs_Bhard_hi], [y, y], color=P_COL, lw=1.2)
        a.scatter([r.AUC_P_vs_Bhard], [y], s=30, color=P_COL, zorder=3)
        a.scatter([r.AUC_P_vs_Beasy], [y], s=30, facecolor='white', edgecolor=BE_COL, lw=1.1, zorder=3)
    a.set_yticks(ys)
    a.set_yticklabels([NAME[m] for m in order])
    a.set_xlim(0.45, 0.92)
    a.set_ylim(ys[-1] - 0.65, ys[0] + 0.5)
    a.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
    a.set_xlabel('AUC on matched pairs')
    a.set_title('(a) Phishing vs legitimate twin', pad=4)
    a.grid(axis='x', color=GRID, lw=0.6)
    a.set_axisbelow(True)
    a.scatter([], [], s=30, color=P_COL, label='P vs $B_{\\mathrm{hard}}$ (95% cluster CI)')
    a.scatter([], [], s=30, facecolor='white', edgecolor=BE_COL, lw=1.1, label='P vs $B_{\\mathrm{easy}}$')
    a.legend(loc='lower center', bbox_to_anchor=(0.5, 1.09), ncol=2, handletextpad=0.3, columnspacing=1.0)

    order_e = ['distilbert', 'roberta', 'tfidf']
    colr = {'distilbert': BLUE, 'roberta': PURPLE, 'tfidf': SLATE}
    ys = np.arange(len(order_e))[::-1]
    for y, m in zip(ys, order_e):
        r = ep.loc[m]
        b.plot([r.f1_seed_min, r.f1_seed_max], [y, y], color=colr[m], lw=4, alpha=0.25, solid_capstyle='butt')
        b.plot([r.f1_lo, r.f1_hi], [y, y], color=colr[m], lw=1.2)
        b.scatter([r.f1], [y], s=30, color=colr[m], zorder=3)
        b.scatter([r.auc], [y], s=30, marker='D', facecolor='white', edgecolor=colr[m], lw=1.1, zorder=3)
    b.axvspan(0.55, 0.71, color='#eeeeee', zorder=0)
    b.text(0.63, ys[-1] - 0.62, 'F1 range reported\nby E-PhishLLM authors', fontsize=7.2, color=MUTED,
           ha='center', va='center')
    b.set_yticks(ys)
    b.set_yticklabels([NAME[m] for m in order_e])
    b.set_ylim(ys[-1] - 0.95, ys[0] + 0.5)
    b.set_xlim(0.45, 0.92)
    b.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
    b.set_xlabel('Score on E-PhishLLM (English, n = 11,502)')
    b.set_title('(b) External benchmark', pad=4)
    b.grid(axis='x', color=GRID, lw=0.6)
    b.set_axisbelow(True)
    b.scatter([], [], s=30, color=MUTED, label='F1 (95% CI)')
    b.scatter([], [], s=30, marker='D', facecolor='white', edgecolor=MUTED, lw=1.1, label='ROC AUC')
    b.plot([], [], color=MUTED, lw=4, alpha=0.3, label='5-seed range')
    b.legend(loc='lower center', bbox_to_anchor=(0.5, 1.09), ncol=2, handletextpad=0.3, columnspacing=1.0)

    fig.tight_layout(w_pad=1.6)
    save(fig, 'fig10_baselines')


if __name__ == '__main__':
    main()
