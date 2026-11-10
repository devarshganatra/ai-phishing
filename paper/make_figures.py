"""Publication figures for the manuscript, in one consistent style.

Every figure is drawn at the text width of the single-column template (6.3 in), in Times New
Roman at 9 pt, with one shared palette, and saved as SVG so the PDF keeps vector quality.
Numbers are recomputed from the per-email prediction dumps, not typed in.

  python paper/make_figures.py      (run from the repository root)
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'analysis'))
from paper_stats import load_preds, collapse_seeds, THRESH  # noqa: E402

OUT = os.path.join(ROOT, 'paper', 'figures')
TEXT_W = 6.3

# One palette for the whole paper. Conditions keep the same colour in every figure.
INK, MUTED, GRID = '#1f1f1f', '#8a9099', '#e3e3e3'
BLUE, SLATE = '#2c5f8d', '#4a4a4a'                    # DistilBERT, TF-IDF
P_COL, BH_COL, BE_COL = '#b5523b', '#c9952f', '#4b8f7f'  # phishing, legitimate twin, plain legitimate
TINT = {'blue': ('#e3ecf5', BLUE), 'brick': ('#f5e4df', P_COL), 'ochre': ('#f7eedb', BH_COL),
        'teal': ('#e0eeea', BE_COL), 'grey': ('#efefef', '#5b5b5b')}

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'STIX Two Text', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9,
    'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 8.5,
    'axes.linewidth': 0.6, 'axes.edgecolor': '#444444', 'axes.labelcolor': INK,
    'xtick.color': '#444444', 'ytick.color': '#444444', 'xtick.major.width': 0.6,
    'ytick.major.width': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False, 'svg.fonttype': 'none', 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.03,
    'pdf.fonttype': 42,
})

VARIANTS = ['baseline', 'perplexity', 'grammar', 'length', 'behavioral', 'psychological', 'full', 'tfidf']
LABEL = {'baseline': 'DistilBERT baseline', 'perplexity': '+ Perplexity', 'grammar': '+ Grammar',
         'length': '+ Length', 'behavioral': '+ Behavioural', 'psychological': '+ Psychological',
         'full': '+ Full (6)', 'tfidf': 'TF-IDF + LR'}
SCENARIOS = ['IT password reset', 'Wire-transfer request (BEC)', 'Payment-service suspension',
             'HR policy document', 'Software renewal invoice', 'Courier delivery failure',
             'Prize notification', 'Bank login alert', 'Mailbox-full warning', 'Cloud-drive report share']


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + '.svg'))
    fig.savefig(os.path.join(OUT, name + '.pdf'))
    fig.savefig(os.path.join(OUT, name + '.png'), dpi=220)
    plt.close(fig)
    print('  wrote', name)


def ygrid(ax):
    ax.grid(axis='x', color=GRID, lw=0.6)
    ax.set_axisbelow(True)


# ----------------------------------------------------------------------------- data
def load():
    raw = load_preds(os.path.join(ROOT, 'predictions_long.csv'),
                     os.path.join(ROOT, 'tfidf_predictions_long.csv'))
    col = collapse_seeds(raw)
    return col[col['dataset'] == 'test'], col[col['dataset'] == 'mpg_emails']


def auc(s, a, b):
    x = s[s['condition'].isin([a, b])]
    return roc_auc_score((x['condition'] == a).astype(int), x['prob'])


def cluster_ci(s, a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = s[s['condition'].isin([a, b])].reset_index(drop=True)
    cells = list(x.groupby(['generator_model', 'scenario_id']).groups.values())
    vals = []
    for _ in range(n):
        idx = np.concatenate([cells[i] for i in rng.integers(0, len(cells), len(cells))])
        t = x.loc[idx]
        vals.append(roc_auc_score((t['condition'] == a).astype(int), t['prob']))
    return np.percentile(vals, [2.5, 97.5])


# ----------------------------------------------------------------------------- diagrams
def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis('off')
    return fig, ax


def box(ax, cx, cy, w, h, title, lines=(), tint='grey', dashed=False, bold=True):
    fc, ec = TINT[tint]
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle='round,pad=0,rounding_size=0.05',
                                fc=fc, ec=ec, lw=0.8, ls=(0, (3, 2)) if dashed else '-'))
    n = len(lines)
    top = cy + (n * 0.13) / 2
    ax.text(cx, top, title, ha='center', va='center', fontsize=9, color=INK,
            fontweight='bold' if bold else 'normal')
    for i, line in enumerate(lines):
        ax.text(cx, top - 0.15 * (i + 1), line, ha='center', va='center', fontsize=8, color='#3a3a3a')


def group(ax, x0, y0, w, h, label):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle='round,pad=0,rounding_size=0.06',
                                fc='none', ec='#9a9a9a', lw=0.7, ls=(0, (4, 3))))
    ax.text(x0 + 0.08, y0 + h - 0.11, label, ha='left', va='center', fontsize=8.5,
            style='italic', color='#555555')


def arrow(ax, pts):
    pts = list(pts)
    for (x1, y1), (x2, y2) in zip(pts[:-2], pts[1:-1]):
        ax.plot([x1, x2], [y1, y2], color='#555555', lw=0.8, solid_capstyle='butt')
    ax.annotate('', xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle='-|>,head_length=0.35,head_width=0.18', color='#555555', lw=0.8,
                                shrinkA=0, shrinkB=0))


def fig_workflow():
    W, H = 6.3, 2.75
    fig, ax = canvas(W, H - 0.22)
    ax.set_ylim(0.22, H)
    bw, bh = 1.32, 0.62
    xs = [0.86, 2.36, 3.86]
    ya, yb = 2.05, 0.78
    group(ax, 0.1, 1.6, 4.52, 0.98, 'Training: human-written corpus')
    group(ax, 0.1, 0.33, 4.52, 0.98, 'Evaluation only: LLM-written matched pairs')
    box(ax, xs[0], ya, bw, bh, 'Public corpus', ['17,537 emails', 'Kaggle / Enron'], 'grey')
    box(ax, xs[1], ya, bw, bh, 'Split and features', ['80/10/10, seed 42', 'six features'], 'grey')
    box(ax, xs[2], ya, bw, bh, 'Train detectors', ['7 variants x 5 seeds', '+ TF-IDF baseline'], 'blue')
    box(ax, xs[0], yb, bw, bh, 'LLM generation', ['3 models, 10 lures', 'temperature 0.8'], 'grey')
    box(ax, xs[1], yb, bw, bh, 'Quality checks', ['refusals, length', 'every attempt logged'], 'grey')
    box(ax, xs[2], yb, bw, bh, 'Matched-pair set', [r'P, $B_{\mathrm{hard}}$, $B_{\mathrm{easy}}$', '+ 96 rewrites'], 'brick')
    box(ax, 5.5, 1.42, 1.3, 0.9, 'Score and test', ['1,754 + 1,446 emails', 'per-email probabilities', 'AUC, FPR, McNemar,', 'bootstrap, Holm'], 'grey')
    for y in (ya, yb):
        arrow(ax, [(xs[0] + bw / 2, y), (xs[1] - bw / 2, y)])
        arrow(ax, [(xs[1] + bw / 2, y), (xs[2] - bw / 2, y)])
    arrow(ax, [(xs[2] + bw / 2, ya), (4.72, ya), (4.72, 1.6), (4.85, 1.6)])
    arrow(ax, [(xs[2] + bw / 2, yb), (4.72, yb), (4.72, 1.24), (4.85, 1.24)])
    save(fig, 'fig1_workflow')


def fig_design():
    W, H = 6.3, 3.05
    fig, ax = canvas(W, H)
    bw, bh = 1.78, 0.62
    xs = [1.05, 3.15, 5.25]
    box(ax, 3.15, 2.66, 3.6, 0.46, 'Scenario: IT password reset', ['same sender role and topic in every condition'], 'grey')
    y2, y3 = 1.68, 0.55
    box(ax, xs[0], y2, bw, bh, 'P: phishing (label 1)', ['reset within 24 hours', 'at an outside link'], 'brick')
    box(ax, xs[1], y2, bw, bh, r'$B_{\mathrm{hard}}$: legitimate twin', ['password expires in 3 days;', 'use the intranet portal'], 'ochre')
    box(ax, xs[2], y2, bw, bh, r'$B_{\mathrm{easy}}$: everyday mail', ['coworker suggests coffee;', 'no deadline, link or authority'], 'teal')
    box(ax, xs[0], y3, bw, bh, r'$P_{\mathrm{R}}$: register rewrite', ['same request, terse or', 'courteous wording'], 'brick', dashed=True)
    box(ax, xs[1], y3, bw, bh, r'$B_{\mathrm{hard,R}}$: rewrite', ['same benign request, terse', 'or courteous wording'], 'ochre', dashed=True)
    for x in xs:
        arrow(ax, [(3.15, 2.66 - 0.23), (3.15, 2.27), (x, 2.27), (x, y2 + bh / 2)])
    for x in xs[:2]:
        arrow(ax, [(x, y2 - bh / 2), (x, y3 + bh / 2)])
        ax.text(x + 0.07, (y2 + y3) / 2, 'rewrite', fontsize=8, style='italic', color='#555555', va='center')
    # comparisons key
    ax.add_patch(FancyBboxPatch((xs[2] - bw / 2, y3 - 0.4), bw, 0.8, boxstyle='round,pad=0,rounding_size=0.05',
                                fc='white', ec='#9a9a9a', lw=0.7))
    ax.text(xs[2], y3 + 0.26, 'Comparisons', ha='center', va='center', fontsize=9, fontweight='bold', color=INK)
    for i, t in enumerate([r'P vs $B_{\mathrm{hard}}$: intent (H1)', r'P vs $P_{\mathrm{R}}$: register (H2)', r'P vs $B_{\mathrm{easy}}$: cues present']):
        ax.text(xs[2], y3 + 0.06 - 0.16 * i, t, ha='center', va='center', fontsize=8, color='#3a3a3a')
    save(fig, 'fig2_design')


def fig_architecture():
    W, H = 6.3, 2.9
    fig, ax = canvas(W, H)
    bw, bh = 1.25, 0.5
    xs = [0.8, 2.25, 3.7]
    ya, yb = 2.25, 1.18
    group(ax, 0.08, 1.88, 4.37, 0.88, 'Text encoder')
    group(ax, 0.08, 0.81, 4.37, 0.88, 'Handcrafted features')
    box(ax, xs[0], ya, bw, bh, 'Email text', ['up to 512 tokens'], 'blue')
    box(ax, xs[1], ya, bw, bh, 'DistilBERT', ['6 layers, uncased'], 'blue')
    box(ax, xs[2], ya, bw, bh, '[CLS] vector h', ['768 dimensions'], 'blue')
    box(ax, xs[0], yb, bw, bh, 'Features s', ['0, 1, 3 or 6 values'], 'teal')
    box(ax, xs[1], yb, bw, bh, 'Standardise', ['training mean and SD'], 'teal')
    box(ax, xs[2], yb, bw, bh, 'Scaled features', ['k inputs per variant'], 'teal')
    xc = 5.4
    box(ax, xc, 1.72, 1.4, bh, 'Concatenate z', ['768 + k inputs'], 'grey')
    box(ax, xc, 0.98, 1.4, bh, 'Classifier head', ['256 ReLU, dropout 0.2'], 'grey')
    box(ax, xc, 0.26, 1.4, 0.42, 'P(phishing)', [], 'grey')
    for y in (ya, yb):
        arrow(ax, [(xs[0] + bw / 2, y), (xs[1] - bw / 2, y)])
        arrow(ax, [(xs[1] + bw / 2, y), (xs[2] - bw / 2, y)])
    arrow(ax, [(xs[2] + bw / 2, ya), (5.15, ya), (5.15, 1.72 + bh / 2)])
    arrow(ax, [(xs[2] + bw / 2, yb), (4.62, yb), (4.62, 1.72), (xc - 0.7, 1.72)])
    arrow(ax, [(xc, 1.72 - bh / 2), (xc, 0.98 + bh / 2)])
    arrow(ax, [(xc, 0.98 - bh / 2), (xc, 0.26 + 0.21)])
    ax.text(0.1, 0.48, 'The baseline variant uses k = 0 and passes h alone to the head.',
            fontsize=8, style='italic', color='#555555', va='center')
    save(fig, 'fig3_architecture')


# ----------------------------------------------------------------------------- results
def fig_collapse(test, mpg):
    rows = []
    for v in VARIANTS:
        t = test[(test['variant'] == v) & (test['label'] == 1)]
        p = mpg[(mpg['variant'] == v) & (mpg['condition'] == 'P')]
        rows.append((v, (t['prob'] >= THRESH).mean(), (p['prob'] >= THRESH).mean()))
    fig, ax = plt.subplots(figsize=(TEXT_W, 2.6))
    ys = np.arange(len(rows))[::-1]
    for y, (v, conv, mp) in zip(ys, rows):
        c = SLATE if v == 'tfidf' else BLUE
        ax.plot([mp, conv], [y, y], color=GRID, lw=2.2, solid_capstyle='round', zorder=1)
        ax.scatter([conv], [y], s=34, facecolor='white', edgecolor=c, lw=1.1, zorder=3)
        ax.scatter([mp], [y], s=34, color=c, zorder=3)
        ax.text(mp - 0.012, y, f'{mp:.2f}', ha='right', va='center', fontsize=8, color='#3a3a3a')
    ax.set_yticks(ys)
    ax.set_yticklabels([LABEL[v] for v, _, _ in rows])
    ax.set_xlim(0.3, 1.02)
    ax.set_xlabel('Recall on phishing')
    ygrid(ax)
    ax.scatter([], [], s=34, facecolor='white', edgecolor=BLUE, lw=1.1, label='Conventional test split')
    ax.scatter([], [], s=34, color=BLUE, label='Matched-pair phishing (P)')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2)
    save(fig, 'fig4_collapse')


def fig_intent_auc(mpg):
    rows = []
    for v in VARIANTS:
        s = mpg[mpg['variant'] == v]
        lo, hi = cluster_ci(s, 'P', 'B_hard')
        rows.append((v, auc(s, 'P', 'B_hard'), lo, hi, auc(s, 'P', 'B_easy')))
    fig, ax = plt.subplots(figsize=(TEXT_W, 2.6))
    ys = np.arange(len(rows))[::-1]
    ax.axvline(0.5, color=MUTED, ls=(0, (3, 2)), lw=0.8)
    ax.text(0.505, ys[-1] - 0.45, 'chance', fontsize=8, color=MUTED, va='center')
    for y, (v, a, lo, hi, e) in zip(ys, rows):
        ax.plot([lo, hi], [y, y], color=P_COL, lw=1.2)
        ax.scatter([a], [y], s=32, color=P_COL, zorder=3)
        ax.scatter([e], [y], s=32, facecolor='white', edgecolor=BE_COL, lw=1.1, zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([LABEL[r[0]] for r in rows])
    ax.set_xlim(0.45, 0.92)
    ax.set_xlabel('AUC')
    ygrid(ax)
    ax.scatter([], [], s=32, color=P_COL, label=r'P vs legitimate twin ($B_{\mathrm{hard}}$), 95% cluster CI')
    ax.scatter([], [], s=32, facecolor='white', edgecolor=BE_COL, lw=1.1, label=r'P vs plain legitimate ($B_{\mathrm{easy}}$)')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2)
    save(fig, 'fig5_intent_auc')


def fig_features():
    f = pd.read_csv(os.path.join(ROOT, 'mpg_emails_features.csv'))
    feats = [('perplexity', 'Perplexity'), ('grammar_errors', 'Grammar errors per 100 words'),
             ('word_count', 'Word count'), ('urgency_score', 'Urgency count'),
             ('fear_score', 'Fear count'), ('authority_score', 'Authority count')]
    conds = [('P', P_COL), ('B_hard', BH_COL), ('B_easy', BE_COL)]
    fig, axes = plt.subplots(2, 3, figsize=(TEXT_W, 3.7))
    for ax, (col, title) in zip(axes.ravel(), feats):
        data = [f[f['condition'] == c][col].dropna().values for c, _ in conds]
        hi = np.percentile(np.concatenate(data), 99)
        parts = ax.violinplot([np.clip(d, None, hi) for d in data], showextrema=False, widths=0.8)
        for body, (_, colr) in zip(parts['bodies'], conds):
            body.set_facecolor(colr)
            body.set_edgecolor(colr)
            body.set_alpha(0.35)
            body.set_linewidth(0.6)
        for i, d in enumerate(data, start=1):
            ax.plot([i - 0.22, i + 0.22], [np.median(d)] * 2, color=INK, lw=1.1)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels([{'P': 'P', 'B_hard': r'$B_{\mathrm{hard}}$', 'B_easy': r'$B_{\mathrm{easy}}$'}[c] for c, _ in conds])
        ax.set_title(title, pad=4)
        ax.grid(axis='y', color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    fig.tight_layout(h_pad=1.2, w_pad=1.0)
    save(fig, 'fig6_features')


def fig_forest():
    t = pd.read_csv(os.path.join(ROOT, 'analysis_out', 'table_ablation_vs_baseline.csv'))
    p = t['mcnemar_p'].values
    order = np.argsort(p)
    holm, run = np.empty(len(p)), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - r) * p[i]))
        holm[i] = run
    t['holm'] = holm
    t = t.set_index('variant').loc[[v for v in VARIANTS if v != 'baseline']].reset_index()
    fig, ax = plt.subplots(figsize=(TEXT_W, 2.4))
    ys = np.arange(len(t))[::-1]
    ax.axvline(0, color=MUTED, lw=0.8)
    for y, r in zip(ys, t.itertuples()):
        c = SLATE if r.variant == 'tfidf' else BLUE
        sig = r.holm < 0.05
        ax.plot([r.ci_lo, r.ci_hi], [y, y], color=c, lw=1.2)
        ax.scatter([r.delta_recall], [y], s=32, color=c if sig else 'white', edgecolor=c, lw=1.1, zorder=3)
        ax.text(r.ci_hi + 0.006, y, f'{r.delta_recall:+.3f}', va='center', fontsize=8, color='#3a3a3a')
    ax.set_yticks(ys)
    ax.set_yticklabels([LABEL[v] for v in t['variant']])
    ax.set_xlabel('Difference in recall on P against the DistilBERT baseline')
    ax.set_xlim(-0.05, 0.31)
    ygrid(ax)
    ax.scatter([], [], s=32, color=BLUE, label='Holm-adjusted p < 0.05')
    ax.scatter([], [], s=32, facecolor='white', edgecolor=BLUE, lw=1.1, label='not significant')
    ax.legend(loc='upper right')
    save(fig, 'fig7_forest')


def fig_tfidf_terms():
    t = pd.read_csv(os.path.join(ROOT, 'analysis_out', 'table_tfidf_top_tokens.csv'))
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 2.75))
    for ax, side, colr, title in [(axes[0], 'safe', BE_COL, 'Terms that push toward legitimate'),
                                   (axes[1], 'phish', P_COL, 'Terms that push toward phishing')]:
        d = t[t['side'] == side].assign(mag=lambda x: x['coef'].abs()).sort_values('mag')
        ys = np.arange(len(d))
        ax.barh(ys, d['mag'], color=colr, height=0.68)
        ax.set_yticks(ys)
        ax.set_yticklabels(d['token'])
        ax.set_title(title, pad=4)
        ax.set_xlabel('|coefficient|')
        ax.set_xlim(0, 15)
        ygrid(ax)
    fig.tight_layout(w_pad=2.0)
    save(fig, 'fig8_tfidf_terms')


def fig_scenarios(mpg):
    p = mpg[mpg['condition'] == 'P']
    base = p[p['variant'] == 'baseline'].groupby('scenario_id')['pred'].mean()
    tf = p[p['variant'] == 'tfidf'].groupby('scenario_id')['pred'].mean()
    order = base.sort_values(ascending=True).index
    fig, ax = plt.subplots(figsize=(TEXT_W, 2.9))
    ys = np.arange(len(order))
    ax.axhspan(-0.5, 1.5, color='#f2f2f2', zorder=0)
    for y, sid in zip(ys, order):
        b, t = base[sid] * 100, tf[sid] * 100
        ax.plot([b, t], [y, y], color=GRID, lw=2.2, zorder=1, solid_capstyle='round')
        ax.scatter([t], [y], s=40, facecolor='white', edgecolor=SLATE, lw=1.1, zorder=2)
        ax.scatter([b], [y], s=26, color=BLUE, zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([SCENARIOS[int(s)] for s in order])
    ax.set_xlim(-2, 104)
    ax.set_xlabel('Recall on the 45 phishing emails of each scenario (%)')
    ygrid(ax)
    ax.text(102, 0.5, 'both models below 15%', ha='right', va='center', fontsize=8, style='italic', color='#555555')
    ax.scatter([], [], s=26, color=BLUE, label='DistilBERT baseline')
    ax.scatter([], [], s=40, facecolor='white', edgecolor=SLATE, lw=1.1, label='TF-IDF + LR')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2)
    save(fig, 'fig9_scenarios')


def equation():
    eqs = {'eq1': r'$\tilde{s}_j = \dfrac{s_j - \mu_j}{\sigma_j}\,,\qquad z = \left[\, h \,;\, \tilde{s} \,\right] \in \mathbb{R}^{768+k}$',
           'eq2': r'$\hat{y} = \mathrm{sigmoid}\left(w_2^{\top}\,\mathrm{Dropout}\left(\mathrm{ReLU}(W_1 z + b_1)\right) + b_2\right)$'}
    for name, tex in eqs.items():
        fig = plt.figure(figsize=(TEXT_W, 0.42 if name == 'eq1' else 0.32))
        fig.text(0.5, 0.5, tex, ha='center', va='center', fontsize=11)
        save(fig, name)


if __name__ == '__main__':
    test, mpg = load()
    fig_workflow()
    fig_design()
    fig_architecture()
    fig_collapse(test, mpg)
    fig_intent_auc(mpg)
    fig_features()
    fig_forest()
    fig_tfidf_terms()
    fig_scenarios(mpg)
    equation()
