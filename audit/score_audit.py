"""Score the returned human audit sheets (Review 2, Critical item 1).

Put each annotator's completed file in audit/returned/ (any name ending in .xlsx), then:
  python audit/score_audit.py

Reports, for the paper:
  - per-annotator agreement with the intended label (P = phishing, B_hard/B_easy = legitimate)
  - inter-annotator agreement on the verdict: Cohen's kappa (2 annotators) or Fleiss' kappa (3+),
    plus raw percentage agreement and Krippendorff-style note on "Cannot tell"
  - share of B_hard twins the majority judged legitimate, and the ambiguous ones (for the appendix)
  - H1 sensitivity: AUC(P vs B_hard) for each model on all 60 audited twins vs only the twins every
    annotator called legitimate
Writes audit/audit_results.csv (item level) and audit/audit_summary.txt.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'analysis'))
from extra_baselines import collapse, load  # noqa: E402

CATS = ['Phishing', 'Legitimate', 'Cannot tell']


def read_sheet(path):
    if path.endswith('.csv'):
        d = pd.read_csv(path)
        return pd.DataFrame({'item': d['item'], 'annotator': os.path.splitext(os.path.basename(path))[0],
                             'verdict': d['verdict'].where(d['verdict'].isin(CATS)),
                             'suspicion': pd.to_numeric(d['suspicion'], errors='coerce'),
                             'note': d['note'] if 'note' in d else None})
    ws = load_workbook(path, data_only=True)['Emails']
    name = str(ws['B1'].value or os.path.splitext(os.path.basename(path))[0]).strip()
    rows = []
    for r in ws.iter_rows(min_row=4, values_only=True):
        if not r[0]:
            continue
        v = str(r[2]).strip() if r[2] else None
        rows.append({'item': r[0], 'annotator': name, 'verdict': v if v in CATS else None,
                     'suspicion': pd.to_numeric(r[3], errors='coerce'), 'note': r[4]})
    return pd.DataFrame(rows)


def cohen_kappa(a, b):
    cats = sorted(set(a) | set(b))
    po = np.mean(np.asarray(a) == np.asarray(b))
    pe = sum(np.mean(np.asarray(a) == c) * np.mean(np.asarray(b) == c) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def fleiss_kappa(table):
    """table: items x categories counts, same number of raters per item."""
    n = table.sum(axis=1)[0]
    p_j = table.sum(axis=0) / table.sum()
    P_i = ((table ** 2).sum(axis=1) - n) / (n * (n - 1))
    pe = (p_j ** 2).sum()
    return (P_i.mean() - pe) / (1 - pe)


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'returned')
    files = sorted(glob.glob(os.path.join(folder, '*.xlsx')) + glob.glob(os.path.join(folder, '*.csv')))
    if len(files) < 1:
        raise SystemExit('Put the completed sheets in audit/returned/ first.')
    key = pd.read_csv(os.path.join(HERE, 'KEY_do_not_share.csv'))
    ann = pd.concat([read_sheet(f) for f in files], ignore_index=True)
    missing = ann[ann['verdict'].isna()]
    if len(missing):
        print(f'WARNING: {len(missing)} blank/invalid verdicts:', missing.groupby('annotator').size().to_dict())
    ann = ann.merge(key, on='item')
    ann['intended'] = np.where(ann['condition'] == 'P', 'Phishing', 'Legitimate')
    names = sorted(ann['annotator'].unique())
    out = [f'Annotators: {len(names)} ({", ".join(names)}); items: {key.shape[0]}', '']

    out.append('Agreement with the intended label (excluding "Cannot tell"):')
    for nm, g in ann.groupby('annotator'):
        dec = g[g['verdict'].isin(['Phishing', 'Legitimate'])]
        by = dec.groupby('condition').apply(lambda s: (s['verdict'] == s['intended']).mean(), include_groups=False)
        out.append(f'  {nm}: overall {np.mean(dec.verdict == dec.intended):.3f} | ' +
                   ', '.join(f'{c} {v:.3f}' for c, v in by.items()) +
                   f' | cannot tell {np.mean(g.verdict == "Cannot tell"):.3f}')

    wide = ann.pivot_table(index='item', columns='annotator', values='verdict', aggfunc='first')
    wide = wide.dropna()
    if len(names) == 2:
        k = cohen_kappa(wide[names[0]], wide[names[1]])
        out.append(f"\nCohen's kappa (verdict, 3 categories, n={len(wide)}): {k:.3f}; "
                   f'raw agreement {np.mean(wide[names[0]] == wide[names[1]]):.3f}')
    elif len(names) >= 3:
        table = np.stack([(wide == c).sum(axis=1).values for c in CATS], axis=1)
        out.append(f"\nFleiss' kappa (verdict, {len(names)} raters, n={len(wide)}): {fleiss_kappa(table):.3f}")

    # item level: majority verdict and unanimity
    item = ann.groupby('item').agg(condition=('condition', 'first'), sample_id=('sample_id', 'first'),
                                   generator=('generator_model', 'first'), scenario=('scenario_id', 'first'),
                                   n_legit=('verdict', lambda s: (s == 'Legitimate').sum()),
                                   n_phish=('verdict', lambda s: (s == 'Phishing').sum()),
                                   n_cannot=('verdict', lambda s: (s == 'Cannot tell').sum()),
                                   mean_suspicion=('suspicion', 'mean'),
                                   notes=('note', lambda s: ' | '.join(str(x) for x in s if x))).reset_index()
    item['majority'] = np.select([item.n_legit > item.n_phish + item.n_cannot,
                                  item.n_phish > item.n_legit + item.n_cannot], ['Legitimate', 'Phishing'],
                                 'No majority')
    item['unanimous_legit'] = item.n_legit == len(names)
    item.to_csv(os.path.join(HERE, 'audit_results.csv'), index=False)

    bh = item[item.condition == 'B_hard']
    out.append(f'\nB_hard twins (n={len(bh)}): majority legitimate {np.mean(bh.majority == "Legitimate"):.3f}, '
               f'unanimously legitimate {bh.unanimous_legit.mean():.3f}, '
               f'majority phishing {np.mean(bh.majority == "Phishing"):.3f}')
    out.append(f'Mean suspicion (1-5): P {item[item.condition == "P"].mean_suspicion.mean():.2f}, '
               f'B_hard {bh.mean_suspicion.mean():.2f}, B_easy {item[item.condition == "B_easy"].mean_suspicion.mean():.2f}')
    amb = bh[bh.majority != 'Legitimate']
    out.append(f'Ambiguous or disputed twins ({len(amb)}): ' + ', '.join(amb['item']))

    # H1 sensitivity on the audited twins
    col = collapse(load())
    mpg = col[col['dataset'] == 'mpg_emails']
    keep_all = set(bh.sample_id)
    keep_ok = set(bh[bh.unanimous_legit].sample_id)
    out.append('\nH1 sensitivity: AUC(all 450 P vs audited B_hard twins)')
    for v in ['baseline', 'distilbert', 'roberta', 'tfidf']:
        s = mpg[mpg['variant'] == v]
        P = s[s['condition'] == 'P']['prob'].values
        res = []
        for ids in (keep_all, keep_ok):
            B = s[s['sample_id'].isin(ids)]['prob'].values
            res.append(roc_auc_score(np.r_[np.ones(len(P)), np.zeros(len(B))], np.r_[P, B]) if len(B) else np.nan)
        out.append(f'  {v:<11} all 60 audited: {res[0]:.3f} | unanimously legitimate only (n={len(keep_ok)}): {res[1]:.3f}')

    text = '\n'.join(out)
    print(text)
    open(os.path.join(HERE, 'audit_summary.txt'), 'w').write(text + '\n')


if __name__ == '__main__':
    main()
