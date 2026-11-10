"""Review 2 (High): does the TF-IDF result depend on corpus-provenance tokens, or on phishing cues?

The TF-IDF + LR baseline is retrained under controlled lexical masking. The same masking is
applied to training and evaluation text, and the model configuration is unchanged
(tfidf_baseline.py). Conditions:

  M0 none          original text
  M1 provenance    remove corpus-identity artefacts: years and other digits, URLs, e-mail
                   addresses, reply/forward header words, and Enron-specific names/terms
  M2 top-weight    M1, plus remove the 200 tokens with the largest |coefficient| in the M1 model
                   (data-driven: whatever the model leaned on most, provenance or not)
  M3 cue words     M1, plus remove the urgency, fear and authority lexicons of extract_features.py
                   (tests whether the cue words themselves drive false alarms on B_hard)

For each condition: F1 on the Kaggle/Enron test split, matched-pair recall/FPR/AUC, and E-PhishLLM.

  python analysis/tfidf_provenance.py   -> analysis_out/review2/table_tfidf_masking.csv, _top_tokens.csv
"""
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tfidf_baseline import build_pipeline  # noqa: E402

OUT = os.path.join(ROOT, 'analysis_out', 'review2')

PROVENANCE_TERMS = ['enron', 'ect', 'hou', 'vince', 'kaminski', 'louise', 'kitchen', 'lavorato', 'shirley',
                    'crenshaw', 'stinson', 'gibner', 'jeff', 'skilling', 'dasovich', 'kean', 'steffes',
                    'wrote', 'forwarded', 'original message', 'cc', 'bcc', 'subject', 're', 'fw', 'fwd',
                    'pm', 'am', 'edu', 'com', 'www', 'http', 'https', 'net', 'org']
URGENCY = [r'immediately', r'urgent', r'within \d+ hours', r'action required', r'asap', r'important',
           r'soon', r'final notice']
FEAR = [r'suspended', r'terminated', r'penalty', r'legal action', r'blocked', r'deactivated',
        r'compromised', r'unauthorized', r'failure', r'warning']
AUTHORITY = [r'ceo', r'cfo', r'hr', r'it department', r'admin', r'support', r'manager', r'director',
             r'president', r'helpdesk']

URL = re.compile(r'(https?://\S+|www\.\S+|\b\S+\.(com|net|org|edu|gov|io|co)(/\S*)?\b)', re.I)
EMAIL = re.compile(r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b')
DIGITS = re.compile(r'\d+')


def words_re(terms):
    return re.compile(r'\b(' + '|'.join(sorted(terms, key=len, reverse=True)) + r')\b', re.I)


PROV_RE = words_re([re.escape(t) for t in PROVENANCE_TERMS])
CUE_RE = words_re(URGENCY + FEAR + AUTHORITY)


def m1(text):
    t = EMAIL.sub(' ', str(text))
    t = URL.sub(' ', t)
    t = DIGITS.sub(' ', t)
    return PROV_RE.sub(' ', t)


def m3(text):
    return CUE_RE.sub(' ', m1(text))


def drop_tokens(tokens):
    rx = words_re([re.escape(t) for t in tokens])
    return lambda text: rx.sub(' ', m1(text))


def evaluate(pipe, f, test, mpg, eph):
    r = {}
    pt = pipe.predict_proba(test['text'].map(f))[:, 1]
    r['test_f1'] = f1_score(test['label'], pt >= 0.5)
    pm = pipe.predict_proba(mpg['text'].map(f))[:, 1]
    m = mpg.assign(prob=pm, pred=(pm >= 0.5).astype(int))
    c = {k: m[m.condition == k] for k in ['P', 'B_hard', 'B_easy']}
    r['recall_P'] = c['P'].pred.mean()
    r['FPR_B_hard'] = c['B_hard'].pred.mean()
    r['FPR_B_easy'] = c['B_easy'].pred.mean()
    for b in ['B_hard', 'B_easy']:
        x = pd.concat([c['P'], c[b]])
        r[f'AUC_P_vs_{b}'] = roc_auc_score((x.condition == 'P').astype(int), x.prob)
    pe = pipe.predict_proba(eph['text'].map(f))[:, 1]
    r['ephish_f1'] = f1_score(eph['label'], pe >= 0.5)
    r['ephish_auc'] = roc_auc_score(eph['label'], pe)
    return r


def top_tokens(pipe, k=15):
    names = np.array(pipe.named_steps['tfidf'].get_feature_names_out())
    coef = pipe.named_steps['clf'].coef_[0]
    o = np.argsort(coef)
    return list(names[o[:k]]), list(names[o[-k:][::-1]]), names, coef


def main():
    os.makedirs(OUT, exist_ok=True)
    train = pd.read_csv(os.path.join(ROOT, 'data', 'train.csv')).dropna(subset=['text', 'label'])
    test = pd.read_csv(os.path.join(ROOT, 'data', 'test.csv')).dropna(subset=['text', 'label'])
    mpg = pd.read_csv(os.path.join(ROOT, 'mpg_emails.csv'))
    eph = pd.read_csv(os.path.join(ROOT, 'results_kaggle', 'ephishllm_en.csv'))
    y = train['label'].astype(int)

    rows, tops = [], []

    def run(name, f):
        pipe = build_pipeline().fit(train['text'].map(f), y)
        r = {'condition': name, 'vocab': len(pipe.named_steps['tfidf'].vocabulary_), **evaluate(pipe, f, test, mpg, eph)}
        rows.append(r)
        safe, phish, names, coef = top_tokens(pipe)
        tops.append({'condition': name, 'top_legitimate': ', '.join(safe), 'top_phishing': ', '.join(phish)})
        print(f"{name:<14} " + ' '.join(f'{k}={v:.3f}' for k, v in r.items() if isinstance(v, float)))
        return names, coef

    run('M0 none', lambda t: str(t))
    names, coef = run('M1 provenance', m1)
    top200 = names[np.argsort(-np.abs(coef))[:200]]
    run('M2 top-weight', drop_tokens([t for t in top200 if ' ' not in t]))
    run('M3 cue words', m3)

    pd.DataFrame(rows).round(4).to_csv(os.path.join(OUT, 'table_tfidf_masking.csv'), index=False)
    pd.DataFrame(tops).to_csv(os.path.join(OUT, 'table_tfidf_masking_top_tokens.csv'), index=False)
    pd.Series(top200, name='token').to_csv(os.path.join(OUT, 'tfidf_M2_removed_tokens.csv'), index=False)
    print(pd.DataFrame(tops).to_string(index=False))


if __name__ == '__main__':
    main()
