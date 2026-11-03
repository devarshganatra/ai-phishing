"""TF-IDF + logistic-regression baseline, trained on the SAME split as DistilBERT (E2).

Fixes the reviewer's "unmatched training volume" objection: the baseline is fit on
data/train.csv (the same file DistilBERT trains on), its config is committed here rather
than hidden in code, and it dumps per-email predictions in the same long format as
evaluate_predictions.py so the paired tests compare like with like. Also computes a learning
curve (recall vs. training size) on request.

Usage:
  python tfidf_baseline.py --inputs data/test_features.csv mpg_features.csv --out tfidf_predictions_long.csv
  python tfidf_baseline.py --inputs data/test_features.csv --learning_curve
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Committed config (reviewer asked for exactly this to be in the paper, not "set in code").
TFIDF_CONFIG = dict(ngram_range=(1, 2), min_df=2, max_df=0.9, max_features=50000, sublinear_tf=True)
LOGREG_CONFIG = dict(C=10.0, max_iter=2000, class_weight='balanced')
SEED = 42
CARRY_COLS = ['generator_model', 'scenario_id', 'condition', 'register', 'sample_id', 'row_id']


def build_pipeline():
    return Pipeline([
        ('tfidf', TfidfVectorizer(**TFIDF_CONFIG)),
        ('clf', LogisticRegression(random_state=SEED, **LOGREG_CONFIG)),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', default='data/train.csv')
    ap.add_argument('--inputs', nargs='+', required=True)
    ap.add_argument('--out', default='tfidf_predictions_long.csv')
    ap.add_argument('--learning_curve', action='store_true')
    args = ap.parse_args()

    train = pd.read_csv(args.train).dropna(subset=['text', 'label'])
    pipe = build_pipeline()
    pipe.fit(train['text'].astype(str), train['label'].astype(int))
    print(f"Trained TF-IDF on {len(train)} rows | vocab={len(pipe.named_steps['tfidf'].vocabulary_)}")

    # Top coefficients: shows whether decisions rest on provenance tokens (enron, 2004...) vs. phishing cues.
    names = np.array(pipe.named_steps['tfidf'].get_feature_names_out())
    coef = pipe.named_steps['clf'].coef_[0]
    order = np.argsort(coef)
    print("  top safe tokens :", list(names[order[:15]]))
    print("  top phish tokens:", list(names[order[-15:]]))

    rows = []
    for path in args.inputs:
        if not os.path.exists(path):
            print(f"  skip missing {path}")
            continue
        df = pd.read_csv(path).reset_index(drop=True)
        name = os.path.basename(path).replace('_features.csv', '').replace('.csv', '')
        probs = pipe.predict_proba(df['text'].astype(str))[:, 1]
        labels = df['label'].values if 'label' in df.columns else np.full(len(df), -1)
        for i in range(len(df)):
            r = {'dataset': name, 'email_idx': i, 'variant': 'tfidf', 'seed': SEED,
                 'label': int(labels[i]), 'prob': float(probs[i])}
            for c in CARRY_COLS:
                if c in df.columns:
                    r[c] = df[c].iloc[i]
            rows.append(r)
        print(f"  scored {name}: {len(df)} emails")

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Wrote {len(rows)} prediction rows to {args.out}")

    if args.learning_curve:
        print("\nLearning curve (recall on first input, trained on subsamples):")
        test = pd.read_csv(args.inputs[0])
        yt = test['label'].values if 'label' in test.columns else None
        lc = []
        for n in [1600, 4000, 8000, len(train)]:
            n = min(n, len(train))
            sub = train.sample(n, random_state=SEED)
            p = build_pipeline().fit(sub['text'].astype(str), sub['label'].astype(int))
            if yt is not None:
                pred = (p.predict_proba(test['text'].astype(str))[:, 1] >= 0.5).astype(int)
                rec = ((pred == 1) & (yt == 1)).sum() / max(1, (yt == 1).sum())
                lc.append({'n_train': n, 'recall': round(float(rec), 4)})
                print(f"  n={n:6d}  recall={rec:.4f}")
        pd.DataFrame(lc).to_csv('tfidf_learning_curve.csv', index=False)


if __name__ == '__main__':
    main()
