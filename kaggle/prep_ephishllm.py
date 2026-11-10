"""Turn the E-PhishLLM release (ephishLLM.json from github.com/pajola/e-phishGen) into an English
CSV that the detectors can score.

The README describes a list of dictionaries with Subject, Body, type (0 legitimate, 1 phishing)
and Language; key casing varies, so keys are matched case-insensitively. Subject and body are
joined the same way as the matched-pair emails ("Subject: ...", blank line, body). Exact-duplicate
texts are dropped and the counts are printed, so they can be checked against the paper
(English subset: 11,502 emails, 5,996 phishing and 5,506 legitimate).

  python prep_ephishllm.py --json ephishLLM.json --out ephishllm_en.csv
"""
import argparse
import json

import pandas as pd


def pick(d, *names):
    low = {k.lower(): v for k, v in d.items()}
    for n in names:
        if n in low and low[n] is not None:
            return low[n]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True)
    ap.add_argument('--out', default='ephishllm_en.csv')
    ap.add_argument('--language', default='en')
    args = ap.parse_args()

    data = json.load(open(args.json, encoding='utf-8'))
    if isinstance(data, dict):  # tolerate {"emails": [...]} style wrappers
        data = next(v for v in data.values() if isinstance(v, list))
    print(f'records in file: {len(data)}; keys of first record: {sorted(data[0].keys())}')

    rows = []
    for i, d in enumerate(data):
        subj = pick(d, 'subject') or ''
        body = pick(d, 'body', 'text', 'content') or ''
        label = pick(d, 'type', 'label', 'class')
        lang = str(pick(d, 'language', 'lang') or '').strip().lower()
        if label is None or not str(body).strip():
            continue
        rows.append({'text': f'Subject: {str(subj).strip()}\n\n{str(body).strip()}',
                     'label': int(label), 'language': lang, 'sample_id': f'ephish-{i}'})
    df = pd.DataFrame(rows)
    print('by language:', df['language'].value_counts().to_dict())
    en = df[df['language'].isin([args.language, 'english'])]
    before = len(en)
    en = en.drop_duplicates(subset='text').reset_index(drop=True)
    print(f'{args.language}: {before} records, {len(en)} after removing exact duplicates; '
          f'labels {en["label"].value_counts().to_dict()}')
    en.to_csv(args.out, index=False)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
