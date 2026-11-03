"""Score saved models on one or more feature CSVs and dump PER-EMAIL predictions.

Writes a long-format CSV with one row per (email, variant, seed): the predicted probability,
the label, and any generator/condition columns present. McNemar and paired-bootstrap tests all
read from this file, so nothing downstream has to re-run the models.

Usage:
  python evaluate_predictions.py --inputs data/test_features.csv mpg_features.csv --out predictions_long.csv
  python evaluate_predictions.py --inputs data/test_features.csv --variants baseline,behavioral --seeds 0,1,2
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast

from model import PhishingDistilBERT
from train import VARIANT_FEATURES, PhishingDataset

CARRY_COLS = ['generator_model', 'scenario_id', 'condition', 'register', 'sample_id', 'row_id']


def find_checkpoints(variants, seeds):
    """Return list of (variant, seed, path) for model_<variant>_seed<seed>.pt that exist."""
    found = []
    for path in sorted(glob.glob('saved_models/model_*_seed*.pt')):
        m = re.match(r'saved_models/model_(.+)_seed(\d+)\.pt', path)
        if not m:
            continue
        variant, seed = m.group(1), int(m.group(2))
        if variants and variant not in variants:
            continue
        if seeds and seed not in seeds:
            continue
        found.append((variant, seed, path))
    return found


def score(model, loader, device):
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            extra = batch.get('extra_features')
            if extra is not None:
                extra = extra.to(device)
            logits = model(input_ids, attention_mask, extra)
            probs.extend(torch.sigmoid(logits).cpu().numpy().flatten())
    return np.array(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', nargs='+', required=True, help="feature CSV(s) to score")
    ap.add_argument('--train', default='data/train_features.csv', help="for fitting the feature scaler")
    ap.add_argument('--variants', default='', help="comma list; default = all found")
    ap.add_argument('--seeds', default='', help="comma list; default = all found")
    ap.add_argument('--out', default='predictions_long.csv')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    from sklearn.preprocessing import StandardScaler
    feat_cols = ['perplexity', 'grammar_errors', 'word_count', 'urgency_score', 'fear_score', 'authority_score']
    scaler = StandardScaler()
    tr = pd.read_csv(args.train).fillna(0)
    scaler.fit(tr[[c for c in feat_cols if c in tr.columns]])

    variants = [v for v in args.variants.split(',') if v] or None
    seeds = [int(s) for s in args.seeds.split(',') if s] or None
    ckpts = find_checkpoints(variants, seeds)
    if not ckpts:
        print("No matching checkpoints under saved_models/ (expected model_<variant>_seed<seed>.pt).")
        return
    print(f"Scoring {len(ckpts)} checkpoints on {len(args.inputs)} file(s).")

    rows = []
    for input_path in args.inputs:
        if not os.path.exists(input_path):
            print(f"  skip missing {input_path}")
            continue
        df = pd.read_csv(input_path).reset_index(drop=True)
        dataset_name = os.path.basename(input_path).replace('_features.csv', '').replace('.csv', '')
        carried = {c: df[c].values for c in CARRY_COLS if c in df.columns}
        labels = df['label'].values if 'label' in df.columns else np.full(len(df), -1)

        for variant, seed, path in ckpts:
            model = PhishingDistilBERT(num_extra_features=len(VARIANT_FEATURES[variant])).to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            ds = PhishingDataset(input_path, tokenizer, variant=variant, scaler=scaler)
            loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)
            probs = score(model, loader, device)
            for i in range(len(df)):
                r = {'dataset': dataset_name, 'email_idx': i, 'variant': variant, 'seed': seed,
                     'label': int(labels[i]), 'prob': float(probs[i])}
                for c, vals in carried.items():
                    r[c] = vals[i]
                rows.append(r)
            print(f"  {dataset_name}: {variant} seed{seed} scored ({len(df)} emails)")

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"\nWrote {len(out)} prediction rows to {args.out}")


if __name__ == '__main__':
    main()
