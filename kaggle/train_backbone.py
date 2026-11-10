"""Train a text-only Transformer phishing classifier and score every test set in one run.

Used for the stronger-baseline experiment (RoBERTa-base) and, when the original checkpoints
are gone, to re-run the DistilBERT baseline. The setup matches train.py for a controlled
comparison: same split, [CLS]/<s> embedding -> 256 ReLU -> dropout 0.2 -> 1 logit, AdamW lr 2e-5,
weight decay 0.01, batch 16, 3 epochs, 10% warm-up, clip 1.0, fp16, best-validation-F1 checkpoint.

Each seed is scored on every --score file, and predictions are written in the same long format
as evaluate_predictions.py (dataset, email_idx, variant, seed, label, prob + carried columns), so
analysis/paper_stats.py reads them unchanged.

Two GPUs: run one process per card, e.g.
  CUDA_VISIBLE_DEVICES=0 python train_backbone.py --shard 0 --num_shards 2 ... &
  CUDA_VISIBLE_DEVICES=1 python train_backbone.py --shard 1 --num_shards 2 ... &
  wait
"""
import argparse
import copy
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

CARRY = ['generator_model', 'scenario_id', 'condition', 'register', 'sample_id', 'language']


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Classifier(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone)
        self.head = torch.nn.Sequential(torch.nn.Linear(self.encoder.config.hidden_size, 256), torch.nn.ReLU(),
                                        torch.nn.Dropout(0.2), torch.nn.Linear(256, 1))

    def forward(self, ids, mask):
        return self.head(self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0, :])


def encode(tok, texts, max_len):
    enc = tok([str(t) for t in texts], truncation=True, padding='max_length', max_length=max_len,
              return_tensors='pt')
    return enc['input_ids'], enc['attention_mask']


def predict(model, ids, mask, device, use_amp, bs=64):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(ids), bs):
            with torch.cuda.amp.autocast(enabled=use_amp):
                logit = model(ids[i:i + bs].to(device), mask[i:i + bs].to(device))
            out.append(torch.sigmoid(logit.float()).cpu().numpy().ravel())
    return np.concatenate(out) if out else np.array([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbone', default='roberta-base')
    ap.add_argument('--variant', default='roberta', help='name written into the prediction files')
    ap.add_argument('--train', required=True)
    ap.add_argument('--val', required=True)
    ap.add_argument('--score', nargs='+', required=True, help='name=path pairs, e.g. test=test_features.csv')
    ap.add_argument('--seeds', default='0,1,2,3,4')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num_shards', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--max_len', type=int, default=512)
    ap.add_argument('--max_train', type=int, default=None, help='smoke test only: subsample the training set')
    ap.add_argument('--out_dir', default='.')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    seeds = [int(s) for s in args.seeds.split(',')][args.shard::args.num_shards]
    print(f'{args.backbone} on {device} | shard {args.shard}/{args.num_shards} | seeds {seeds}')
    if not seeds:
        return

    tok = AutoTokenizer.from_pretrained(args.backbone)
    train = pd.read_csv(args.train).dropna(subset=['text', 'label'])
    if args.max_train:
        train = train.sample(min(args.max_train, len(train)), random_state=0)
    val = pd.read_csv(args.val).dropna(subset=['text', 'label'])
    # Tokenise once; every seed reuses the tensors.
    tr_ids, tr_mask = encode(tok, train['text'], args.max_len)
    tr_y = torch.tensor(train['label'].astype(float).values, dtype=torch.float32)
    va_ids, va_mask = encode(tok, val['text'], args.max_len)
    va_y = val['label'].astype(int).values
    scored = {}
    for pair in args.score:
        name, path = pair.split('=', 1)
        df = pd.read_csv(path).reset_index(drop=True)
        scored[name] = (df, *encode(tok, df['text'].fillna(''), args.max_len))
        print(f'  scoring set {name}: {len(df)} emails')

    rows, summary = [], []
    for seed in seeds:
        set_seed(seed)
        model = Classifier(args.backbone).to(device)
        g = torch.Generator()
        g.manual_seed(seed)
        loader = DataLoader(TensorDataset(tr_ids, tr_mask, tr_y), batch_size=args.batch_size, shuffle=True,
                            generator=g)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        steps = len(loader) * args.epochs
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        lossf = torch.nn.BCEWithLogitsLoss()
        best_f1, best_state = -1.0, None
        for epoch in range(args.epochs):
            model.train()
            for ids, mask, y in tqdm(loader, desc=f'seed {seed} epoch {epoch + 1}', mininterval=30):
                opt.zero_grad()
                with torch.cuda.amp.autocast(enabled=use_amp):
                    loss = lossf(model(ids.to(device), mask.to(device)).float(), y.to(device).unsqueeze(1))
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
            p = predict(model, va_ids, va_mask, device, use_amp)
            _, _, f1, _ = precision_recall_fscore_support(va_y, (p >= 0.5).astype(int), average='binary',
                                                          zero_division=0)
            print(f'  seed {seed} epoch {epoch + 1}: val F1 {f1:.4f}')
            if f1 > best_f1:
                best_f1, best_state = f1, copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
        model.load_state_dict(best_state)
        summary.append({'variant': args.variant, 'seed': seed, 'best_val_f1': best_f1})
        for name, (df, ids, mask) in scored.items():
            prob = predict(model, ids, mask, device, use_amp)
            labels = df['label'].values if 'label' in df.columns else np.full(len(df), -1)
            carry = {c: df[c].values for c in CARRY if c in df.columns}
            for i in range(len(df)):
                r = {'dataset': name, 'email_idx': i, 'variant': args.variant, 'seed': seed,
                     'label': int(labels[i]), 'prob': float(prob[i])}
                for c, v in carry.items():
                    r[c] = v[i]
                rows.append(r)
            print(f'  seed {seed}: scored {name}')
        del model
        torch.cuda.empty_cache()
        tag = f'{args.variant}_shard{args.shard}'
        pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, f'predictions_{tag}.csv'), index=False)
        pd.DataFrame(summary).to_csv(os.path.join(args.out_dir, f'train_summary_{tag}.csv'), index=False)
    print('done')


if __name__ == '__main__':
    main()
