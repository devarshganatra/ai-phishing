import os
import random
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler
from model import PhishingDistilBERT
from tqdm import tqdm


def set_seed(seed):
    """Seed every RNG so a run is reproducible and seed-to-seed variance is measurable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Feature columns used by each model variant. Single-feature variants (perplexity, grammar,
# length) let the ablation attribute any gain to one feature instead of the three-feature block.
VARIANT_FEATURES = {
    'baseline': [],
    'perplexity': ['perplexity'],
    'grammar': ['grammar_errors'],
    'length': ['word_count'],
    'behavioral': ['perplexity', 'grammar_errors', 'word_count'],
    'psychological': ['urgency_score', 'authority_score', 'fear_score'],
    'full': ['perplexity', 'grammar_errors', 'word_count', 'urgency_score', 'authority_score', 'fear_score'],
}

class PhishingDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=512, variant='baseline', scaler=None):
        self.df = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.variant = variant
        
        # Fill NaNs in features
        feature_cols = ['perplexity', 'grammar_errors', 'word_count', 
                        'urgency_score', 'fear_score', 'authority_score']
        for col in feature_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(0)

        # Apply pre-fitted scaler (train fits it; val/test reuse it to prevent data leakage)
        available = [c for c in feature_cols if c in self.df.columns]
        if available and scaler is not None:
            self.df[available] = scaler.transform(self.df[available])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.df.iloc[idx]['text'])
        label = float(self.df.iloc[idx]['label'])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.float)
        }
        
        cols = VARIANT_FEATURES.get(self.variant, [])
        if cols:
            features = self.df.iloc[idx][cols].values.astype(np.float32)
            item['extra_features'] = torch.tensor(features)

        return item

def train_model(variant, train_loader, val_loader, device, seed, epochs=3):
    print(f"\n--- Training Variant: {variant.upper()} | seed {seed} ---")

    num_extra_features = len(VARIANT_FEATURES[variant])
    model = PhishingDistilBERT(num_extra_features=num_extra_features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Mixed precision (fp16) on the T4 tensor cores: ~1.5-2x faster, half the memory.
    # No-op on CPU/MPS, so a single code path works everywhere.
    use_amp = (device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps), # 10% warmup
        num_training_steps=total_steps
    )

    best_f1 = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device).unsqueeze(1)
            extra_features = batch.get('extra_features')
            if extra_features is not None:
                extra_features = extra_features.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(input_ids, attention_mask, extra_features)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)                      # unscale before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_preds, val_labels = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device).unsqueeze(1)
                extra_features = batch.get('extra_features')
                if extra_features is not None:
                    extra_features = extra_features.to(device)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(input_ids, attention_mask, extra_features)
                preds = torch.sigmoid(logits).round().cpu().numpy().flatten()

                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy().flatten())
                
        # Metrics
        precision, recall, f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='binary')
        print(f"Epoch {epoch+1} | Loss: {avg_train_loss:.4f} | Val F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            os.makedirs('saved_models', exist_ok=True)
            torch.save(model.state_dict(), f'saved_models/model_{variant}_seed{seed}.pt')
            print("  --> Model saved!")
    return best_f1


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', default=','.join(VARIANT_FEATURES.keys()),
                    help="comma list; default all 7 variants")
    ap.add_argument('--seeds', default='0,1,2,3,4', help="comma list of seeds")
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--train', default='data/train_features.csv')
    ap.add_argument('--val', default='data/val_features.csv')
    ap.add_argument('--batch_size', type=int, default=16)
    # Multi-GPU without DataParallel: split the independent (seed, variant) runs across
    # processes. Launch one process per GPU, each with CUDA_VISIBLE_DEVICES set so it sees
    # a single card, e.g. on Kaggle's 2x T4:
    #   CUDA_VISIBLE_DEVICES=0 python train.py --shard 0 --num_shards 2 ... &
    #   CUDA_VISIBLE_DEVICES=1 python train.py --shard 1 --num_shards 2 ... &
    #   wait
    # Each run stays numerically identical to a single-GPU run (unlike DataParallel), which
    # keeps the seed-variance analysis clean.
    ap.add_argument('--num_shards', type=int, default=1, help="total parallel processes (= number of GPUs)")
    ap.add_argument('--shard', type=int, default=0, help="this process's shard index, 0..num_shards-1")
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device} | shard {args.shard}/{args.num_shards} | batch_size {args.batch_size}")

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    if not os.path.exists(args.train):
        print(f"{args.train} not found! Run extract_features.py first.")
        return

    variants = [v.strip() for v in args.variants.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]

    # Scaler is fit once on the training features and reused for val (no leakage).
    feature_cols = ['perplexity', 'grammar_errors', 'word_count',
                    'urgency_score', 'fear_score', 'authority_score']
    train_df_raw = pd.read_csv(args.train).fillna(0)
    available = [c for c in feature_cols if c in train_df_raw.columns]
    scaler = StandardScaler()
    if available:
        scaler.fit(train_df_raw[available])

    # Flatten to one job list, then keep only this process's slice. The stride split
    # (jobs[shard::num_shards]) interleaves seeds/variants so each GPU gets a balanced mix.
    all_jobs = [(seed, variant) for seed in seeds for variant in variants]
    jobs = all_jobs[args.shard::args.num_shards]
    print(f"This shard runs {len(jobs)}/{len(all_jobs)} jobs.")

    # Per-shard summary file so parallel processes don't clobber each other; merge after.
    summary_path = 'train_summary.csv' if args.num_shards == 1 else f'train_summary_shard{args.shard}.csv'

    summary = []
    for seed, variant in jobs:
        set_seed(seed)
        train_dataset = PhishingDataset(args.train, tokenizer, variant=variant, scaler=scaler)
        val_dataset = PhishingDataset(args.val, tokenizer, variant=variant, scaler=scaler)

        g = torch.Generator(); g.manual_seed(seed)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=2, pin_memory=True, generator=g)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2,
                                num_workers=2, pin_memory=True)

        best_f1 = train_model(variant, train_loader, val_loader, device, seed, epochs=args.epochs)
        summary.append({'variant': variant, 'seed': seed, 'best_val_f1': best_f1})
        pd.DataFrame(summary).to_csv(summary_path, index=False)

    print(f"\nTraining complete. Summary written to {summary_path}")
    if args.num_shards > 1:
        print("Merge shards afterwards with: "
              "pd.concat([pd.read_csv(f) for f in glob('train_summary_shard*.csv')]).to_csv('train_summary.csv', index=False)")


if __name__ == "__main__":
    main()
