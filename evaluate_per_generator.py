import os
import glob
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertModel, DistilBertTokenizerFast
from sklearn.preprocessing import StandardScaler

# Self-contained Model definition (no external model.py dependency needed)
class PhishingDistilBERT(nn.Module):
    def __init__(self, num_extra_features=0):
        super(PhishingDistilBERT, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.hidden_size = self.distilbert.config.hidden_size
        self.num_extra_features = num_extra_features
        
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size + self.num_extra_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        
    def forward(self, input_ids, attention_mask, extra_features=None):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        if self.num_extra_features > 0 and extra_features is not None:
            combined_output = torch.cat((cls_output, extra_features), dim=1)
        else:
            combined_output = cls_output
            
        logits = self.classifier(combined_output)
        return logits

class InMemoryPhishingDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512, variant='baseline', scaler=None):
        self.df = df.copy().reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.variant = variant
        
        feature_cols = ['perplexity', 'grammar_errors', 'word_count', 
                        'urgency_score', 'fear_score', 'authority_score']
        for col in feature_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(0)

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
        
        if self.variant == 'behavioral':
            features = self.df.iloc[idx][['perplexity', 'grammar_errors', 'word_count']].values.astype(np.float32)
            item['extra_features'] = torch.tensor(features)
        elif self.variant == 'psychological':
            features = self.df.iloc[idx][['urgency_score', 'authority_score', 'fear_score']].values.astype(np.float32)
            item['extra_features'] = torch.tensor(features)
        elif self.variant == 'full':
            features = self.df.iloc[idx][['perplexity', 'grammar_errors', 'word_count', 
                                          'urgency_score', 'authority_score', 'fear_score']].values.astype(np.float32)
            item['extra_features'] = torch.tensor(features)
            
        return item

def evaluate_subset(model, loader, device):
    model.eval()
    all_probs = []
    all_preds = []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            extra_features = batch.get('extra_features')
            if extra_features is not None:
                extra_features = extra_features.to(device)
                
            logits = model(input_ids, attention_mask, extra_features)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs >= 0.5).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    
    detection_rate = np.mean(all_preds == 1) * 100
    mean_prob = np.mean(all_probs) * 100
    total = len(all_preds)
    detected = int(np.sum(all_preds == 1))
    
    return {
        'total': total,
        'detected': detected,
        'detection_rate': detection_rate,
        'mean_prob': mean_prob
    }

def find_file(target_name):
    # Direct check
    if os.path.exists(target_name):
        return target_name
    for root_dir in ['.', '/content']:
        if os.path.exists(root_dir):
            for root, dirs, files in os.walk(root_dir):
                if target_name in files:
                    return os.path.join(root, target_name)
    return None

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device for evaluation: {device}")
    
    # Locate dataset files
    train_path = find_file('train_features.csv') or find_file('val_features.csv') or find_file('test_features.csv')
    ai_test_path = find_file('ai_heldout_test_features.csv') or find_file('ai_heldout_test.csv')
    
    if not train_path or not ai_test_path:
        print(f"Error: Missing files. train_path={train_path}, ai_test_path={ai_test_path}")
        print("\nScanning /content for available CSV/model files...")
        found_files = []
        for root_dir in ['.', '/content']:
            if os.path.exists(root_dir):
                for root, dirs, files in os.walk(root_dir):
                    for f in files:
                        if f.endswith('.csv') or f.endswith('.pt') or f.endswith('.zip'):
                            found_files.append(os.path.join(root, f))
        print("Found matching files in environment:")
        for f in sorted(set(found_files))[:20]:
            print("  ", f)
        if not found_files:
            print("  (No .csv, .pt, or .zip files found in current directory or /content)")
        return

    print(f"Fitting feature scaler on reference data: {train_path}")
    print(f"Loading AI test data from: {ai_test_path}")

    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    
    feature_cols = ['perplexity', 'grammar_errors', 'word_count', 
                    'urgency_score', 'fear_score', 'authority_score']
    train_df = pd.read_csv(train_path).fillna(0)
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])

    test_df = pd.read_csv(ai_test_path)
    if 'generator_model' not in test_df.columns:
        print("Warning: 'generator_model' column not found in test file. Available columns:", list(test_df.columns))
        return

    generators = sorted(test_df['generator_model'].unique())
    print(f"\nDiscovered Generators in Test Set: {generators}")
    for g in generators:
        count = len(test_df[test_df['generator_model'] == g])
        print(f"  - {g}: {count} samples")
    print(f"  - OVERALL: {len(test_df)} samples\n")

    variants = ['baseline', 'behavioral', 'psychological', 'full']
    models = {}
    for v in variants:
        path = find_file(f"model_{v}.pt")
        if path and os.path.exists(path):
            n_feats = 0 if v == 'baseline' else (3 if v in ['behavioral', 'psychological'] else 6)
            m = PhishingDistilBERT(num_extra_features=n_feats).to(device)
            m.load_state_dict(torch.load(path, map_location=device))
            models[v] = m
            print(f"Loaded {v} model from {path}")
        else:
            print(f"Warning: model_{v}.pt not found!")

    if not models:
        print("Error: No models found or loaded!")
        return

    summary_rows = []
    groups = [(g, test_df[test_df['generator_model'] == g]) for g in generators]
    groups.append(('OVERALL', test_df))

    for group_name, sub_df in groups:
        row = {'Generator / Source': group_name, 'N': len(sub_df)}
        for v in variants:
            if v not in models:
                row[f"{v.capitalize()} (%)"] = "N/A"
                continue
            
            dataset = InMemoryPhishingDataset(sub_df, tokenizer, variant=v, scaler=scaler)
            loader = DataLoader(dataset, batch_size=32, shuffle=False)
            metrics = evaluate_subset(models[v], loader, device)
            row[f"{v.capitalize()} (%)"] = f"{metrics['detection_rate']:.2f}%"
        summary_rows.append(row)

    res_df = pd.DataFrame(summary_rows)
    print("\n" + "="*70)
    print("  PER-GENERATOR DETECTION RATES (RECALL ON AI PHISHING EMAILS)")
    print("="*70)
    print(res_df.to_string(index=False))
    
    out_csv = 'per_generator_results.csv'
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved results to {out_csv}")

if __name__ == '__main__':
    main()
