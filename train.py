import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler
from model import PhishingDistilBERT
from tqdm import tqdm

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

def train_model(variant, train_loader, val_loader, device, epochs=3):
    print(f"\n--- Training Variant: {variant.upper()} ---")
    
    num_extra_features = 0
    if variant == 'behavioral' or variant == 'psychological':
        num_extra_features = 3
    elif variant == 'full':
        num_extra_features = 6
        
    model = PhishingDistilBERT(num_extra_features=num_extra_features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    
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
                
            logits = model(input_ids, attention_mask, extra_features)
            loss = criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
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
            torch.save(model.state_dict(), f'saved_models/model_{variant}.pt')
            print("  --> Model saved!")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    
    # Check if feature files exist
    if not os.path.exists('data/train_features.csv'):
        print("Feature files not found! Please run extract_features.py on the datasets first.")
        return
        
    variants = ['baseline', 'behavioral', 'psychological', 'full']
    
    for variant in variants:
        # Fit scaler on training data, then reuse it for val (prevents data leakage)
        feature_cols = ['perplexity', 'grammar_errors', 'word_count', 
                        'urgency_score', 'fear_score', 'authority_score']
        train_df_raw = pd.read_csv('data/train_features.csv').fillna(0)
        available = [c for c in feature_cols if c in train_df_raw.columns]
        scaler = StandardScaler()
        if available:
            scaler.fit(train_df_raw[available])

        train_dataset = PhishingDataset('data/train_features.csv', tokenizer, variant=variant, scaler=scaler)
        val_dataset = PhishingDataset('data/val_features.csv', tokenizer, variant=variant, scaler=scaler)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=32, num_workers=2, pin_memory=True)
        
        train_model(variant, train_loader, val_loader, device, epochs=3)

if __name__ == "__main__":
    main()
