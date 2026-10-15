import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler
from model import PhishingDistilBERT
from train import PhishingDataset
from tqdm import tqdm

def evaluate_dataset(model, loader, device):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            extra_features = batch.get('extra_features')
            if extra_features is not None:
                extra_features = extra_features.to(device)
                
            logits = model(input_ids, attention_mask, extra_features)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs >= 0.5).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy().flatten())
            
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.0
        
    return {
        'Accuracy': acc,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'AUC': auc
    }

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device for evaluation: {device}")
    
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    
    # Feature scaler
    feature_cols = ['perplexity', 'grammar_errors', 'word_count', 
                    'urgency_score', 'fear_score', 'authority_score']
    
    train_path = 'data/train_features.csv'
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found. Please run feature extraction first.")
        return
        
    train_df = pd.read_csv(train_path).fillna(0)
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])
    
    # Identify test files
    test_files = {}
    if os.path.exists('data/test_features.csv'):
        test_files['Standard Test (Kaggle/Enron)'] = 'data/test_features.csv'
        
    ai_test_path = 'data/ai_heldout_test_features.csv' if os.path.exists('data/ai_heldout_test_features.csv') else 'ai_heldout_test_features.csv'
    if os.path.exists(ai_test_path):
        test_files['AI-Generated Test (Groq Held-Out)'] = ai_test_path

    if not test_files:
        print("No test feature files found to evaluate!")
        return

    variants = ['baseline', 'behavioral', 'psychological', 'full']
    
    for test_name, file_path in test_files.items():
        print(f"\n=======================================================")
        print(f"  EVALUATION ON: {test_name}")
        print(f"  File: {file_path}")
        print(f"=======================================================")
        results = []
        
        for variant in variants:
            model_path = f'saved_models/model_{variant}.pt'
            if not os.path.exists(model_path):
                print(f"Model {model_path} not found, skipping...")
                continue
                
            num_extra_features = 0
            if variant in ['behavioral', 'psychological']:
                num_extra_features = 3
            elif variant == 'full':
                num_extra_features = 6
                
            model = PhishingDistilBERT(num_extra_features=num_extra_features).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            
            test_dataset = PhishingDataset(file_path, tokenizer, variant=variant, scaler=scaler)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
            
            metrics = evaluate_dataset(model, test_loader, device)
            results.append({
                'Variant': variant.upper(),
                'Accuracy': f"{metrics['Accuracy']*100:.2f}%",
                'Precision': f"{metrics['Precision']*100:.2f}%",
                'Recall': f"{metrics['Recall']*100:.2f}%",
                'F1-Score': f"{metrics['F1']*100:.2f}%",
                'ROC-AUC': f"{metrics['AUC']*100:.2f}%"
            })
            
        if results:
            df_res = pd.DataFrame(results)
            print("\n" + df_res.to_markdown(index=False) if hasattr(df_res, 'to_markdown') else df_res.to_string(index=False))

if __name__ == '__main__':
    main()
