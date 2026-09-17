import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
import os

def prepare_data(input_csv, output_dir):
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    print(f"Original shape: {df.shape}")
    
    # Basic cleaning
    # Assuming the Kaggle dataset has 'Email Text' and 'Email Type' columns
    # We will rename them to 'text' and 'label' for consistency
    
    # Print columns to handle variations in Kaggle datasets
    print("Columns in dataset:", df.columns.tolist())
    
    text_col = 'Email Text'
    label_col = 'Email Type'
    
    if 'text' in df.columns:
        text_col = 'text'
    if 'label' in df.columns:
        label_col = 'label'
        
    df = df.rename(columns={text_col: 'text', label_col: 'label'})
    
    # Drop rows with missing text or labels
    df = df.dropna(subset=['text', 'label'])
    
    # Drop exact duplicates
    df = df.drop_duplicates(subset=['text'])
    
    print(f"Shape after cleaning: {df.shape}")
    
    print("Mapping string labels to binary...")
    df['label'] = df['label'].astype(str).str.strip()
    label_map = {
        'Safe Email': 0,
        'Phishing Email': 1,
        'ham': 0,
        'spam': 1,
        '0': 0,
        '1': 1
    }
    df['label'] = df['label'].map(label_map)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    
    print("Label distribution:")
    print(df['label'].value_counts())
    
    # Train/Val/Test Split (80/10/10)
    print("Splitting dataset...")
    # First split into 80% train and 20% temp (val+test)
    X_train_text, X_temp_text, y_train, y_temp = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )
    
    # Then split temp into 50% val and 50% test (10% and 10% of total)
    X_val_text, X_test_text, y_val, y_test = train_test_split(
        X_temp_text, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"Train size: {len(X_train_text)}, Val size: {len(X_val_text)}, Test size: {len(X_test_text)}")
    
    # Apply SMOTE to training data
    # To apply SMOTE to text, we first need to vectorize it (TF-IDF is a common baseline for this step)
    # However, since we are going to fine-tune DistilBERT which takes raw text, 
    # applying SMOTE on raw text is tricky.
    # Usually, for deep learning, we handle imbalance via Class Weights in the Loss Function (BCEWithLogitsLoss)
    # rather than SMOTE (which generates synthetic TF-IDF vectors, not synthetic raw text).
    #
    # If the user strictly wants SMOTE for DistilBERT, we'd have to oversample the text strings by duplication,
    # or apply SMOTE on the DistilBERT embeddings.
    #
    # For now, we will perform Random Oversampling on the training text to balance the classes as a 
    # text-compatible alternative to SMOTE.
    
    print("Applying Random Oversampling to training set to handle imbalance...")
    from imblearn.over_sampling import RandomOverSampler
    ros = RandomOverSampler(random_state=42)
    # ROS expects 2D array for X
    X_train_resampled, y_train_resampled = ros.fit_resample(
        X_train_text.values.reshape(-1, 1), y_train
    )
    X_train_text_resampled = pd.Series(X_train_resampled.flatten())
    
    print("New Train label distribution:")
    print(pd.Series(y_train_resampled).value_counts())
    
    # Save splits to output directory
    os.makedirs(output_dir, exist_ok=True)
    
    train_df = pd.DataFrame({'text': X_train_text_resampled, 'label': y_train_resampled})
    val_df = pd.DataFrame({'text': X_val_text, 'label': y_val})
    test_df = pd.DataFrame({'text': X_test_text, 'label': y_test})
    
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    print("Data preparation complete. Files saved to:", output_dir)

if __name__ == "__main__":
    prepare_data('Phishing_Email.csv', 'data')
