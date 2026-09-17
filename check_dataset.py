import pandas as pd
import re

def check_dataset(filepath):
    df = pd.read_csv(filepath)
    print(f"Total samples: {len(df)}")
    
    # 1. Check for outright refusals
    refusal_keywords = [
        "I cannot", "I am unable", "As an AI", "I'm sorry, but", 
        "ethical guidelines", "I cannot fulfill", "against my programming"
    ]
    
    refusals = 0
    refusal_samples = []
    valid_rows = []
    
    # 2. Check for placeholders
    placeholder_pattern = r"\[.*?link.*?\]|<.*?link.*?>"
    placeholders = 0
    
    for idx, row in df.iterrows():
        text = str(row['text'])
        
        # Check refusals
        is_refusal = False
        for kw in refusal_keywords:
            if kw.lower() in text.lower():
                is_refusal = True
                break
        
        if is_refusal:
            refusals += 1
            if len(refusal_samples) < 3:
                refusal_samples.append(text)
        else:
            valid_rows.append(row)
                
        # Check placeholders
        if re.search(placeholder_pattern, text, re.IGNORECASE):
            placeholders += 1
            
    print(f"\n--- Analysis Results ---")
    print(f"Total Outright Refusals (Guardrails triggered): {refusals} ({(refusals/len(df))*100:.2f}%)")
    print(f"Emails containing placeholders (e.g. [Insert Link]): {placeholders} ({(placeholders/len(df))*100:.2f}%)")
    
    # Save the cleaned dataset
    clean_df = pd.DataFrame(valid_rows)
    clean_df.to_csv(filepath, index=False)
    print(f"\nSaved cleaned dataset with {len(clean_df)} valid samples back to {filepath}")
    
    if refusals > 0:
        print("\n--- Examples of Refusals ---")
        for i, sample in enumerate(refusal_samples):
            print(f"\nExample {i+1}:")
            print("-" * 40)
            # Print first 200 chars of the refusal
            print(sample[:200] + "...")

if __name__ == "__main__":
    check_dataset('ai_heldout_test.csv')
