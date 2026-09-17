import pandas as pd
import numpy as np
import spacy
import language_tool_python
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
from tqdm import tqdm
import os
import re

# Initialize tools
print("Loading NLP models...")
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading en_core_web_sm...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

lang_tool = language_tool_python.LanguageTool('en-US')

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device for perplexity: {device}")
gpt2_model_id = "gpt2"
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_id).to(device)
gpt2_tokenizer = GPT2TokenizerFast.from_pretrained(gpt2_model_id)

# Keywords
URGENCY_KEYWORDS = [
    r"\bimmediately\b", r"\burgent\b", r"\bwithin \d+ hours\b", r"\baction required\b",
    r"\basap\b", r"\bimportant\b", r"\bsoon\b", r"\bfinal notice\b"
]

FEAR_KEYWORDS = [
    r"\bsuspended\b", r"\bterminated\b", r"\bpenalty\b", r"\blegal action\b",
    r"\bblocked\b", r"\bdeactivated\b", r"\bcompromised\b", r"\bunauthorized\b",
    r"\bfailure\b", r"\bwarning\b"
]

AUTHORITY_TITLES = [
    r"\bceo\b", r"\bcfo\b", r"\bhr\b", r"\bit department\b", r"\badmin\b",
    r"\bsupport\b", r"\bmanager\b", r"\bdirector\b", r"\bpresident\b", r"\bhelpdesk\b"
]

def calculate_perplexity(text):
    if not text.strip(): return 0.0
    # Truncate to 1024 tokens to prevent GPT-2 indexing errors on long emails
    encodings = gpt2_tokenizer(text, return_tensors='pt', truncation=True, max_length=1024)
    input_ids = encodings.input_ids.to(device)
    
    max_length = gpt2_model.config.n_positions
    stride = 512
    
    nlls = []
    # If text is too long, we use a sliding window
    for i in range(0, input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, input_ids.size(1))
        trg_len = end_loc - i
        input_ids_chunk = input_ids[:, begin_loc:end_loc]
        target_ids = input_ids_chunk.clone()
        target_ids[:, :-trg_len] = -100
        
        with torch.no_grad():
            outputs = gpt2_model(input_ids_chunk, labels=target_ids)
            # loss is calculated using CrossEntropyLoss which averages over valid labels
            neg_log_likelihood = outputs.loss
            # check if nan
            if not torch.isnan(neg_log_likelihood):
                nlls.append(neg_log_likelihood)
                
    if not nlls:
        return 0.0
    
    ppl = torch.exp(torch.stack(nlls).mean()).item()
    return min(ppl, 10000.0) # Cap at 10000 to avoid inf

def calculate_grammar_errors(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return 0.0
    words = len(text.split())
    if words == 0:
        return 0.0
    matches = lang_tool.check(text)
    # Errors per 100 words
    return (len(matches) / words) * 100

def extract_features(text):
    if not isinstance(text, str):
        text = ""
    
    # Truncate extremely long anomalous emails to 10,000 characters to prevent Spacy memory exhaustion
    # DistilBERT only supports 512 tokens (~2000 chars) anyway.
    if len(text) > 10000:
        text = text[:10000]
        
    text_lower = text.lower()
    doc = nlp(text)
    
    # 1. Message Length (Words)
    word_count = len(text.split())
    
    # 2. Perplexity
    perplexity = calculate_perplexity(text)
    
    # 3. Grammar Consistency
    grammar_errors = calculate_grammar_errors(text)
    
    # 4. Urgency Lexicon
    urgency_count = sum(len(re.findall(kw, text_lower)) for kw in URGENCY_KEYWORDS)
    
    # 5. Fear Appeals
    fear_count = sum(len(re.findall(kw, text_lower)) for kw in FEAR_KEYWORDS)
    
    # 6. Authority Impersonation
    # Count org entities from Spacy + regex for job titles
    org_entities = sum(1 for ent in doc.ents if ent.label_ == "ORG")
    title_count = sum(len(re.findall(kw, text_lower)) for kw in AUTHORITY_TITLES)
    authority_score = org_entities + title_count
    
    return {
        'perplexity': perplexity,
        'grammar_errors': grammar_errors,
        'word_count': word_count,
        'urgency_score': urgency_count,
        'fear_score': fear_count,
        'authority_score': authority_score
    }

from concurrent.futures import ThreadPoolExecutor

def process_file(filepath):
    print(f"Processing {filepath}...")
    df = pd.read_csv(filepath)
    
    with ThreadPoolExecutor(max_workers=16) as executor:
        # We use executor.map to parallelize the extraction
        # LanguageTool is heavily IO-bound (HTTP requests), so 16 threads will speed this up by 10-15x!
        features_list = list(tqdm(executor.map(extract_features, df['text']), total=len(df), desc="Extracting features"))
        
    features_df = pd.DataFrame(features_list)
    
    # Combine original df with features
    final_df = pd.concat([df, features_df], axis=1)
    
    output_path = filepath.replace('.csv', '_features.csv')
    final_df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help="Input CSV file")
    args = parser.parse_args()
    
    process_file(args.input)
