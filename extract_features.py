import pandas as pd
import numpy as np
import spacy
import language_tool_python
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
from tqdm import tqdm
import os
import re
from concurrent.futures import ThreadPoolExecutor

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
gpt2_model.eval()  # set to eval mode once at startup

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
    if not text.strip():
        return 0.0
    encodings = gpt2_tokenizer(text, return_tensors='pt', truncation=True, max_length=1024)
    input_ids = encodings.input_ids.to(device)
    stride = 512
    max_length = gpt2_model.config.n_positions
    nlls = []
    for i in range(0, input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, input_ids.size(1))
        trg_len = end_loc - i
        chunk = input_ids[:, begin_loc:end_loc]
        target_ids = chunk.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = gpt2_model(chunk, labels=target_ids)
            nll = outputs.loss
            if not torch.isnan(nll):
                nlls.append(nll)
    if not nlls:
        return 0.0
    return min(torch.exp(torch.stack(nlls).mean()).item(), 10000.0)

def calculate_grammar_errors(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    sample = text[:1500]
    words = len(sample.split())
    if words == 0:
        return 0.0
    try:
        matches = lang_tool.check(sample)
        return (len(matches) / words) * 100
    except Exception:
        return 0.0

def extract_features(text):
    try:
        if not isinstance(text, str):
            text = ""
        if len(text) > 10000:
            text = text[:10000]
        text_lower = text.lower()
        word_count = len(text.split())
        try:
            perplexity = calculate_perplexity(text)
        except Exception:
            perplexity = 0.0
        grammar_errors = calculate_grammar_errors(text)
        urgency_count = sum(len(re.findall(kw, text_lower)) for kw in URGENCY_KEYWORDS)
        fear_count    = sum(len(re.findall(kw, text_lower)) for kw in FEAR_KEYWORDS)
        try:
            doc = nlp(text[:3000])
            org_entities = sum(1 for ent in doc.ents if ent.label_ == "ORG")
        except Exception:
            org_entities = 0
        title_count     = sum(len(re.findall(kw, text_lower)) for kw in AUTHORITY_TITLES)
        authority_score = org_entities + title_count
        return {
            'perplexity':      perplexity,
            'grammar_errors':  grammar_errors,
            'word_count':      word_count,
            'urgency_score':   urgency_count,
            'fear_score':      fear_count,
            'authority_score': authority_score
        }
    except Exception:
        return {'perplexity': 0.0, 'grammar_errors': 0.0, 'word_count': 0,
                'urgency_score': 0, 'fear_score': 0, 'authority_score': 0}

# ------------------------------------------------------------------
# CHECKPOINT-AWARE process_file
# Saves progress every CHECKPOINT_EVERY rows to a .checkpoint.csv
# If the runtime crashes or internet drops, re-run the same command
# and it will automatically resume from where it left off!
# ------------------------------------------------------------------
CHECKPOINT_EVERY = 200

def process_file(filepath, max_rows=None):
    print(f"\nProcessing: {filepath}")
    df = pd.read_csv(filepath)

    # Fast demo mode: balanced class subsample
    if max_rows is not None and max_rows < len(df):
        print(f"  Fast-mode: using {max_rows} rows out of {len(df)}")
        df = df.groupby('label', group_keys=False).apply(
            lambda x: x.sample(min(len(x), max_rows // 2), random_state=42)
        ).reset_index(drop=True)

    output_path     = filepath.replace('.csv', '_features.csv')
    checkpoint_path = output_path + '.checkpoint.csv'

    # Resume from checkpoint if it exists
    already_done = 0
    done_df      = None
    if os.path.exists(checkpoint_path):
        done_df      = pd.read_csv(checkpoint_path)
        already_done = len(done_df)
        print(f"  Resuming: {already_done} rows already done, {len(df) - already_done} remaining.")
        df = df.iloc[already_done:].reset_index(drop=True)
    else:
        print(f"  Starting fresh ({len(df)} rows)")

    if len(df) == 0:
        print("  All rows already processed. Writing final output...")
        done_df.to_csv(output_path, index=False)
        os.remove(checkpoint_path)
        print(f"  Saved: {output_path}")
        return

    all_features = []
    batch_texts  = df['text'].tolist()

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(extract_features, batch_texts)
        for i, feat in enumerate(tqdm(results, total=len(batch_texts), desc="Extracting features")):
            all_features.append(feat)
            # Save checkpoint every N rows
            if (i + 1) % CHECKPOINT_EVERY == 0:
                partial_df = pd.concat([df.iloc[:i+1].reset_index(drop=True),
                                        pd.DataFrame(all_features)], axis=1)
                if done_df is not None:
                    partial_df = pd.concat([done_df, partial_df], ignore_index=True)
                partial_df.to_csv(checkpoint_path, index=False)
                print(f"\n  Checkpoint saved at row {already_done + i + 1}")

    # Final merge and save
    new_df   = pd.concat([df.reset_index(drop=True), pd.DataFrame(all_features)], axis=1)
    final_df = pd.concat([done_df, new_df], ignore_index=True) if done_df is not None else new_df
    final_df.to_csv(output_path, index=False)
    print(f"\n  Done! Saved {len(final_df)} rows to: {output_path}")

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',    type=str, required=True,
                        help="Input CSV file path (e.g. data/train.csv)")
    parser.add_argument('--max_rows', type=int, default=None,
                        help="Optional: subsample N balanced rows for a fast demo run")
    args = parser.parse_args()
    process_file(args.input, max_rows=args.max_rows)
