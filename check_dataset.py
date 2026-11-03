import os
import pandas as pd
from text_quality import is_refusal, placeholder_count


def check_dataset(filepath, features_path=None):
    """Flag refusals and placeholders without touching the source file.

    Writes:
      <name>_audit.csv          every row with is_refusal / n_placeholders / is_exact_dup flags
      <name>_valid.csv          rows that are real emails (not refusals)
      <name>_valid_features.csv same filter applied to the features file, if given
    """
    df = pd.read_csv(filepath)
    df['row_id'] = range(len(df))
    df['is_refusal'] = df['text'].apply(is_refusal)
    df['n_placeholders'] = df['text'].apply(placeholder_count)
    df['is_exact_dup'] = df.duplicated(subset=['text'], keep='first')

    group_col = 'generator_model' if 'generator_model' in df.columns else None
    print(f"Total samples: {len(df)}")
    if group_col:
        summary = df.groupby(group_col).agg(
            n=('text', 'size'),
            unique_texts=('text', 'nunique'),
            refusals=('is_refusal', 'sum'),
            with_placeholders=('n_placeholders', lambda s: int((s > 0).sum())),
        )
        summary['valid'] = summary['n'] - summary['refusals']
        print(summary.to_string())
    print(f"\nRefusals: {int(df['is_refusal'].sum())}  |  "
          f"Rows with placeholders: {int((df['n_placeholders'] > 0).sum())}  |  "
          f"Valid emails: {int((~df['is_refusal']).sum())}")

    base = os.path.splitext(filepath)[0]
    df.to_csv(f'{base}_audit.csv', index=False)
    valid = df[~df['is_refusal']]
    valid.to_csv(f'{base}_valid.csv', index=False)
    print(f"\nSaved {base}_audit.csv and {base}_valid.csv ({len(valid)} rows)")

    if features_path and os.path.exists(features_path):
        feats = pd.read_csv(features_path)
        if len(feats) != len(df) or not (feats['text'].values == df['text'].values).all():
            print(f"Warning: {features_path} rows do not line up with {filepath}; skipping features filter.")
        else:
            feats = feats.assign(row_id=df['row_id'], is_refusal=df['is_refusal'],
                                 n_placeholders=df['n_placeholders'])
            out = f'{base}_valid_features.csv'
            feats[~feats['is_refusal']].to_csv(out, index=False)
            print(f"Saved {out}")


if __name__ == "__main__":
    check_dataset('ai_heldout_test.csv', features_path='ai_heldout_test_features.csv')
