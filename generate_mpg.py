"""Matched-pair generation harness (MPG) for the Intent-vs-Register study.

For each generator x scenario it produces three conditions (see scenarios.py):
  P       phishing email (label 1)
  B_hard  legitimate twin with the same cues (label 0)
  B_easy  plain legitimate email (label 0)

The `rewrite` stage then rewrites a subset of P and B_hard into one fixed register with one
fixed rewriter model, holding intent and facts constant (conditions P_R, B_hard_R).

Every API attempt (success, refusal, placeholder-reject, duplicate, error) is appended to
mpg_attempts.csv, so refusal/attrition rates become a reported result rather than a hidden
loss. Accepted emails go to mpg_emails.csv. Both files are append-only and the harness
resumes from them, so re-running after a quota reset just continues.

Usage:
  python generate_mpg.py --stage generate
  python generate_mpg.py --stage generate --generators groq:allam-2-7b --per_cell 1   # smoke test
  python generate_mpg.py --stage rewrite
  python generate_mpg.py --stage status
  python generate_mpg.py --stage audit_sheet --n 60
"""
import argparse
import os
import random
import sys
import time
import uuid
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

import scenarios
from text_quality import (is_refusal, jaccard, normalize_quotes,
                          placeholder_count, shingles, strip_reasoning)

load_dotenv()

PROMPT_VERSION = 'mpg-v1'
EMAILS_FILE = 'mpg_emails.csv'
ATTEMPTS_FILE = 'mpg_attempts.csv'

# OpenAI-compatible endpoints. Add a provider here and its key in .env to use it.
PROVIDERS = {
    'groq': ('https://api.groq.com/openai/v1', 'GROQ_API_KEY'),
    'gemini': ('https://generativelanguage.googleapis.com/v1beta/openai/', 'GEMINI_API_KEY'),
    'openrouter': ('https://openrouter.ai/api/v1', 'OPENROUTER_API_KEY'),
    'cerebras': ('https://api.cerebras.ai/v1', 'CEREBRAS_API_KEY'),
    'mistral': ('https://api.mistral.ai/v1', 'MISTRAL_API_KEY'),
}

DEFAULT_GENERATORS = [
    'groq:allam-2-7b',
    'groq:qwen/qwen3.8-27b',
    'groq:openai/gpt-oss-20b',
    'groq:openai/gpt-oss-120b',
]
DEFAULT_REWRITER = 'groq:openai/gpt-oss-120b'

# Keep reasoning hidden/short so it cannot consume the token budget (the likely cause of the
# original run keeping only 2/500 qwen emails).
MODEL_EXTRAS = {
    'qwen/qwen3.8-27b': {'reasoning_format': 'hidden'},
    'openai/gpt-oss-20b': {'reasoning_effort': 'low'},
    'openai/gpt-oss-120b': {'reasoning_effort': 'low'},
}

TEMPERATURE = 0.8
MAX_TOKENS = 1200

_clients = {}


def get_client(provider):
    if provider not in PROVIDERS:
        raise SystemExit(f"Unknown provider {provider!r}. Known: {list(PROVIDERS)}")
    base_url, key_env = PROVIDERS[provider]
    key = os.environ.get(key_env)
    if not key:
        raise SystemExit(f"{key_env} not set in environment/.env (needed for provider {provider!r}).")
    if provider not in _clients:
        _clients[provider] = OpenAI(api_key=key, base_url=base_url)
    return _clients[provider]


def call_model(gen_id, system_prompt, user_prompt):
    """gen_id is 'provider:model'. Returns (status, text, raw_len)."""
    provider, model = gen_id.split(':', 1)
    client = get_client(provider)
    extras = MODEL_EXTRAS.get(model, {})
    delay = 2
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{'role': 'system', 'content': system_prompt},
                          {'role': 'user', 'content': user_prompt}],
                temperature=TEMPERATURE,
                max_completion_tokens=MAX_TOKENS,
                extra_body=extras,
            )
            raw = resp.choices[0].message.content or ''
            return 'ok', strip_reasoning(raw), len(raw)
        except Exception as e:
            msg = str(e).lower()
            if 'quota' in msg or 'insufficient' in msg:
                return 'quota', str(e), 0
            if 'rate limit' in msg or '429' in msg:
                time.sleep(delay)
                delay *= 2
                continue
            return 'error', str(e), 0
    return 'error', 'max retries', 0


def load_done():
    """Return (accepted_counts, seen_shingles) from existing output, for resume + dedup."""
    counts = {}
    shings = []
    if os.path.exists(EMAILS_FILE):
        df = pd.read_csv(EMAILS_FILE)
        for _, r in df.iterrows():
            counts[(r['generator_model'], r['scenario_id'], r['condition'])] = \
                counts.get((r['generator_model'], r['scenario_id'], r['condition']), 0) + 1
            shings.append(shingles(r['text']))
    return counts, shings


def append_rows(path, rows, cols):
    if not rows:
        return
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(path, mode='a', header=not os.path.exists(path), index=False)


EMAIL_COLS = ['sample_id', 'generator_model', 'scenario_id', 'condition', 'label',
              'text', 'n_placeholders', 'source_sample_id', 'register', 'prompt_version', 'timestamp']
ATTEMPT_COLS = ['generator_model', 'scenario_id', 'condition', 'status', 'reason',
                'raw_len', 'prompt_version', 'timestamp']


def now():
    return datetime.now(timezone.utc).isoformat()


def accept_or_reject(text, reject_placeholders=False):
    """Decide whether a generated email is usable. Returns (ok, reason, n_placeholders).

    Placeholders are FLAGGED, not rejected, by default: some generators (e.g. allam-2-7b)
    emit [Name]/[link] slots in phishing and legitimate emails alike despite being told not
    to, so they are a generator signature rather than a phishing cue, and the matched-pair
    design controls for generator. The n_placeholders count is kept so a run can filter on it
    later; pass reject_placeholders=True to drop them at generation time instead.
    """
    text = normalize_quotes(text).strip()
    if len(text.split()) < 10:
        return False, 'too_short', 0
    if is_refusal(text):
        return False, 'refusal', 0
    n_ph = placeholder_count(text)
    if reject_placeholders and n_ph > 0:
        return False, 'placeholder', n_ph
    return True, 'ok', n_ph


def stage_generate(generators, per_cell, conditions, polite_sleep, reject_placeholders=False):
    counts, seen = load_done()
    attempts, accepted = [], []
    quota_blocked = set()
    try:
        for gen in generators:
            for sidx in range(scenarios.NUM_SCENARIOS):
                for cond in conditions:
                    if gen in quota_blocked:
                        continue
                    have = counts.get((gen, sidx, cond), 0)
                    need = max(0, per_cell - have)
                    tries = 0
                    while need > 0 and tries < per_cell * 4:
                        tries += 1
                        status, text, raw_len = call_model(
                            gen, SYSTEM_PROMPT, scenarios.build_request(sidx, cond) + STYLE_RULES)
                        if status == 'quota':
                            print(f"  [quota] {gen} exhausted; moving on.")
                            quota_blocked.add(gen)
                            attempts.append([gen, sidx, cond, 'quota', text[:120], raw_len, PROMPT_VERSION, now()])
                            break
                        if status == 'error':
                            attempts.append([gen, sidx, cond, 'error', text[:120], raw_len, PROMPT_VERSION, now()])
                            time.sleep(polite_sleep)
                            continue
                        ok, reason, n_ph = accept_or_reject(text, reject_placeholders)
                        if ok:
                            sh = shingles(text)
                            if any(jaccard(sh, s) > 0.8 for s in seen):
                                reason, ok = 'duplicate', False
                        attempts.append([gen, sidx, cond, 'ok' if ok else 'reject', reason, raw_len, PROMPT_VERSION, now()])
                        if ok:
                            seen.append(sh)
                            accepted.append([str(uuid.uuid4()), gen, sidx, cond, LABELS[cond],
                                             normalize_quotes(text).strip(), n_ph, '', '', PROMPT_VERSION, now()])
                            counts[(gen, sidx, cond)] = counts.get((gen, sidx, cond), 0) + 1
                            need -= 1
                        time.sleep(polite_sleep)
                        if len(accepted) >= 20:
                            append_rows(EMAILS_FILE, accepted, EMAIL_COLS); accepted = []
                            append_rows(ATTEMPTS_FILE, attempts, ATTEMPT_COLS); attempts = []
            print(f"[done] {gen}")
    finally:
        append_rows(EMAILS_FILE, accepted, EMAIL_COLS)
        append_rows(ATTEMPTS_FILE, attempts, ATTEMPT_COLS)
    stage_status()


def stage_rewrite(rewriter, per_source):
    if not os.path.exists(EMAILS_FILE):
        raise SystemExit("No mpg_emails.csv yet; run --stage generate first.")
    df = pd.read_csv(EMAILS_FILE)
    base = df[df['condition'].isin(['P', 'B_hard'])]
    existing = set(df.get('source_sample_id', pd.Series(dtype=str)).dropna())
    accepted, attempts = [], []
    for register, instruction in REWRITES.items():
        pool = base[~base['sample_id'].isin(existing)]
        sample = pool.groupby(['generator_model', 'condition'], group_keys=False).apply(
            lambda g: g.sample(min(len(g), per_source), random_state=42))
        for _, r in sample.iterrows():
            status, text, raw_len = call_model(
                rewriter, "You rewrite emails. Output only the rewritten email.",
                instruction + "\n\n---\n" + str(r['text']))
            cond = r['condition'] + '_R'
            if status != 'ok':
                attempts.append([rewriter, r['scenario_id'], cond, status, text[:120], raw_len, PROMPT_VERSION, now()])
                continue
            ok, reason, n_ph = accept_or_reject(text)
            attempts.append([rewriter, r['scenario_id'], cond, 'ok' if ok else 'reject', reason, raw_len, PROMPT_VERSION, now()])
            if ok:
                accepted.append([str(uuid.uuid4()), r['generator_model'], r['scenario_id'], cond,
                                 r['label'], normalize_quotes(text).strip(), n_ph, r['sample_id'],
                                 register, PROMPT_VERSION, now()])
    append_rows(EMAILS_FILE, accepted, EMAIL_COLS)
    append_rows(ATTEMPTS_FILE, attempts, ATTEMPT_COLS)
    print(f"Rewrites accepted: {len(accepted)}")
    stage_status()


def stage_status():
    if os.path.exists(EMAILS_FILE):
        df = pd.read_csv(EMAILS_FILE)
        print("\nAccepted emails by generator x condition:")
        print(pd.crosstab(df['generator_model'], df['condition']).to_string())
        print(f"Total accepted: {len(df)}")
    else:
        print("No emails generated yet.")
    if os.path.exists(ATTEMPTS_FILE):
        a = pd.read_csv(ATTEMPTS_FILE)
        print("\nAttempt outcomes by generator:")
        print(pd.crosstab(a['generator_model'], a['reason']).to_string())
        ref = a.groupby('generator_model')['reason'].apply(lambda s: (s == 'refusal').mean())
        print("\nRefusal rate by generator:")
        print(ref.to_string())


def stage_audit_sheet(n):
    """Blinded sheet of P / B_hard emails for human raters (label hidden)."""
    df = pd.read_csv(EMAILS_FILE)
    sub = df[df['condition'].isin(['P', 'B_hard'])].sample(min(n, len(df)), random_state=7)
    sheet = sub[['sample_id', 'text']].copy()
    sheet['rater_is_phishing'] = ''
    sheet['rater_is_legit'] = ''
    sheet.to_csv('mpg_audit_sheet.csv', index=False)
    sub[['sample_id', 'condition', 'label', 'generator_model']].to_csv('mpg_audit_key.csv', index=False)
    print(f"Wrote mpg_audit_sheet.csv ({len(sheet)} rows, blinded) and mpg_audit_key.csv")


# Fallbacks if scenarios.py does not define these (keeps prompt text centralized there when it does).
SYSTEM_PROMPT = getattr(scenarios, 'SYSTEM_PROMPT',
    "You are helping build a labeled email dataset for training and evaluating phishing "
    "detectors. Write exactly the email that is requested. Output only the email: a 'Subject:' "
    "line, a blank line, then the body. No commentary.")
STYLE_RULES = getattr(scenarios, 'STYLE_RULES',
    " Use concrete fictional details (names, companies, URLs, amounts, dates). Never use "
    "square-bracket or curly-brace placeholders. Do not add disclaimers or example notes.")
LABELS = getattr(scenarios, 'LABELS', {'P': 1, 'B_hard': 0, 'B_easy': 0, 'P_R': 1, 'B_hard_R': 0})
REWRITES = getattr(scenarios, 'REGISTER_REWRITES', {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['generate', 'rewrite', 'status', 'audit_sheet'])
    ap.add_argument('--generators', default=','.join(DEFAULT_GENERATORS),
                    help="comma list of provider:model")
    ap.add_argument('--rewriter', default=DEFAULT_REWRITER)
    ap.add_argument('--per_cell', type=int, default=15, help="accepted emails per generator x scenario x condition")
    ap.add_argument('--per_source', type=int, default=8, help="rewrite: emails per generator x condition per register")
    ap.add_argument('--conditions', default='P,B_hard,B_easy')
    ap.add_argument('--sleep', type=float, default=1.0)
    ap.add_argument('--reject_placeholders', action='store_true',
                    help="drop emails containing [slot] placeholders instead of just flagging them")
    ap.add_argument('--n', type=int, default=60, help="audit_sheet size")
    args = ap.parse_args()

    if args.stage == 'generate':
        stage_generate([g.strip() for g in args.generators.split(',') if g.strip()],
                       args.per_cell, [c.strip() for c in args.conditions.split(',')], args.sleep,
                       reject_placeholders=args.reject_placeholders)
    elif args.stage == 'rewrite':
        stage_rewrite(args.rewriter, args.per_source)
    elif args.stage == 'status':
        stage_status()
    elif args.stage == 'audit_sheet':
        stage_audit_sheet(args.n)


if __name__ == '__main__':
    main()
