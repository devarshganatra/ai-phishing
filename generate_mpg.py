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
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

import scenarios
from rate_limiter import RateLimiter
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

# Three Groq free-tier models that comply with phishing (P) prompts. allam and qwen refuse 0%;
# gpt-oss-20b refuses ~21% (the harness retries to hit the target and logs the refusals, so this
# only costs a few extra calls). gpt-oss-120b is excluded: it refuses ~62% and cannot form
# complete P/B pairs. Add a generator from another provider (gemini:/openrouter:/...) to go beyond.
DEFAULT_GENERATORS = [
    'groq:allam-2-7b',
    'groq:qwen/qwen3.8-27b',
    'groq:openai/gpt-oss-20b',
]
# A single fixed rewriter for the register stage; gpt-oss-20b follows the "keep intent, change
# only style" instruction cleanly.
DEFAULT_REWRITER = 'groq:openai/gpt-oss-20b'

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


def call_model(gen_id, system_prompt, user_prompt, limiter=None):
    """gen_id is 'provider:model'. Returns (status, text, raw_len).

    When a limiter is given, it is acquired before each attempt (pacing) and updated from the
    response's rate-limit headers afterward; a 429 waits for the limiter's header-aware backoff.
    """
    provider, model = gen_id.split(':', 1)
    client = get_client(provider)
    extras = MODEL_EXTRAS.get(model, {})
    for attempt in range(5):
        if limiter:
            limiter.acquire()
        try:
            raw_resp = client.chat.completions.with_raw_response.create(
                model=model,
                messages=[{'role': 'system', 'content': system_prompt},
                          {'role': 'user', 'content': user_prompt}],
                temperature=TEMPERATURE,
                max_completion_tokens=MAX_TOKENS,
                extra_body=extras,
            )
            if limiter:
                limiter.update(raw_resp.headers)
            resp = raw_resp.parse()
            raw = resp.choices[0].message.content or ''
            return 'ok', strip_reasoning(raw), len(raw)
        except Exception as e:
            msg = str(e).lower()
            if 'quota' in msg or 'insufficient' in msg:
                return 'quota', str(e), 0
            if 'rate limit' in msg or '429' in msg:
                hdrs = getattr(getattr(e, 'response', None), 'headers', None)
                if limiter:
                    limiter.backoff_429(hdrs)
                else:
                    time.sleep(2 * (attempt + 1))
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
    # Refusal is checked before length: refusals are short, and mislabeling them 'too_short'
    # would corrupt the per-generator refusal rate (a planned result).
    if is_refusal(text):
        return False, 'refusal', 0
    if len(text.split()) < 10:
        return False, 'too_short', 0
    n_ph = placeholder_count(text)
    if reject_placeholders and n_ph > 0:
        return False, 'placeholder', n_ph
    return True, 'ok', n_ph


def stage_generate(generators, per_cell, conditions, reject_placeholders=False, min_interval=None):
    """Generate the grid with one worker thread per model.

    Parallelism is ACROSS models: each model's calls are serialized through its own
    RateLimiter (so its free-tier limits are respected), but the four models run at the same
    time, so total wall time is bounded by the slowest model rather than their sum. Shared
    state (the output files and the dedup set) is guarded by locks.
    """
    counts, seen = load_done()
    write_lock = threading.Lock()
    seen_lock = threading.Lock()
    limiters = {}
    for gen in generators:
        lim = RateLimiter(gen.split(':', 1)[1])
        if min_interval is not None:
            lim.min_interval = max(lim.min_interval, min_interval)
        limiters[gen] = lim

    def worker(gen):
        limiter = limiters[gen]
        local_acc, local_att = [], []

        def flush():
            with write_lock:
                append_rows(EMAILS_FILE, local_acc, EMAIL_COLS)
                append_rows(ATTEMPTS_FILE, local_att, ATTEMPT_COLS)
            local_acc.clear()
            local_att.clear()

        for sidx in range(scenarios.NUM_SCENARIOS):
            for cond in conditions:
                need = max(0, per_cell - counts.get((gen, sidx, cond), 0))
                tries = 0
                consec_refusals = 0
                accepted_here = 0
                while need > 0 and tries < per_cell * 4:
                    tries += 1
                    # A model that refuses this condition outright (e.g. gpt-oss-120b on phishing)
                    # should not burn its daily quota retrying; bail after 5 straight refusals
                    # with nothing accepted. The refusals are still logged for the refusal rate.
                    if consec_refusals >= 5 and accepted_here == 0:
                        print(f"  [refusing] {gen} scenario {sidx} {cond}: {consec_refusals} refusals, skipping cell.")
                        break
                    status, text, raw_len = call_model(
                        gen, SYSTEM_PROMPT, scenarios.build_request(sidx, cond) + STYLE_RULES, limiter)
                    if status == 'quota':
                        local_att.append([gen, sidx, cond, 'quota', text[:120], raw_len, PROMPT_VERSION, now()])
                        flush()
                        print(f"  [quota] {gen} exhausted; stopping this model.")
                        return
                    if status == 'error':
                        local_att.append([gen, sidx, cond, 'error', text[:120], raw_len, PROMPT_VERSION, now()])
                        continue
                    ok, reason, n_ph = accept_or_reject(text, reject_placeholders)
                    consec_refusals = consec_refusals + 1 if reason == 'refusal' else 0
                    if ok:
                        sh = shingles(text)
                        with seen_lock:
                            if any(jaccard(sh, s) > 0.8 for s in seen):
                                ok, reason = False, 'duplicate'
                            else:
                                seen.append(sh)
                    local_att.append([gen, sidx, cond, 'ok' if ok else 'reject', reason, raw_len, PROMPT_VERSION, now()])
                    if ok:
                        local_acc.append([str(uuid.uuid4()), gen, sidx, cond, LABELS[cond],
                                          normalize_quotes(text).strip(), n_ph, '', '', PROMPT_VERSION, now()])
                        need -= 1
                        accepted_here += 1
                        if len(local_acc) >= 20:
                            flush()
        flush()
        print(f"[done] {gen}")

    with ThreadPoolExecutor(max_workers=len(generators)) as ex:
        futures = {ex.submit(worker, g): g for g in generators}
        for fut in futures:
            fut.result()
    stage_status()


def stage_rewrite(rewriter, per_source):
    if not os.path.exists(EMAILS_FILE):
        raise SystemExit("No mpg_emails.csv yet; run --stage generate first.")
    df = pd.read_csv(EMAILS_FILE)
    base = df[df['condition'].isin(['P', 'B_hard'])]
    existing = set(df.get('source_sample_id', pd.Series(dtype=str)).dropna())
    accepted, attempts = [], []
    limiter = RateLimiter(rewriter.split(':', 1)[1])
    for register, instruction in REWRITES.items():
        pool = base[~base['sample_id'].isin(existing)]
        # Sample per (generator, condition) with a manual loop: pandas 3.0's groupby().apply()
        # drops the grouping columns, which would lose 'condition'/'generator_model'.
        parts = [g.sample(min(len(g), per_source), random_state=42)
                 for _, g in pool.groupby(['generator_model', 'condition'])]
        sample = pd.concat(parts) if parts else pool.iloc[0:0]
        for _, r in sample.iterrows():
            status, text, raw_len = call_model(
                rewriter, "You rewrite emails. Output only the rewritten email.",
                instruction + "\n\n---\n" + str(r['text']), limiter)
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
    ap.add_argument('--min_interval', type=float, default=None,
                    help="floor (seconds) on the gap between calls, applied to every model on "
                         "top of its per-model default in rate_limiter.RATE_LIMITS")
    ap.add_argument('--reject_placeholders', action='store_true',
                    help="drop emails containing [slot] placeholders instead of just flagging them")
    ap.add_argument('--n', type=int, default=60, help="audit_sheet size")
    args = ap.parse_args()

    if args.stage == 'generate':
        stage_generate([g.strip() for g in args.generators.split(',') if g.strip()],
                       args.per_cell, [c.strip() for c in args.conditions.split(',')],
                       reject_placeholders=args.reject_placeholders, min_interval=args.min_interval)
    elif args.stage == 'rewrite':
        stage_rewrite(args.rewriter, args.per_source)
    elif args.stage == 'status':
        stage_status()
    elif args.stage == 'audit_sheet':
        stage_audit_sheet(args.n)


if __name__ == '__main__':
    main()
