"""Per-model, header-aware rate limiter for Groq (and other OpenAI-compatible) free tiers.

Each model gets its own limiter so the four generators can run concurrently without one
starving another. The limiter combines two mechanisms:

  1. A configurable minimum interval between calls (the "5 seconds empty" the brief asks for),
     set per model in RATE_LIMITS so a slow/strict model waits longer than a fast one.
  2. Header-driven throttling: after every response it reads Groq's x-ratelimit-remaining-*
     and x-ratelimit-reset-* headers and, when the remaining token/request budget would not
     cover the next call, sleeps until the bucket resets. This adapts automatically to the
     real free-tier limits (for allam the binding limit is ~6000 tokens/min) instead of
     relying on hardcoded guesses, and honours 429 resets.

Everything is thread-safe so one limiter can be shared by one model's worker thread.
"""
import re
import threading
import time

# Per-model floors. min_interval is the smallest gap between calls; the header logic adds
# more waiting when the token/minute budget is tight. Values reflect the limits observed on
# the free tier (allam: 7000 req/day, 6000 tok/min; others: 1000 req/day, 8000 tok/min).
RATE_LIMITS = {
    'allam-2-7b':            dict(min_interval=1.5, token_safety=900),
    'qwen/qwen3.8-27b':      dict(min_interval=2.0, token_safety=1200),
    'openai/gpt-oss-20b':    dict(min_interval=2.0, token_safety=1200),
    'openai/gpt-oss-120b':   dict(min_interval=2.0, token_safety=1200),
}
# Anything not listed is treated conservatively: the 5-second gap from the brief.
DEFAULT_LIMIT = dict(min_interval=5.0, token_safety=1500)


def parse_reset(value):
    """Groq reset strings look like '280ms', '7.66s', '1m2.5s', '26m7.542s'. Return seconds."""
    if not value:
        return 0.0
    value = str(value).strip()
    if value.endswith('ms') and 'm' == value[-2]:  # pure milliseconds, e.g. '280ms'
        try:
            return float(value[:-2]) / 1000.0
        except ValueError:
            return 0.0
    total = 0.0
    for num, unit in re.findall(r'([\d.]+)\s*(ms|m|s|h)', value):
        n = float(num)
        total += {'ms': n / 1000.0, 's': n, 'm': n * 60.0, 'h': n * 3600.0}[unit]
    return total


def config_for(model):
    return RATE_LIMITS.get(model, DEFAULT_LIMIT)


class RateLimiter:
    def __init__(self, model):
        cfg = config_for(model)
        self.model = model
        self.min_interval = cfg['min_interval']
        self.token_safety = cfg['token_safety']   # keep this many tokens in the bucket as headroom
        self.lock = threading.Lock()
        self.last = 0.0
        self.limit_tokens = None
        self.remaining_tokens = None
        self.reset_tokens = 0.0
        self.remaining_requests = None
        self.reset_requests = 0.0

    def acquire(self):
        """Block until it is safe to make the next call for this model.

        When the per-minute token bucket is below the safety headroom, wait only long enough
        for it to refill the deficit (the bucket refills continuously), not the full reset
        time. This is the difference between a ~6 s pause and a ~60 s stall per low point.
        """
        with self.lock:
            now = time.time()
            wait = self.min_interval - (now - self.last)
            if (self.remaining_tokens is not None and self.limit_tokens
                    and self.remaining_tokens < self.token_safety and self.reset_tokens > 0):
                # refill rate (tokens/sec) implied by headers: (limit - remaining) / time-to-full
                refill_rate = (self.limit_tokens - self.remaining_tokens) / self.reset_tokens
                deficit = self.token_safety - self.remaining_tokens
                partial = deficit / refill_rate if refill_rate > 0 else self.reset_tokens
                wait = max(wait, min(partial, self.reset_tokens) + 0.2)
            if self.remaining_requests is not None and self.remaining_requests <= 0:
                wait = max(wait, self.reset_requests + 0.2)
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()

    def update(self, headers):
        """Record the rate-limit headers returned by the last successful call."""
        if not headers:
            return
        h = {k.lower(): v for k, v in dict(headers).items()}
        with self.lock:
            lt = h.get('x-ratelimit-limit-tokens')
            rt = h.get('x-ratelimit-remaining-tokens')
            rr = h.get('x-ratelimit-remaining-requests')
            if lt is not None:
                try:
                    self.limit_tokens = float(lt)
                except ValueError:
                    pass
            if rt is not None:
                try:
                    self.remaining_tokens = float(rt)
                except ValueError:
                    pass
                self.reset_tokens = parse_reset(h.get('x-ratelimit-reset-tokens'))
            if rr is not None:
                try:
                    self.remaining_requests = float(rr)
                except ValueError:
                    pass
                self.reset_requests = parse_reset(h.get('x-ratelimit-reset-requests'))

    def backoff_429(self, headers):
        """Sleep after a 429, honouring Retry-After / reset headers when present."""
        secs = 2.0
        if headers:
            h = {k.lower(): v for k, v in dict(headers).items()}
            secs = max(parse_reset(h.get('retry-after')),
                       parse_reset(h.get('x-ratelimit-reset-tokens')),
                       parse_reset(h.get('x-ratelimit-reset-requests')), 2.0)
        time.sleep(min(secs + 0.5, 65.0))
