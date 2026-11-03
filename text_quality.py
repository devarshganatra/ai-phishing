import re

# Curly quotes are what let "I’m sorry, but I can’t help with that." slip past
# the old straight-apostrophe filter, so every check normalizes them first.
QUOTE_MAP = str.maketrans({
    '‘': "'", '’': "'", '‚': "'", '′': "'",
    '“': '"', '”': '"', '„': '"', '″': '"',
})

REFUSAL_PATTERNS = [
    r"\bi'?m sorry\b", r"\bi am sorry\b", r"\bsorry, but\b",
    r"\bi can'?t (help|assist|comply|create|write|provide|do that|fulfill)",
    r"\bi cannot (help|assist|comply|create|write|provide|do that|fulfill)",
    r"\bi won'?t be able to\b", r"\bi'?m (not able|unable) to\b", r"\bi am (not able|unable) to\b",
    r"\bas an ai\b", r"\bagainst my (guidelines|programming|policy)\b",
    r"\bethical guidelines\b", r"\bnot able to help with\b",
]
REFUSAL_RE = re.compile('|'.join(REFUSAL_PATTERNS))

# [Insert Link], <link>, {recipient_name} ... anything that reads like an unfilled template slot
PLACEHOLDER_RE = re.compile(r"\[[^\]\n]{2,80}\]|\{[^}\n]{2,60}\}|<[^>\n]*(link|url|name|insert)[^>\n]*>", re.IGNORECASE)

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def normalize_quotes(text):
    return str(text).translate(QUOTE_MAP)


def strip_reasoning(text):
    """Drop <think>...</think> blocks that some reasoning models put in the content."""
    return THINK_RE.sub('', str(text)).strip()


def is_refusal(text):
    """A refusal phrase near the start of a short or subject-less reply.

    Real emails can say "I'm sorry for the inconvenience", so a phrase match alone
    is not enough: it must appear in the first 300 characters AND the reply must be
    short (< 80 words) or lack a Subject line.
    """
    t = normalize_quotes(text).lower().strip()
    if not t:
        return False
    if not REFUSAL_RE.search(t[:300]):
        return False
    return len(t.split()) < 80 or 'subject:' not in t


def placeholder_count(text):
    return len(PLACEHOLDER_RE.findall(str(text)))


def shingles(text, k=5):
    words = re.findall(r"\w+", normalize_quotes(text).lower())
    return {' '.join(words[i:i + k]) for i in range(max(1, len(words) - k + 1))}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
