"""Profanity / blocked-word filtering for training data and generation."""

from __future__ import annotations

import re

from config import BLOCKED_WORDS, ENABLE_PROFANITY_FILTER

try:
    from better_profanity import profanity

    profanity.load_censor_words()
    _HAS_PROFANITY = True
except ImportError:
    _HAS_PROFANITY = False

# better_profanity is very slow on long text; use it only for short strings
_MAX_PROFANITY_CHARS = 512
_BLOCK_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in BLOCKED_WORDS) + r")\b",
    re.I,
)


def contains_blocked_text(text: str) -> bool:
    if not ENABLE_PROFANITY_FILTER or not text:
        return False
    if _BLOCK_RE.search(text):
        return True
    if _HAS_PROFANITY and len(text) <= _MAX_PROFANITY_CHARS:
        return profanity.contains_profanity(text)
    return False


def is_safe_pair(text: str, title: str) -> bool:
    return not contains_blocked_text(text) and not contains_blocked_text(title)


def blocked_word_token_ids(tokenizer) -> set[int]:
    """Token ids whose decoded form is a blocked word (for logits masking)."""
    ids: set[int] = set()
    if not ENABLE_PROFANITY_FILTER:
        return ids
    for token, idx in tokenizer.get_vocab().items():
        decoded = token.replace("##", "").lower()
        if decoded in BLOCKED_WORDS:
            ids.add(idx)
    return ids
