"""A small BM25 ranker with a tokenizer that handles both Latin and CJK text."""

from __future__ import annotations

import math
import re
from collections import Counter

_WORD = re.compile(r"[a-z0-9_]+")
_CJK = re.compile(r"[一-鿿]")

# Function words carry no topic. Without removing them, "what is the capital of Japan" would keyword-match any
# stored sentence containing "is" or "the", and unrelated memories would be injected into the prompt.
STOPWORDS = frozenset(
    "a an and are as at be been but by can could did do does for from had has have he her his how i if in into is it its "
    "me my no not of on or our she should so than that the their them then there these they this to was we were what "
    "when where which who why will with would you your about one".split()
)


def tokenize(text: str) -> list[str]:
    """Lower-cased content words plus CJK characters and character bigrams (no word segmenter needed)."""
    lowered = text.lower()
    tokens = [word for word in _WORD.findall(lowered) if word not in STOPWORDS]
    cjk = _CJK.findall(lowered)
    tokens += cjk
    tokens += [a + b for a, b in zip(cjk, cjk[1:])]
    return tokens


def bm25_scores(query: str, documents: list[str], *, k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Return one BM25 score per document (0.0 when no query term matches)."""
    if not documents:
        return []
    tokenized = [tokenize(doc) for doc in documents]
    query_terms = set(tokenize(query))
    lengths = [len(t) for t in tokenized]
    average = (sum(lengths) / len(lengths)) or 1.0
    document_frequency: Counter[str] = Counter()
    for tokens in tokenized:
        document_frequency.update(set(tokens) & query_terms)
    n = len(documents)
    scores = []
    for tokens, length in zip(tokenized, lengths):
        counts = Counter(tokens)
        score = 0.0
        for term in query_terms:
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            idf = math.log(1 + (n - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
            score += idf * frequency * (k1 + 1) / (frequency + k1 * (1 - b + b * length / average))
        scores.append(score)
    return scores
