import re

CHUNK_MIN = 60
CHUNK_MAX = 110

# Splits after sentence-ending punctuation followed by whitespace + uppercase letter (German-aware)
_SENTENCE_END = re.compile(r'(?<=[.!?…])\s+(?=[A-ZÄÖÜ\"\'\(])')


def count_words(text: str) -> int:
    return len(text.split())


def split_sentences(text: str) -> list:
    """Split German text into sentences using punctuation-based heuristic."""
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


def chunk_text(text: str, min_words: int = CHUNK_MIN, max_words: int = CHUNK_MAX) -> list:
    """
    Split text into sentence-based chunks between min_words and max_words.
    Sentences that alone exceed max_words are skipped.
    Returns list of chunk strings.
    """
    sentences = split_sentences(text)
    chunks = []
    current = []
    current_wc = 0

    for sent in sentences:
        wc = count_words(sent)
        if wc > max_words:
            # Single sentence too long — flush current if valid, skip this sentence
            if current_wc >= min_words:
                chunks.append(" ".join(current))
            current, current_wc = [], 0
            continue

        if current_wc + wc > max_words:
            # Adding this sentence would exceed limit — flush current if valid
            if current_wc >= min_words:
                chunks.append(" ".join(current))
            current, current_wc = [sent], wc
        else:
            current.append(sent)
            current_wc += wc

    if current_wc >= min_words:
        chunks.append(" ".join(current))

    return chunks
