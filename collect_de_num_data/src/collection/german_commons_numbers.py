"""
Extract ALL clean, high-quality High German sentences containing digits/numeric 
formats from coral-nlp/german-commons (unbounded execution).

Output format (JSONL):
  {"text": "...", "numbers": [{"value": "...", "type": "..."}], "source": "..."}
"""

import json
import re
import sys
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.pipeline_runtime import configure_job_environment, require_compute_node, storage_path

configure_job_environment()

try:
    from .text_utils import split_sentences
except ImportError:  # Supports direct execution via the collection launcher.
    from text_utils import split_sentences

# ── Configuration ────────────────────────────────────────────────────────────
OUTPUT_FILE = storage_path("raw", "german_commons_numbers.jsonl")

# Target high-quality sources (skipping unfiltered raw web)
PRIORITY_CONFIGS = ["wikipedia", "news", "books", "legal"]
CANDIDATE_SPLITS = ["train", "wikipedia", "news", "web"]

# Strict document-level quality thresholds
MIN_OCR_SCORE = 98.0
MIN_PERPLEXITY = 20.0
MAX_PERPLEXITY = 120.0

# Strict sentence geometry & density rules
MIN_WORDS = 10
MAX_WORDS = 45
MAX_NUM_DENSITY = 0.15
MIN_ALPHA_WORDS = 8

BOILERPLATE_TERMS = {
    "cookie", "datenschutz", "akzeptieren", "impressum", "agb", 
    "newsletter", "warenkorb", "anmelden", "registrieren", "copyright",
    "all rights reserved", "weiterlesen", "klicken sie", "datenschutzerklärung"
}

# ── Standard High German Numeric Patterns (Digits Only) ─────────────────────
_MONTHS = (
    r"Januar|Februar|März|April|Mai|Juni|Juli|August|"
    r"September|Oktober|November|Dezember"
)

_NUM_PATTERNS: list[tuple[str, re.Pattern]] = [
    # Full dates with month name: "12. März 2024" or "März 2024"
    ("DATUM", re.compile(
        rf"\b\d{{1,2}}\.\s*(?:{_MONTHS})\s+\d{{2,4}}\b"
        rf"|\b(?:{_MONTHS})\s+\d{{2,4}}\b",
        re.IGNORECASE,
    )),
    # Numeric dates: 12.03.2024 / 1.1.24
    ("DATUM", re.compile(r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b")),
    # Ordinals: Digit followed by dot, then space + lowercase letter (e.g. "3. Platz")
    ("ORDINAL", re.compile(r"\b\d{1,3}\.(?=\s+[a-zäöüß])")),
    # Percentages: "45 %" / "3,5%"
    ("PROZENT", re.compile(r"\b\d+(?:[,.]\d+)?\s*%")),
    # Standard German Phone numbers (+49 30 123456 or 030/123456)
    ("TELEFON", re.compile(
        r"(?<!\w)(?:\+49[\s\-/]?)?\(?\d{2,5}\)?[\s\-/]\d{3,8}(?:[\s\-/]\d{2,5})*(?!\w)"
    )),
    # Standard High German Decimal (comma only: 3,14 or 1.234,56)
    ("DEZIMAL", re.compile(r"\b\d{1,3}(?:\.\d{3})*,\d+\b")),
    # 4-digit years (1000–2099)
    ("JAHR", re.compile(r"\b(?:1\d{3}|20\d{2})\b")),
    # Catch-all integers / plain digits (1000, 10.000, 42)
    ("KARDINAL", re.compile(r"\b\d+(?:\.\d{3})*\b")),
]

_HTML = re.compile(r"<[^>]+>|&[a-z]{2,6};", re.IGNORECASE)
_LONG_DIGIT = re.compile(r"\d{9,}")
_URL_PATTERN = re.compile(r"https?://|www\.", re.IGNORECASE)


# ── Extraction & Strict Quality Logic ────────────────────────────────────────

def extract_numbers(text: str) -> list[dict]:
    """Find non-overlapping digit spans in text."""
    found: list[dict] = []
    covered: list[tuple[int, int]] = []

    for num_type, pattern in _NUM_PATTERNS:
        for m in pattern.finditer(text):
            s, e = m.start(), m.end()
            if any(cs <= s < ce or cs < e <= ce for cs, ce in covered):
                continue
            covered.append((s, e))
            found.append({
                "number": m.group().strip(),
                "value": m.group().strip(),
                "type": num_type,
                "start": s,
                "end": e,
            })

    return found


def is_high_quality_sentence(sent: str, numbers: list[dict]) -> bool:
    """Hardened quality gate for pristine High German text."""
    if not numbers:
        return False

    sent = sent.strip()
    
    # 1. Clean capitalization and terminal punctuation
    if not sent[0].isupper() or sent[-1] not in ".!?":
        return False

    # Balanced quotes & brackets
    if sent.count('"') % 2 != 0 or sent.count('(') != sent.count(')'):
        return False

    # Disallow punctuation clustering (e.g. '...', '!!', ',-')
    if re.search(r'[,.\-:;!?/]{2,}', sent):
        return False

    words = sent.split()
    n_words = len(words)

    # Tightened word bounds
    if n_words < MIN_WORDS or n_words > MAX_WORDS:
        return False

    # 2. Character purity (>=88% valid High German characters)
    de_chars = len(re.findall(r'[a-zA-ZäöüÄÖÜß\s,.\-!?]', sent))
    if de_chars / len(sent) < 0.88:
        return False

    # 3. Reject web artifacts & long serial numbers
    if _URL_PATTERN.search(sent) or _HTML.search(sent) or _LONG_DIGIT.search(sent):
        return False

    # 4. Boilerplate filter
    sent_lower = sent.lower()
    if any(term in sent_lower for term in BOILERPLATE_TERMS):
        return False

    # 5. ALL-CAPS threshold (<=10% capital tokens)
    upper_tokens = sum(1 for w in words if w.isupper() and len(w) > 1)
    if upper_tokens / n_words > 0.10:
        return False

    # 6. Natural digit density (<=15% of tokens contain digits)
    digit_tokens = sum(1 for w in words if re.search(r"\d", w))
    if digit_tokens / n_words > MAX_NUM_DENSITY:
        return False

    # 7. Requires genuine alphabetic German words
    alpha_words = sum(1 for w in words if re.fullmatch(r"[A-Za-zÄÖÜäöüß]{2,}", w))
    if alpha_words < MIN_ALPHA_WORDS:
        return False

    return True


def passes_document_quality(doc: dict) -> bool:
    """Filter bad documents using strict dataset metadata."""
    ocr = doc.get("ocr_score") or doc.get("quality_score")
    if ocr is not None and float(ocr) < MIN_OCR_SCORE:
        return False

    perp = doc.get("perplexity") or doc.get("perplexity_score")
    if perp is not None:
        p_val = float(perp)
        if p_val < MIN_PERPLEXITY or p_val > MAX_PERPLEXITY:
            return False

    return True


# ── Main Execution Loop (Unbounded) ──────────────────────────────────────────

def main():
    require_compute_node("German number corpus collection")
    from datasets import get_dataset_config_names, load_dataset

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        available_configs = get_dataset_config_names("coral-nlp/german-commons")
        configs = [c for c in PRIORITY_CONFIGS if c in available_configs]
        configs += [c for c in available_configs if c not in configs]
    except Exception as exc:
        print(f"Could not fetch configs automatically ({exc}), using fallback list.")
        configs = PRIORITY_CONFIGS

    total_records = 0
    type_counts = Counter()
    seen_sentences = set()

    temporary = OUTPUT_FILE.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as fh:
        for cfg in configs:
            for split in CANDIDATE_SPLITS:
                tag = f"german-commons/{cfg}/{split}"
                print(f"\n--- Streaming dataset: {tag} ---")

                try:
                    ds = load_dataset("coral-nlp/german-commons", cfg, split=split, streaming=True)
                except Exception as exc:
                    print(f"Skipping {tag}: {exc}")
                    continue

                cfg_yielded = 0
                for doc in ds:
                    if not passes_document_quality(doc):
                        continue

                    text = (doc.get("text") or "").strip()
                    if not text:
                        continue

                    for sent in split_sentences(text):
                        sent_clean = sent.strip()

                        if sent_clean in seen_sentences:
                            continue

                        nums = extract_numbers(sent_clean)

                        if is_high_quality_sentence(sent_clean, nums):
                            record = {
                                "text": sent_clean,
                                "numbers": nums,
                                "source": tag
                            }

                            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                            seen_sentences.add(sent_clean)

                            for n in nums:
                                type_counts[n["type"]] += 1

                            cfg_yielded += 1
                            total_records += 1

                            if total_records % 25_000 == 0:
                                print(f"Progress: {total_records:,} clean sentences written...")

                print(f"Finished {tag} -> Pushed {cfg_yielded:,} sentences")

    temporary.replace(OUTPUT_FILE)

    print("\n==================================================")
    print(f"EXTRACTION COMPLETE: Saved {total_records:,} total sentences to:\n  {OUTPUT_FILE}")
    print("==================================================")
    print("\nFinal Numeric Type Distribution:")
    for t, count in type_counts.most_common():
        print(f"  {t:<12}: {count:,}")


if __name__ == "__main__":
    main()
