"""Phase 2: Remove junk, spam, and non-color entries using categorized blacklists."""

from pathlib import Path

import pandas as pd

BLACKLISTS_DIR = Path(__file__).resolve().parent.parent / "blacklists"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def _load_wordlist(filename: str) -> set[str]:
    """Load a blacklist .txt file as a set of lowercase strings (ignoring comments)."""
    path = BLACKLISTS_DIR / filename
    if not path.exists():
        return set()
    with open(path) as f:
        return {
            line.strip().lower()
            for line in f
            if line.strip() and not line.startswith("#")
        }


def _has_toxic_word(name: str, toxic_words: set[str]) -> bool:
    """Return True if any word in the name is a toxic/offensive word."""
    return bool(set(name.split()) & toxic_words)


def _strip_modifiers(name: str, modifiers: set[str]) -> str:
    """Strip junk modifier words from the beginning and end of a name.

    E.g. 'ugly green' → 'green', 'green again' → 'green',
         'yet another blue' → 'blue'.
    """
    words = name.split()

    # Strip from left: junk modifiers + filler words
    strip_left = modifiers | {"yet", "another", "really", "so", "still", "not", "its", "just", "some", "a"}
    while words and words[0] in strip_left:
        words = words[1:]

    # Strip from right: "again" and junk modifiers (for compound-split leftovers)
    strip_right = modifiers | {"again"}
    while words and words[-1] in strip_right:
        words = words[:-1]

    return " ".join(words) if words else ""


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove junk entries and strip noise modifiers.

    Loads categorized blacklists from blacklists/ folder:
      toxic_words.txt    — offensive words (word-level match → remove entire row)
      junk_modifiers.txt — subjective modifiers (strip from names)
      junk_phrases.txt   — exact non-color phrases (remove entire row)
      junk_standalone.txt — non-color words (remove if standalone)
      spam_gibberish.txt — keyboard spam, names, gibberish (exact match → remove)

    Returns cleaned DataFrame.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = df[df["colorname"].notna()].copy()
    before = len(df)
    before_unique = df["colorname"].nunique()

    toxic_words = _load_wordlist("toxic_words.txt")
    modifiers = _load_wordlist("junk_modifiers.txt")
    phrases = _load_wordlist("junk_phrases.txt")
    standalone = _load_wordlist("junk_standalone.txt")
    spam = _load_wordlist("spam_gibberish.txt")

    # 1. Spam/gibberish removal (exact match)
    mask_spam = df["colorname"].isin(spam)
    print(f"  Spam/gibberish: {mask_spam.sum():,} rows")

    # 2. Toxic word removal (any word in name matches)
    mask_toxic = df["colorname"].apply(lambda n: _has_toxic_word(n, toxic_words)) & ~mask_spam
    print(f"  Toxic words: {mask_toxic.sum():,} rows")

    # 3. Junk phrase removal (exact match)
    mask_phrase = df["colorname"].isin(phrases) & ~mask_spam & ~mask_toxic
    print(f"  Junk phrases: {mask_phrase.sum():,} rows")

    df = df[~(mask_spam | mask_toxic | mask_phrase)].copy()

    # 4. Strip modifier prefixes/suffixes
    df["colorname"] = df["colorname"].apply(lambda n: _strip_modifiers(n, modifiers))

    # 5. Drop empty, standalone-modifier, or standalone-junk names
    drop = modifiers | standalone
    mask_empty = (df["colorname"] == "") | df["colorname"].isin(drop)
    print(f"  Empty/junk after stripping: {mask_empty.sum():,} rows")
    df = df[~mask_empty].copy()

    print(f"  After cleaning: {len(df):,} rows, {df['colorname'].nunique():,} unique names")
    print(f"  Removed: {before - len(df):,} rows, {before_unique - df['colorname'].nunique():,} unique names")
    return df
