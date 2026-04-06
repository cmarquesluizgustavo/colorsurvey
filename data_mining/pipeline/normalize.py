"""Phase 3: Word-level normalization — compound splitting, fuzzy matching, reconstruction."""

from pathlib import Path
from collections import Counter

import pandas as pd
from rapidfuzz import fuzz, process

from .rules import (
    NEVER_CORRECT, NOSPLIT_SUFFIXES, NOSPLIT_WORDS,
    MERGE_SUFFIXES, DETERMINISTIC_NAME_MAP, DETERMINISTIC_WORD_MAP,
)

# ---------------------------------------------------------------------------
# Paths & configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = DATA_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Output file paths
VOCAB_PATH = OUTPUT_DIR / "word_vocabulary.txt"
FUZZY_WORD_REVIEW_PATH = OUTPUT_DIR / "fuzzy_word_review.csv"
FUZZY_NAME_REVIEW_PATH = OUTPUT_DIR / "fuzzy_name_review.csv"
TOKEN_MATCHES_PATH = OUTPUT_DIR / "all_token_matches.csv"
COLOR_FREQ_PATH = OUTPUT_DIR / "color_frequencies.csv"
WORD_FREQ_PATH = OUTPUT_DIR / "word_frequencies.csv"

# Seed vocabulary files (generic names — swap files without renaming code)
SEED_CORE = DATA_DIR / "fuzzy_matching" / "top_100_colors.txt"
SEED_EXTENDED = DATA_DIR / "fuzzy_matching" / "top_1250_colors.txt"

# Pipeline parameters
MIN_COLOR_FREQ = 10      # drop colors appearing fewer than this many times
DEDUP_WORDS = True       # remove duplicate words in a color name
SORT_WORDS = True        # alphabetically order words in a color name

# Fuzzy matching thresholds
AUTO_ACCEPT_SCORE = 90   # score >= this: auto-correct
REVIEW_SCORE = 80        # score >= this but < AUTO_ACCEPT: propose for review
MIN_FUZZY_LEN = 5        # skip fuzzy matching for words shorter than this


def _load_seed_names(path: Path) -> list[str]:
    """Load one-name-per-line file, strip and lowercase."""
    if not path.exists():
        return []
    with open(path) as f:
        return [line.strip().lower() for line in f if line.strip()]


def _split_words(name: str) -> list[str]:
    """Split a color name into words (by whitespace)."""
    return name.split()


def _deterministic_replace(words: list[str]) -> list[str]:
    """Apply deterministic word-level substitutions (e.g. grey → gray)."""
    return [DETERMINISTIC_WORD_MAP.get(w, w) for w in words]


# ---------------------------------------------------------------------------
# Vocabulary building
# ---------------------------------------------------------------------------

def build_vocabulary(names: list[str], min_word_freq: int = 3) -> set[str]:
    """Build word vocabulary from known-good color names.

    Strategy:
    1. Seed from core colors (all words)
    2. Expand with words from the extended list that appear >= min_word_freq times
    3. Words below threshold get fuzzy-matched or added as new

    Returns the vocabulary set.
    """
    vocab: set[str] = set()

    # Round 1: seed from core colors
    core_names = _load_seed_names(SEED_CORE)
    for name in core_names:
        vocab.update(_split_words(name))
    # Apply deterministic fixes to the vocab itself
    vocab = {DETERMINISTIC_WORD_MAP.get(w, w) for w in vocab}
    print(f"  Vocabulary after core-seed: {len(vocab)} words")

    # Round 2: expand from extended list
    extended_names = _load_seed_names(SEED_EXTENDED)
    word_counter: Counter[str] = Counter()
    for name in extended_names:
        words = _deterministic_replace(_split_words(name))
        word_counter.update(words)

    # Frequent words are trusted
    for word, count in word_counter.items():
        if count >= min_word_freq:
            vocab.add(word)
    print(f"  Vocabulary after extended-expansion (freq >= {min_word_freq}): {len(vocab)} words")

    # Rare words: fuzzy match against vocab or add as new
    rare_words = {w for w, c in word_counter.items() if c < min_word_freq and w not in vocab}
    fuzzy_additions: list[dict] = []
    for word in sorted(rare_words):
        if word in NEVER_CORRECT:
            vocab.add(word)
            continue

        if len(word) < MIN_FUZZY_LEN:
            # Too short to fuzzy match safely — add as-is
            vocab.add(word)
            continue

        match = process.extractOne(word, vocab, scorer=fuzz.ratio)
        if match is None:
            vocab.add(word)
            continue

        best_match, score, _ = match
        if score >= AUTO_ACCEPT_SCORE:
            # Clear typo — don't add the misspelling to vocab
            fuzzy_additions.append({
                "word": word, "correction": best_match,
                "score": score, "action": "auto",
            })
        elif score >= REVIEW_SCORE:
            fuzzy_additions.append({
                "word": word, "correction": best_match,
                "score": score, "action": "review",
            })
            vocab.add(word)  # keep until reviewed
        else:
            vocab.add(word)  # genuinely new word

    print(f"  Final vocabulary: {len(vocab)} words")
    if fuzzy_additions:
        review_df = pd.DataFrame(fuzzy_additions).sort_values("score", ascending=False)
        review_df.to_csv(FUZZY_WORD_REVIEW_PATH, index=False)
        auto = sum(1 for r in fuzzy_additions if r["action"] == "auto")
        review = sum(1 for r in fuzzy_additions if r["action"] == "review")
        print(f"  Fuzzy word corrections: {auto} auto-accepted, {review} for review")
        print(f"  Saved {FUZZY_WORD_REVIEW_PATH}")

    return vocab


# ---------------------------------------------------------------------------
# Compound splitting
# ---------------------------------------------------------------------------


def _try_split(word: str, vocab: set[str]) -> list[str] | None:
    """Try splitting a single token into two known vocabulary words.

    Only splits when BOTH parts are >= 3 chars and in the vocabulary.
    Returns None if no valid split or if the word is in the no-split list.
    """
    if word in NOSPLIT_WORDS:
        return None
    # Don't split words that ARE a known word + suffix (greenish, bluey, etc.)
    # But only block if the base (without suffix) is itself in vocab
    for suffix in NOSPLIT_SUFFIXES:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            base = word[:-len(suffix)]
            base_norm = DETERMINISTIC_WORD_MAP.get(base, base)
            if base_norm in vocab:
                return None

    candidates: list[tuple[str, str]] = []
    for i in range(3, len(word) - 2):  # both parts must be >= 3 chars
        left, right = word[:i], word[i:]
        # Also accept parts that have a deterministic replacement in vocab
        left_norm = DETERMINISTIC_WORD_MAP.get(left, left)
        right_norm = DETERMINISTIC_WORD_MAP.get(right, right)
        if left_norm in vocab and right_norm in vocab:
            candidates.append((left_norm, right_norm))

    if not candidates:
        return None
    # Prefer most balanced split (longest shortest part)
    best = max(candidates, key=lambda pair: min(len(pair[0]), len(pair[1])))
    return list(best)


def split_compounds(names: list[str], vocab: set[str]) -> dict[str, str]:
    """Find single-token names that can be split into known vocabulary words.

    Only attempts splitting on names that are a single word (no spaces).
    Returns dict mapping original → split form (e.g. "bluegreen" → "blue green").
    """
    splits: dict[str, str] = {}
    for name in set(names):
        # Only split single-token names — multi-word names stay as-is
        if " " in name:
            continue
        parts = _try_split(name, vocab)
        if parts:
            splits[name] = " ".join(parts)
    return splits


# ---------------------------------------------------------------------------
# Word-level fuzzy correction for full dataset
# ---------------------------------------------------------------------------

def _build_word_corrections(vocab: set[str]) -> dict[str, str]:
    """Build corrections from fuzzy_word_review.csv (auto-accepted entries)."""
    if not FUZZY_WORD_REVIEW_PATH.exists():
        return {}
    df = pd.read_csv(FUZZY_WORD_REVIEW_PATH)
    # Only apply auto-accepted corrections
    auto = df[df["action"] == "auto"]
    return dict(zip(auto["word"], auto["correction"]))



def normalize_name(name: str, vocab: set[str], word_corrections: dict[str, str],
                   compound_map: dict[str, str]) -> str:
    """Normalize a single color name:
    1. Deterministic word replacements (grey → gray, typos)
    2. Name-level deterministic replacements (aqua marine → aquamarine)
    3. Compound splitting (bluegreen → blue green)
    4. Word-level fuzzy corrections from review file
    5. Fuzzy-match remaining unknown words against vocab
    """
    # Step 1: deterministic word replacements
    words = _deterministic_replace(_split_words(name))
    name = " ".join(words)

    # Step 2: name-level deterministic replacements
    # Check if any key appears as a substring and replace it
    for old, new in DETERMINISTIC_NAME_MAP.items():
        if old in name:
            name = name.replace(old, new)

    # Step 3: compound splitting
    if name in compound_map:
        name = compound_map[name]
    words = name.split()

    # Step 3+4: word corrections
    corrected: list[str] = []
    for w in words:
        if w in NEVER_CORRECT:
            corrected.append(w)
        elif w in word_corrections:
            corrected.append(word_corrections[w])
        elif w in vocab:
            corrected.append(w)
        elif len(w) >= MIN_FUZZY_LEN:
            match = process.extractOne(w, vocab, scorer=fuzz.ratio)
            if match and match[1] >= AUTO_ACCEPT_SCORE:
                corrected.append(match[0])
            else:
                corrected.append(w)
        else:
            corrected.append(w)

    # Step 5: Rejoin separated suffixes ("green ish" → "greenish", "orange y" → "orangey")
    merged: list[str] = []
    i = 0
    while i < len(corrected):
        if i > 0 and corrected[i] in MERGE_SUFFIXES:
            merged[-1] = merged[-1] + corrected[i]
        else:
            merged.append(corrected[i])
        i += 1
    corrected = merged

    return " ".join(corrected)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Run full normalization pipeline on the DataFrame.

    Generates review files in output/ and returns the normalized DataFrame.
    """
    all_names = df["colorname"].tolist()

    print("\n--- Building word vocabulary ---")
    vocab = build_vocabulary(all_names)

    # Save vocabulary
    with open(VOCAB_PATH, "w") as f:
        for w in sorted(vocab):
            f.write(w + "\n")
    print(f"  Saved {VOCAB_PATH}")

    print("\n--- Splitting compounds ---")
    compound_map = split_compounds(all_names, vocab)
    print(f"  Found {len(compound_map)} compound names to split")
    if compound_map:
        examples = list(compound_map.items())[:10]
        for orig, split in examples:
            print(f"    {orig} → {split}")

    print("\n--- Normalizing all names ---")
    word_corrections = _build_word_corrections(vocab)
    print(f"  Word corrections loaded: {len(word_corrections)}")

    df = df.copy()
    df["colorname_original"] = df["colorname"]
    df["colorname"] = df["colorname"].apply(
        lambda n: normalize_name(n, vocab, word_corrections, compound_map)
    )

    # Generate name-level review
    changed = df[df["colorname"] != df["colorname_original"]].copy()
    if not changed.empty:
        review = (
            changed.groupby(["colorname_original", "colorname"])
            .size()
            .reset_index(name="frequency")
        )
        review.columns = ["original", "normalized", "frequency"]
        review = review.sort_values("frequency", ascending=False)
        review.to_csv(FUZZY_NAME_REVIEW_PATH, index=False)
        print(f"  Names changed: {len(review)} unique mappings ({changed.shape[0]:,} rows)")
        print(f"  Saved {FUZZY_NAME_REVIEW_PATH}")
    else:
        print("  No names changed.")

    before_unique = df["colorname_original"].nunique()
    after_unique = df["colorname"].nunique()
    print(f"\n  Unique names: {before_unique:,} → {after_unique:,} ({before_unique - after_unique:,} merged)")
    print(f"  Total rows preserved: {len(df):,}")

    # --- Deduplicate words in each color name ---
    if DEDUP_WORDS:
        before = df["colorname"].nunique()
        df["colorname"] = df["colorname"].apply(
            lambda n: " ".join(dict.fromkeys(n.split()))
        )
        after = df["colorname"].nunique()
        print(f"\n  Dedup words: {before:,} → {after:,} unique ({before - after:,} merged)")

    # --- Alphabetically sort words in each color name ---
    if SORT_WORDS:
        before = df["colorname"].nunique()
        df["colorname"] = df["colorname"].apply(
            lambda n: " ".join(sorted(n.split()))
        )
        after = df["colorname"].nunique()
        print(f"  Sort words: {before:,} → {after:,} unique ({before - after:,} merged)")

    # --- Frequency cutoff: drop colors with < MIN_COLOR_FREQ occurrences ---
    freq = df["colorname"].value_counts()
    rare_names = set(freq[freq < MIN_COLOR_FREQ].index)
    mask_rare = df["colorname"].isin(rare_names)
    df = df[~mask_rare].copy()
    after_cutoff = df["colorname"].nunique()
    print(f"\n  Frequency cutoff (< {MIN_COLOR_FREQ}): dropped {len(rare_names)} rare colors, "
          f"{mask_rare.sum():,} rows")
    print(f"  Final: {len(df):,} rows, {after_cutoff} unique colors")

    # --- Generate analysis outputs (after cutoff) ---
    _save_token_matches(df, compound_map, word_corrections)
    _save_color_frequencies(df)
    _save_word_frequencies(df)

    return df


def _save_token_matches(df: pd.DataFrame, compound_map: dict[str, str],
                        word_corrections: dict[str, str]) -> None:
    """Save all token-level transformations with row counts."""
    changed = df[df["colorname"] != df["colorname_original"]]
    if changed.empty:
        return

    records: list[dict] = []
    for (orig, norm), group in changed.groupby(["colorname_original", "colorname"]):
        rows_affected = len(group)
        orig_words = orig.split()
        norm_words = norm.split()
        # Identify which tokens changed
        changes = []
        if orig in compound_map:
            changes.append(f"split: {orig} → {compound_map[orig]}")
        for ow, nw in zip(orig_words, norm_words):
            if ow != nw:
                changes.append(f"{ow} → {nw}")
        if len(orig_words) != len(norm_words) and orig not in compound_map:
            changes.append(f"[word count: {len(orig_words)} → {len(norm_words)}]")
        records.append({
            "original": orig,
            "normalized": norm,
            "token_changes": "; ".join(changes) if changes else "compound split",
            "rows_affected": rows_affected,
        })

    out = pd.DataFrame(records).sort_values("rows_affected", ascending=False)
    out.to_csv(TOKEN_MATCHES_PATH, index=False)
    print(f"  Saved {TOKEN_MATCHES_PATH} ({len(out)} transformations)")


def _save_color_frequencies(df: pd.DataFrame) -> None:
    """Save final color name frequencies."""
    freq = df["colorname"].value_counts().reset_index()
    freq.columns = ["colorname", "frequency"]
    freq.to_csv(COLOR_FREQ_PATH, index=False)
    print(f"  Saved {COLOR_FREQ_PATH} ({len(freq)} unique colors)")


def _save_word_frequencies(df: pd.DataFrame) -> None:
    """Save word-level token frequencies across all color names."""
    word_counts: Counter = Counter()
    for name in df["colorname"]:
        word_counts.update(name.split())
    freq = pd.DataFrame(word_counts.most_common(), columns=["word", "frequency"])
    freq.to_csv(WORD_FREQ_PATH, index=False)
    print(f"  Saved {WORD_FREQ_PATH} ({len(freq)} unique words)")


def apply_normalizations(df: pd.DataFrame) -> pd.DataFrame:
    """Apply reviewed normalizations and export the final CSV.

    Reads fuzzy_word_review.csv (user may have edited 'action' column)
    and re-runs normalization with the updated corrections.
    """
    # Re-run normalize (it reads the review file)
    df = normalize(df)

    # Drop the helper column for export
    if "colorname_original" in df.columns:
        df = df.drop(columns=["colorname_original"])

    out_path = OUTPUT_DIR / "cleaned_colors.csv"
    df.to_csv(out_path, index=False)
    print(f"\nExported {len(df):,} rows to {out_path}")

    return df
