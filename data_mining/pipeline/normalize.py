"""Phase 3: Word-level normalization — compound splitting, fuzzy matching, reconstruction."""

import re
from pathlib import Path
from collections import Counter

import pandas as pd
from rapidfuzz import fuzz, process

DATA_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = DATA_DIR / "output"

# Paths to seed files (kept from original data_mining)
SEED_28 = DATA_DIR / "28_color_normalization.txt"
SEED_1500 = DATA_DIR / "1500_weird_color_normalization.txt"

# Deterministic replacements applied before anything else
_DETERMINISTIC_WORD_MAP = {
    # British → American
    "grey": "gray",
    "greyish": "grayish",
    "greyed": "grayed",
    "greyey": "grayish",
    "greay": "gray",
    "gery": "gray",
    "greyn": "green",  # typo, not gray

    # Green typos
    "gree": "green",
    "gren": "green",
    "greem": "green",
    "grenn": "green",
    "geen": "green",
    "greeb": "green",
    "grene": "green",
    "groen": "green",
    "grean": "green",
    "grren": "green",
    "geren": "green",
    "gween": "green",
    "greeeeen": "green",
    "greeeen": "green",
    "greeeeeeen": "green",
    "greeeeeen": "green",
    "greeeeeeeen": "green",

    # Purple typos
    "pruple": "purple",
    "pueple": "purple",
    "purpel": "purple",
    "purplr": "purple",
    "purble": "purple",
    "puprle": "purple",
    "poiple": "purple",
    "purlpe": "purple",
    "purplw": "purple",
    "putple": "purple",
    "pirple": "purple",
    "murple": "purple",
    "purpur": "purple",
    "purpke": "purple",
    "purpil": "purple",
    "purpul": "purple",
    "grurple": "purple",
    "porple": "purple",
    "ourple": "purple",
    "purkle": "purple",

    # Chartreuse typos
    "charteuse": "chartreuse",
    "chartruese": "chartreuse",
    "chartruce": "chartreuse",
    "chatreuse": "chartreuse",
    "chartreus": "chartreuse",
    "chatruse": "chartreuse",
    "shartruse": "chartreuse",

    # Fuchsia typos
    "fuschia": "fuchsia",
    "fucia": "fuchsia",
    "fusha": "fuchsia",
    "fuchisa": "fuchsia",
    "fucha": "fuchsia",

    # Mauve typos
    "muave": "mauve",
    "mouve": "mauve",
    "moave": "mauve",
    "mauv": "mauve",
    "maube": "mauve",
    "mauce": "mauve",
    "malve": "mauve",

    # Raspberry typos
    "rasberry": "raspberry",

    # Terracotta typos
    "terracota": "terracotta",
    "teracotta": "terracotta",

    # Khaki typos
    "kaki": "khaki",
    "kahki": "khaki",
    "kakhi": "khaki",
    "karki": "khaki",
    "kacki": "khaki",

    # Taupe typos
    "tope": "taupe",
    "toupe": "taupe",
    "taup": "taupe",
    "toap": "taupe",
    "toape": "taupe",

    # Ochre typos
    "ocre": "ochre",
    "ocker": "ochre",
    "ocra": "ochre",
    "oker": "ochre",
    "ochra": "ochre",
    "ockre": "ochre",

    # Lilac typos
    "lila": "lilac",
    "lilla": "lilac",
    "lylac": "lilac",
    "lilas": "lilac",
    "lilic": "lilac",
    "lilca": "lilac",
    "lylic": "lilac",
    "lilav": "lilac",

    # Seafoam typos
    "seafom": "seafoam",
    "seaform": "seafoam",

    # Blue typos
    "bluue": "blue",
    "bluw": "blue",

    # Avocado typos
    "avacado": "avocado",

    # Beige typos
    "beig": "beige",
    "baige": "beige",
    "biege": "beige",

    # Fuchsia typos (additional)
    "fuscia": "fuchsia",
    "fushia": "fuchsia",
    "fucshia": "fuchsia",
    "fuscha": "fuchsia",

    # Violet typos
    "violot": "violet",
    "voilet": "violet",
    "viloet": "violet",

    # Brown typos
    "brow": "brown",

    # Burgundy typos
    "burgendy": "burgundy",

    # Cream typos
    "creme": "cream",

    # Lavender typos
    "lavendar": "lavender",

    # Magenta typos
    "magneta": "magenta",
    "majenta": "magenta",

    # Puce typos
    "peuce": "puce",
    "puse": "puce",

    # Reddish typos
    "redish": "reddish",

    # Turquoise typos (additional)
    "terquoise": "turquoise",
    "torquise": "turquoise",
    "turqois": "turquoise",
    "turqiose": "turquoise",
    "tourquise": "turquoise",
    "turcoise": "turquoise",
    "tourqoise": "turquoise",
    "turqouis": "turquoise",
    "turqoiuse": "turquoise",
    "turquiose": "turquoise",
    "turqouse": "turquoise",
    "turquase": "turquoise",
    "turquese": "turquoise",
    "turquios": "turquoise",
    "turqiouse": "turquoise",
    "torquiose": "turquoise",
    "torqouise": "turquoise",
    "torqoise": "turquoise",
    "tirquoise": "turquoise",
    "turquis": "turquoise",
    "turqouise": "turquoise",

    # Blue typos (additional)
    "bllue": "blue",
    "blule": "blue",

    # Pink typos
    "pikn": "pink",

    # Orange typos
    "ornage": "orange",
    "oragne": "orange",
    "organe": "orange",
    "oraneg": "orange",
    "orance": "orange",
    "orenge": "orange",
    "oarnge": "orange",

    # Brown typos (additional)
    "bronw": "brown",
    "brwon": "brown",
    "borwn": "brown",
    "bown": "brown",
    "broen": "brown",
    "brwn": "brown",

    # Yellow typos
    "yelloe": "yellow",
    "yellpw": "yellow",
    "yeloow": "yellow",

    # Beige typos (additional)
    "bage": "beige",
    "bege": "beige",
    "beigh": "beige",
    "baise": "beige",
    "bez": "beige",

    # Fuchsia typos (more)
    "fusca": "fuchsia",

    # Cyan typos
    "cyab": "cyan",
    "cya": "cyan",

    # Aqua typos
    "aque": "aqua",
    "aqau": "aqua",
    "awua": "aqua",
    "auqa": "aqua",
    "auqua": "aqua",
    "auqamarine": "aquamarine",

    # Comparative modifiers → base (standalone they get stripped anyway)
    "greener": "green",
    "bluer": "blue",

    # Short fragments commonly mismatched by fuzzy
    "urple": "purple",
    "purpl": "purple",
    "marroon": "maroon",

    # Alternate spellings / additional typos
    "ocher": "ochre",
    "marron": "maroon",
    "blooo": "blue",
    "blud": "blue",
    "pinlk": "pink",
    "pinl": "pink",
    "turkiz": "turquoise",
    "terqoise": "turquoise",
    "forset": "forest",
    "oliv": "olive",
    "ligt": "light",
    "lght": "light",
    "bule": "blue",
    "bleu": "blue",
    "blu": "blue",
    "bue": "blue",
    "violte": "violet",
    "saumon": "salmon",
    "lavenda": "lavender",

    # Blue typos (more)
    "bloue": "blue",
    "bluje": "blue",
    "bluwe": "blue",

    # Magenta typos (more)
    "maginta": "magenta",

    # Cantaloupe / cerise / sherbet
    "cantelope": "cantaloupe",
    "cerice": "cerise",
    "sherbert": "sherbet",

    # Turquoise typos (more)
    "tourquiose": "turquoise",

    # Avocado typos (more)
    "avacodo": "avocado",

    # Pale / navy typos
    "plae": "pale",
    "navi": "navy",

    # Beige typos (more)
    "bauge": "beige",

    # Maroon alternate
    "marrone": "maroon",

    # Misc typos
    "breen": "green",
    "pinkle": "pink",
    "purblue": "purple",
    "limish": "lime",
    "birght": "bright",
    "lav": "lavender",
    "sku": "sky",
}

# Fuzzy matching thresholds
AUTO_ACCEPT_SCORE = 90   # score >= this: auto-correct
REVIEW_SCORE = 80        # score >= this but < AUTO_ACCEPT: propose for review
MIN_FUZZY_LEN = 5        # skip fuzzy matching for words shorter than this

# Words that should NEVER be fuzzy-corrected — they are valid as-is
_NEVER_CORRECT = {
    # Valid color words that look like typos of other words
    "blush", "violent", "manila", "saturated", "desaturated",
    # Valid -y adjective modifiers
    "creamy", "cloudy", "fleshy", "steely", "earthy", "stormy",
    "grassy", "watery", "swampy", "mossy", "leafy", "minty",
    "smokey", "dusty", "rusty", "muddy", "sandy", "milky",
    "chalky", "silky", "peachy", "rosy", "inky", "smoky",
    "foresty",
    # Other valid words
    "burple", "barney", "brass", "bronze", "leather", "custard",
    "caramel", "parrot", "creme", "slime", "hazel", "irish",
    "vanilla", "plumb",
    # Colors that look like typos of other words
    "goldenrod", "buttercup", "candy", "amber", "orangered",
    "night", "russet", "spruce", "fresh", "bubblegum", "limey",
    "sickly", "gross",
}


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
    return [_DETERMINISTIC_WORD_MAP.get(w, w) for w in words]


# ---------------------------------------------------------------------------
# Vocabulary building
# ---------------------------------------------------------------------------

def build_vocabulary(names: list[str], min_word_freq: int = 3) -> set[str]:
    """Build word vocabulary from known-good color names.

    Strategy:
    1. Seed from the 28 canonical colors (all words)
    2. Expand with words from the 1500 list that appear >= min_word_freq times
    3. Words below threshold get fuzzy-matched or added as new

    Returns the vocabulary set.
    """
    vocab: set[str] = set()

    # Round 1: seed from 28 canonical
    seed_28 = _load_seed_names(SEED_28)
    for name in seed_28:
        vocab.update(_split_words(name))
    # Apply deterministic fixes to the vocab itself
    vocab = {_DETERMINISTIC_WORD_MAP.get(w, w) for w in vocab}
    print(f"  Vocabulary after 28-seed: {len(vocab)} words")

    # Round 2: expand from 1500 list
    seed_1500 = _load_seed_names(SEED_1500)
    word_counter: Counter[str] = Counter()
    for name in seed_1500:
        words = _deterministic_replace(_split_words(name))
        word_counter.update(words)

    # Frequent words are trusted
    for word, count in word_counter.items():
        if count >= min_word_freq:
            vocab.add(word)
    print(f"  Vocabulary after 1500-expansion (freq >= {min_word_freq}): {len(vocab)} words")

    # Rare words: fuzzy match against vocab or add as new
    rare_words = {w for w, c in word_counter.items() if c < min_word_freq and w not in vocab}
    fuzzy_additions: list[dict] = []
    for word in sorted(rare_words):
        if word in _NEVER_CORRECT:
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
        review_path = OUTPUT_DIR / "fuzzy_word_review.csv"
        review_df.to_csv(review_path, index=False)
        auto = sum(1 for r in fuzzy_additions if r["action"] == "auto")
        review = sum(1 for r in fuzzy_additions if r["action"] == "review")
        print(f"  Fuzzy word corrections: {auto} auto-accepted, {review} for review")
        print(f"  Saved {review_path}")

    return vocab


# ---------------------------------------------------------------------------
# Compound splitting
# ---------------------------------------------------------------------------

# Common suffixes that produce valid color adjectives — never split these
_NOSPLIT_SUFFIXES = ("ish", "ey", "ly", "ry", "ty", "ny", "al", "er", "le")

# Real English words that happen to contain color substrings — never split
_NOSPLIT_WORDS = {
    "blueberry", "strawberry", "cranberry", "raspberry", "blackberry",
    "goldenrod", "eggplant", "eggshell", "champagne", "tangerine", "chocolate",
    "charcoal", "pumpkin", "pistachio", "aubergine", "vermillion",
    "chartreuse", "turquoise", "periwinkle", "cornflower",
    "office", "officer", "golden", "violet", "scarlet",
    "sandstone", "limestone", "brownstone", "cobblestone",
    "marigold", "shamrock", "evergreen", "aquamarine",
    "midnight", "watermelon", "bubblegum", "redwood", "lemongrass",
    "skintone", "fleshtone", "gunmetal",
}


def _try_split(word: str, vocab: set[str]) -> list[str] | None:
    """Try splitting a single token into two known vocabulary words.

    Only splits when BOTH parts are >= 3 chars and in the vocabulary.
    Returns None if no valid split or if the word is in the no-split list.
    """
    if word in _NOSPLIT_WORDS:
        return None
    # Don't split words that ARE a known word + suffix (greenish, bluey, etc.)
    # But only block if the base (without suffix) is itself in vocab
    for suffix in _NOSPLIT_SUFFIXES:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            base = word[:-len(suffix)]
            base_norm = _DETERMINISTIC_WORD_MAP.get(base, base)
            if base_norm in vocab:
                return None

    candidates: list[tuple[str, str]] = []
    for i in range(3, len(word) - 2):  # both parts must be >= 3 chars
        left, right = word[:i], word[i:]
        # Also accept parts that have a deterministic replacement in vocab
        left_norm = _DETERMINISTIC_WORD_MAP.get(left, left)
        right_norm = _DETERMINISTIC_WORD_MAP.get(right, right)
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
    review_path = OUTPUT_DIR / "fuzzy_word_review.csv"
    if not review_path.exists():
        return {}
    df = pd.read_csv(review_path)
    # Only apply auto-accepted corrections
    auto = df[df["action"] == "auto"]
    return dict(zip(auto["word"], auto["correction"]))


# Deterministic name-level replacements (multi-word → single word, etc.)
_DETERMINISTIC_NAME_MAP = {
    "aqua marine": "aquamarine",
    "sea foam": "seafoam",
    "gun metal": "gunmetal",
}


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
    for old, new in _DETERMINISTIC_NAME_MAP.items():
        if old in name:
            name = name.replace(old, new)

    # Step 3: compound splitting
    if name in compound_map:
        name = compound_map[name]
    words = name.split()

    # Step 3+4: word corrections
    corrected: list[str] = []
    for w in words:
        if w in _NEVER_CORRECT:
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
    _MERGE_SUFFIXES = {"ish", "y", "ey"}
    merged: list[str] = []
    i = 0
    while i < len(corrected):
        if i > 0 and corrected[i] in _MERGE_SUFFIXES:
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_names = df["colorname"].tolist()

    print("\n--- Building word vocabulary ---")
    vocab = build_vocabulary(all_names)

    # Save vocabulary
    vocab_path = OUTPUT_DIR / "word_vocabulary.txt"
    with open(vocab_path, "w") as f:
        for w in sorted(vocab):
            f.write(w + "\n")
    print(f"  Saved {vocab_path}")

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
        review_path = OUTPUT_DIR / "fuzzy_name_review.csv"
        review.to_csv(review_path, index=False)
        print(f"  Names changed: {len(review)} unique mappings ({changed.shape[0]:,} rows)")
        print(f"  Saved {review_path}")
    else:
        print("  No names changed.")

    before_unique = df["colorname_original"].nunique()
    after_unique = df["colorname"].nunique()
    print(f"\n  Unique names: {before_unique:,} → {after_unique:,} ({before_unique - after_unique:,} merged)")
    print(f"  Total rows preserved: {len(df):,}")

    # --- Frequency cutoff: drop colors with < MIN_COLOR_FREQ occurrences ---
    MIN_COLOR_FREQ = 10
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
    path = OUTPUT_DIR / "all_token_matches.csv"
    out.to_csv(path, index=False)
    print(f"  Saved {path} ({len(out)} transformations)")


def _save_color_frequencies(df: pd.DataFrame) -> None:
    """Save final color name frequencies."""
    freq = df["colorname"].value_counts().reset_index()
    freq.columns = ["colorname", "frequency"]
    path = OUTPUT_DIR / "color_frequencies.csv"
    freq.to_csv(path, index=False)
    print(f"  Saved {path} ({len(freq)} unique colors)")


def _save_word_frequencies(df: pd.DataFrame) -> None:
    """Save word-level token frequencies across all color names."""
    word_counts: Counter = Counter()
    for name in df["colorname"]:
        word_counts.update(name.split())
    freq = pd.DataFrame(word_counts.most_common(), columns=["word", "frequency"])
    path = OUTPUT_DIR / "word_frequencies.csv"
    freq.to_csv(path, index=False)
    print(f"  Saved {path} ({len(freq)} unique words)")


def apply_normalizations(df: pd.DataFrame) -> pd.DataFrame:
    """Apply reviewed normalizations and export the final CSV.

    Reads fuzzy_word_review.csv (user may have edited 'action' column)
    and re-runs normalization with the updated corrections.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Re-run normalize (it reads the review file)
    df = normalize(df)

    # Drop the helper column for export
    if "colorname_original" in df.columns:
        df = df.drop(columns=["colorname_original"])

    out_path = OUTPUT_DIR / "cleaned_colors.csv"
    df.to_csv(out_path, index=False)
    print(f"\nExported {len(df):,} rows to {out_path}")

    return df
