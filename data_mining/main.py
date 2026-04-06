"""Color name normalization pipeline — CLI entry point.

Usage:
    python main.py extract      # Phase 1: DB extract + basic cleanup
    python main.py clean        # Phase 1 + 2: + junk removal
    python main.py normalize    # Phase 1 + 2 + 3: + word-level normalization (generates review files)
    python main.py apply        # After review: re-normalize with reviewed corrections + export
    python main.py all          # Full pipeline (same as apply)
"""

import sys
from pathlib import Path

import pandas as pd

# Allow running from data_mining/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import extract, clean, normalize, apply_normalizations

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def _load_raw() -> pd.DataFrame:
    """Load raw_extract.csv if it exists, otherwise run extraction."""
    raw_path = OUTPUT_DIR / "raw_extract.csv"
    if raw_path.exists():
        print(f"Loading cached {raw_path}...")
        return pd.read_csv(raw_path)
    return extract()


def cmd_extract():
    print("=" * 60)
    print("PHASE 1: Extract")
    print("=" * 60)
    extract()


def cmd_clean():
    print("=" * 60)
    print("PHASE 1: Extract")
    print("=" * 60)
    df = _load_raw()

    print("\n" + "=" * 60)
    print("PHASE 2: Clean")
    print("=" * 60)
    clean(df)


def cmd_normalize():
    print("=" * 60)
    print("PHASE 1: Extract")
    print("=" * 60)
    df = _load_raw()

    print("\n" + "=" * 60)
    print("PHASE 2: Clean")
    print("=" * 60)
    df = clean(df)

    print("\n" + "=" * 60)
    print("PHASE 3: Normalize")
    print("=" * 60)
    normalize(df)
    print("\nReview the generated files in output/ before running 'apply'.")


def cmd_apply():
    print("=" * 60)
    print("PHASE 1: Extract")
    print("=" * 60)
    df = _load_raw()

    print("\n" + "=" * 60)
    print("PHASE 2: Clean")
    print("=" * 60)
    df = clean(df)

    print("\n" + "=" * 60)
    print("PHASE 3+4: Normalize + Export")
    print("=" * 60)
    apply_normalizations(df)


COMMANDS = {
    "extract": cmd_extract,
    "clean": cmd_clean,
    "normalize": cmd_normalize,
    "apply": cmd_apply,
    "all": cmd_apply,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(f"Usage: python main.py <{'|'.join(COMMANDS)}>")
        sys.exit(1)

    COMMANDS[sys.argv[1]]()


if __name__ == "__main__":
    main()
