"""Phase 1: Extract color data from PostgreSQL (or CSV fallback) and apply basic string cleanup."""

import re
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

DB_URL = "postgresql://postgres:postgres@localhost:17433/colorsurvey"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CSV_FALLBACK = PROJECT_ROOT / "mainsurvey_data_with_space.csv"


def _get_colorblind_user_ids(engine) -> set:
    """Return set of user IDs flagged as colorblind."""
    query = text("""
        SELECT id FROM public.mainsurvey_users
        WHERE LOWER(TRIM(CAST(colorblind AS TEXT))) IN ('true', 't', '1', 'yes', 'y')
    """)
    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn)
    return set(pd.to_numeric(df["id"], errors="coerce").dropna().astype(int))


def _basic_string_cleanup(s: str) -> str:
    """Lowercase, replace separators, split camelCase, collapse whitespace."""
    s = s.lower()
    s = re.sub(r"[-_]", " ", s)
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)  # camelCase split
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract() -> pd.DataFrame:
    """Load from DB (or CSV fallback), remove colorblind users, clean strings, filter junk chars.

    Saves output/raw_extract.csv and returns the DataFrame.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Try DB first, fall back to pre-exported CSV
    try:
        print("Connecting to database...")
        engine = create_engine(DB_URL)
        print("Loading mainsurvey_answers...")
        df = pd.read_sql_table("mainsurvey_answers", engine)
        print(f"  Raw rows: {len(df):,}")

        # Exclude colorblind users
        colorblind_ids = _get_colorblind_user_ids(engine)
        df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce")
        df = df[df["user_id"].notna()]
        df["user_id"] = df["user_id"].astype(int)
        df = df[~df["user_id"].isin(colorblind_ids)]
        print(f"  After removing {len(colorblind_ids)} colorblind users: {len(df):,}")
    except Exception as e:
        print(f"  DB connection failed: {e}")
        if CSV_FALLBACK.exists():
            print(f"  Falling back to {CSV_FALLBACK}")
            df = pd.read_csv(CSV_FALLBACK)
            print(f"  Loaded {len(df):,} rows from CSV (colorblind users already filtered)")
        else:
            raise

    # Basic string cleanup
    df["colorname"] = df["colorname"].apply(_basic_string_cleanup)

    # Keep only alphabetic + spaces, length >= 3
    mask = df["colorname"].str.match(r"^[a-z ]+$") & (df["colorname"].str.len() >= 3)
    df = df[mask].copy()
    print(f"  After char/length filter: {len(df):,}")
    print(f"  Unique color names: {df['colorname'].nunique():,}")

    out_path = OUTPUT_DIR / "raw_extract.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved {out_path}")
    return df
