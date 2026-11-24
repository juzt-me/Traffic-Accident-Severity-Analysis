"""Loading and inspecting the Kaggle US Accidents CSV."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DATA_PATH = DATA_DIR / "accidents.csv"
SAMPLE_CACHE_PATH = DATA_DIR / "accidents_sample.csv"
SAMPLE_METADATA_PATH = DATA_DIR / "sample_metadata.json"
DEFAULT_SAMPLE_SIZE = 100_000
TARGET_COLUMN = "Severity"


def resolve_dataset_path(filepath: str | Path | None = None) -> Path:
    """Resolve an explicitly supplied CSV or the common local dataset names."""
    if filepath is not None:
        path = Path(filepath).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if path.is_file():
            return path
        raise FileNotFoundError(_dataset_help(path))

    candidates = (
        DEFAULT_DATA_PATH,
        DATA_DIR / "US_Accidents_March23.csv",
        DATA_DIR / "US_Accidents_Dec21_updated.csv",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(_dataset_help(DEFAULT_DATA_PATH))


def _dataset_help(path: Path) -> str:
    return (
        f"US Accidents dataset not found at {path}. Download the real Kaggle dataset "
        "from https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents, then place "
        "the extracted CSV in data/ and name it accidents.csv. See data/README.md "
        "for manual and authenticated download instructions."
    )


def _reservoir_sample_csv(
    path: Path,
    sample_size: int,
    random_state: int,
    chunk_size: int,
) -> tuple[pd.DataFrame, int]:
    """Uniformly sample rows from a large CSV without keeping the whole file in memory."""
    rng = np.random.default_rng(random_state)
    sample = pd.DataFrame()
    seen = 0

    try:
        chunks = pd.read_csv(path, chunksize=chunk_size, low_memory=False)
        for chunk in chunks:
            if chunk.empty:
                continue
            if seen == 0:
                sample = chunk.head(sample_size).copy()
                seen = len(chunk)
                if seen > sample_size:
                    sample = chunk.sample(
                        n=sample_size, random_state=random_state
                    ).copy()
                continue

            old_rows_to_keep = int(
                rng.hypergeometric(
                    ngood=len(sample),
                    nbad=len(chunk),
                    nsample=min(sample_size, seen + len(chunk)),
                )
            )
            new_rows_to_keep = min(sample_size, seen + len(chunk)) - old_rows_to_keep
            if old_rows_to_keep:
                retained = sample.sample(
                    n=old_rows_to_keep,
                    random_state=int(rng.integers(0, 2**32 - 1)),
                )
            else:
                retained = sample.iloc[0:0]
            if new_rows_to_keep:
                incoming = chunk.sample(
                    n=new_rows_to_keep,
                    random_state=int(rng.integers(0, 2**32 - 1)),
                )
                sample = pd.concat([retained, incoming], ignore_index=True)
            else:
                sample = retained.reset_index(drop=True)
            seen += len(chunk)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not read dataset CSV at {path}: {exc}") from exc

    if seen == 0:
        raise ValueError(f"The CSV file at {path} contains no data rows.")
    return sample.reset_index(drop=True), seen


def load_dataset(
    filepath: str | Path | None = None,
    max_rows: int | None = DEFAULT_SAMPLE_SIZE,
    random_state: int = 42,
    chunk_size: int = 100_000,
) -> pd.DataFrame:
    """Load a dataset, uniformly sampling at most ``max_rows`` rows by default.

    Set ``max_rows=None`` to load the full CSV (which may require substantial RAM).
    """
    path = resolve_dataset_path(filepath)
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be a positive integer or None.")
    if chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer.")

    cacheable = (
        filepath is None
        and max_rows == DEFAULT_SAMPLE_SIZE
        and random_state == 42
        and path == DEFAULT_DATA_PATH
    )
    if cacheable and SAMPLE_CACHE_PATH.is_file() and SAMPLE_METADATA_PATH.is_file():
        try:
            metadata = json.loads(SAMPLE_METADATA_PATH.read_text(encoding="utf-8"))
            stat = path.stat()
            cache_matches = (
                metadata.get("source_size") == stat.st_size
                and metadata.get("source_mtime_ns") == stat.st_mtime_ns
                and metadata.get("sample_size") == DEFAULT_SAMPLE_SIZE
                and metadata.get("random_state") == 42
            )
            if cache_matches:
                cached = pd.read_csv(SAMPLE_CACHE_PATH, low_memory=False)
                validate_dataset(cached)
                cached.attrs["source_rows"] = metadata["source_rows"]
                cached.attrs["sampled"] = len(cached) < metadata["source_rows"]
                print(
                    f"Loaded cached sample: {len(cached):,} of "
                    f"{metadata['source_rows']:,} source rows."
                )
                return cached
        except (
            OSError,
            ValueError,
            KeyError,
            json.JSONDecodeError,
            pd.errors.ParserError,
        ) as exc:
            print(f"Cached sample is unusable ({exc}); rebuilding it from the source CSV.")

    try:
        if max_rows is None:
            df = pd.read_csv(path, low_memory=False)
            source_rows = len(df)
        else:
            df, source_rows = _reservoir_sample_csv(
                path, max_rows, random_state, chunk_size
            )
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not read dataset CSV at {path}: {exc}") from exc

    df.columns = [str(column).strip() for column in df.columns]
    validate_dataset(df)
    df.attrs["source_rows"] = source_rows
    df.attrs["sampled"] = len(df) < source_rows
    if cacheable:
        stat = path.stat()
        df.to_csv(SAMPLE_CACHE_PATH, index=False)
        SAMPLE_METADATA_PATH.write_text(
            json.dumps(
                {
                    "source_size": stat.st_size,
                    "source_mtime_ns": stat.st_mtime_ns,
                    "source_rows": source_rows,
                    "sample_size": DEFAULT_SAMPLE_SIZE,
                    "random_state": random_state,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Cached the reproducible sample at {SAMPLE_CACHE_PATH}.")
    return df


def validate_dataset(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> bool:
    """Check that the CSV has rows and contains the target column."""
    if df is None or df.empty:
        raise ValueError("The dataset is empty.")
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found. "
            f"Available columns: {', '.join(map(str, df.columns))}"
        )
    if df[target_column].notna().sum() == 0:
        raise ValueError(f"Target column '{target_column}' has no usable values.")
    return True


def inspect_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Return data quality and target-class information calculated from ``df``."""
    validate_dataset(df)
    return {
        "shape": tuple(df.shape),
        "source_rows": int(df.attrs.get("source_rows", len(df))),
        "sampled_rows": len(df),
        "is_sampled": bool(df.attrs.get("sampled", False)),
        "columns": list(df.columns),
        "dtypes": {str(column): str(dtype) for column, dtype in df.dtypes.items()},
        "missing_values": df.isna().sum().astype(int).to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "severity_counts": df[TARGET_COLUMN].value_counts(dropna=False).to_dict(),
    }


if __name__ == "__main__":
    data = load_dataset()
    summary = inspect_dataset(data)
    print(f"Loaded {summary['shape'][0]:,} records x {summary['shape'][1]} columns.")
    if summary["is_sampled"]:
        print(f"Sampled from {summary['source_rows']:,} total source records.")
    print("Columns:", ", ".join(summary["columns"]))
    print("\nColumn data types:")
    print(pd.Series(summary["dtypes"]).to_string())
    print("\nMissing values by column:")
    missing = pd.Series(summary["missing_values"])
    print(missing[missing > 0].to_string() if missing.any() else "No missing values.")
    print(f"\nDuplicate rows: {summary['duplicate_rows']:,}")
    print("Severity counts:")
    print(pd.Series(summary["severity_counts"]).to_string())
