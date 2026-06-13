import argparse

import numpy as np
import pandas as pd

from config import DATA_DIR, DATA_FILE, FEATURE_COLUMNS, RANDOM_STATE, TARGET_COLUMN


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_wine_data() -> pd.DataFrame:
    """Load wine quality data and normalize column names."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_FILE}. "
            "Add winequality-red.csv or run: python src/data_utils.py --create-sample"
        )

    try:
        data = pd.read_csv(DATA_FILE)
    except pd.errors.ParserError:
        data = pd.read_csv(DATA_FILE, sep=";")

    if len(data.columns) == 1:
        data = pd.read_csv(DATA_FILE, sep=";")

    data.columns = [column.strip().strip('"') for column in data.columns]

    missing_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    return data[FEATURE_COLUMNS + [TARGET_COLUMN]].dropna()


def create_sample_dataset(rows: int = 1200) -> pd.DataFrame:
    """Create a realistic demo dataset shaped like the UCI wine-quality data."""
    rng = np.random.default_rng(RANDOM_STATE)

    data = pd.DataFrame(
        {
            "fixed acidity": rng.normal(8.3, 1.7, rows).clip(4.5, 16.0),
            "volatile acidity": rng.normal(0.53, 0.18, rows).clip(0.1, 1.6),
            "citric acid": rng.normal(0.27, 0.19, rows).clip(0.0, 1.0),
            "residual sugar": rng.gamma(2.0, 1.2, rows).clip(0.8, 15.5),
            "chlorides": rng.normal(0.087, 0.045, rows).clip(0.01, 0.62),
            "free sulfur dioxide": rng.normal(15.9, 10.5, rows).clip(1, 72),
            "total sulfur dioxide": rng.normal(46.5, 32.0, rows).clip(6, 289),
            "density": rng.normal(0.9967, 0.0019, rows).clip(0.990, 1.004),
            "pH": rng.normal(3.31, 0.15, rows).clip(2.7, 4.1),
            "sulphates": rng.normal(0.66, 0.17, rows).clip(0.3, 2.0),
            "alcohol": rng.normal(10.4, 1.05, rows).clip(8.0, 15.0),
        }
    )

    signal = (
        0.42 * data["alcohol"]
        + 1.35 * data["sulphates"]
        + 0.55 * data["citric acid"]
        - 2.1 * data["volatile acidity"]
        - 2.6 * data["chlorides"]
        - 0.01 * data["total sulfur dioxide"]
        + 1.9
        + rng.normal(0, 0.55, rows)
    )
    data[TARGET_COLUMN] = np.rint(signal - 0.25).clip(3, 8).astype(int)

    ensure_directories()
    data.to_csv(DATA_FILE, index=False)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Wine quality data utilities")
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create a realistic sample dataset at data/winequality-red.csv",
    )
    parser.add_argument("--rows", type=int, default=1200, help="Sample rows to generate")
    args = parser.parse_args()

    if args.create_sample:
        data = create_sample_dataset(args.rows)
        print(f"Created sample dataset: {DATA_FILE}")
        print(f"Shape: {data.shape}")
    else:
        data = load_wine_data()
        print(f"Loaded dataset: {DATA_FILE}")
        print(f"Shape: {data.shape}")


if __name__ == "__main__":
    main()
