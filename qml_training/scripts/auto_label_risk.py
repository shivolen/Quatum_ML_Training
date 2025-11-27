from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import Tuple

import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("risk_labeler")

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "dataset" / "features.csv"
BACKUP_PATH = BASE_DIR / "dataset" / "features_backup.csv"
REQUIRED_COLUMNS = ["object_cat", "dist", "motion", "speed", "audio", "risk"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-label risk values in dataset/features.csv")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute risk labels even if a valid label already exists.",
    )
    return parser.parse_args()


def backup_dataset(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    logger.info("Backup stored at %s", destination)


def validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"dataset/features.csv is missing required columns: {missing}")


def compute_risk_row(row: pd.Series) -> int:
    object_cat = int(row["object_cat"])
    dist = float(row["dist"])

    if object_cat == 1:
        return 0

    if object_cat == 3 and dist < 0.5:
        return 2

    if dist < 0.3:
        risk = 2
    elif dist < 0.6:
        risk = 1
    else:
        risk = 0

    if object_cat == 2 and dist < 0.4 and risk < 1:
        risk = 1

    return risk


def assign_risk_labels(df: pd.DataFrame, force: bool) -> Tuple[pd.DataFrame, dict]:
    risk_counts = {0: 0, 1: 0, 2: 0}

    def needs_update(value: float) -> bool:
        if force:
            return True
        try:
            return pd.isna(value) or int(value) not in {0, 1, 2}
        except (ValueError, TypeError):
            return True

    for idx, row in df.iterrows():
        if needs_update(row["risk"]):
            df.at[idx, "risk"] = compute_risk_row(row)
        risk_value = int(df.at[idx, "risk"])
        risk_counts[risk_value] += 1

    return df, risk_counts


def main() -> None:
    args = parse_args()

    if not DATASET_PATH.exists():
        logger.error("Dataset not found at %s", DATASET_PATH)
        return

    logger.info("Loading dataset from %s", DATASET_PATH)
    df = pd.read_csv(DATASET_PATH)
    validate_columns(df)
    logger.info("Loaded %s rows.", len(df))

    backup_dataset(DATASET_PATH, BACKUP_PATH)

    logger.info("Assigning risk labels%s...", " (force mode)" if args.force else "")
    updated_df, risk_counts = assign_risk_labels(df, args.force)

    updated_df.to_csv(DATASET_PATH, index=False)
    logger.info("Summary:\n  Safe: %s\n  Caution: %s\n  Danger: %s", risk_counts[0], risk_counts[1], risk_counts[2])
    logger.info("Saved updated dataset to %s", DATASET_PATH)


if __name__ == "__main__":
    main()

