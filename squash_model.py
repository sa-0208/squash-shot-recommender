"""Shared feature preparation and model helpers for the squash project."""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

LETTERS = {letter: i + 1 for i, letter in enumerate("ABCDEFGH")}

CATEGORICAL_FEATURES = [
    "Hitter", "Shot", "Hand", "Direction", "Boast wall count",
    "Volley?", "Ball speed",
]
NUMERIC_FEATURES = [
    "Hitter position x", "Hitter position y",
    "Opponent position x", "Opponent position y",
    "Ball position x", "Ball position y",
    "Target x", "Target y",
    "Player separation", "Hitter-ball distance",
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def coordinate_to_number(value):
    """Convert A-H to 1-8. A1 is bottom-left; x increases right, y increases up."""
    if pd.isna(value) or str(value).strip() == "":
        return np.nan
    value = str(value).strip().upper()
    if value not in LETTERS:
        raise ValueError(f"Invalid court letter coordinate: {value!r}. Expected A-H.")
    return LETTERS[value]


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-safe model features from spreadsheet columns."""
    data = frame.copy()

    required = [
        "Hitter", "Hitter position X (A-H)", "Hitter position Y (1-8)",
        "Opponent position X (A-H)", "Opponent position Y (1-8)",
        "Ball position X (A-H)", "Ball position Y (1-8)",
        "Shot", "Hand", "Direction", "Boast wall count", "Volley?",
        "Target / landing X (A-H)", "Target / landing Y (1-8)", "Ball speed",
    ]
    missing = [col for col in required if col not in data.columns]
    if missing:
        raise ValueError(
            "The input data is missing columns: " + ", ".join(missing)
        )

    data["Hitter position x"] = data["Hitter position X (A-H)"].map(coordinate_to_number)
    data["Hitter position y"] = pd.to_numeric(data["Hitter position Y (1-8)"], errors="coerce")
    data["Opponent position x"] = data["Opponent position X (A-H)"].map(coordinate_to_number)
    data["Opponent position y"] = pd.to_numeric(data["Opponent position Y (1-8)"], errors="coerce")
    data["Ball position x"] = data["Ball position X (A-H)"].map(coordinate_to_number)
    data["Ball position y"] = pd.to_numeric(data["Ball position Y (1-8)"], errors="coerce")
    data["Target x"] = data["Target / landing X (A-H)"].map(coordinate_to_number)
    data["Target y"] = pd.to_numeric(data["Target / landing Y (1-8)"], errors="coerce")

    data["Boast wall count"] = data["Boast wall count"].replace("", np.nan)
    data["Boast wall count"] = pd.to_numeric(data["Boast wall count"], errors="coerce")
    data["Player separation"] = np.sqrt(
        (data["Hitter position x"] - data["Opponent position x"]) ** 2
        + (data["Hitter position y"] - data["Opponent position y"]) ** 2
    )
    data["Hitter-ball distance"] = np.sqrt(
        (data["Hitter position x"] - data["Ball position x"]) ** 2
        + (data["Hitter position y"] - data["Ball position y"]) ** 2
    )

    # Missing boast-wall counts are represented as "Not applicable".
    data["Boast wall count"] = data["Boast wall count"].fillna(-1).astype(str)
    for col in ["Hitter", "Shot", "Hand", "Direction", "Volley?", "Ball speed"]:
        data[col] = data[col].fillna("Unknown").astype(str)

    return data[FEATURE_COLUMNS]


def load_training_data(path):
    """Load the main worksheet from either the supplied Excel workbook or a CSV."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name="Synthetic Rally Data")
    raise ValueError("Use a .xlsx or .csv dataset file.")


def load_saved_model(path="artifacts/squash_shot_model.joblib"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Run `python train_model.py` first."
        )
    return joblib.load(path)
