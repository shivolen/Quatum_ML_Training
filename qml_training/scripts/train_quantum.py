from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import pennylane as qml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("quantum.training")

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "dataset" / "features.csv"
MODEL_PATH = BASE_DIR / "models" / "quantum" / "qml_model.pkl"
NUM_QUBITS = 5
WIRES = list(range(NUM_QUBITS))

device = qml.device("default.qubit", wires=NUM_QUBITS)

def _state_fidelity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    inner = np.vdot(vec_a, vec_b)
    return float(np.abs(inner) ** 2)


def feature_map(x: np.ndarray) -> None:
    qml.AngleEmbedding(x, wires=WIRES, rotation="Y")
    for control, target in zip(WIRES[:-1], WIRES[1:]):
        qml.CNOT(wires=[control, target])


@qml.qnode(device)
def quantum_state(x: np.ndarray) -> np.ndarray:
    feature_map(x)
    return qml.state()


def load_dataset(dataset_path: Path = DATASET_PATH) -> Tuple[np.ndarray, np.ndarray]:
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        logger.warning("Dataset not found at %s. Generating dummy dataset.", dataset_path)
        df = generate_dummy_dataset()
        df.to_csv(dataset_path, index=False)
    else:
        df = pd.read_csv(dataset_path)
        if df.empty or "risk" not in df.columns:
            logger.warning("Dataset missing risk column or empty. Generating dummy dataset.")
            df = generate_dummy_dataset()
    df = df.dropna(subset=["risk"])
    df["risk"] = pd.to_numeric(df["risk"], errors="coerce")
    df = df.dropna(subset=["risk"])

    feature_cols = ["object_cat", "dist", "motion", "speed", "audio"]
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        logger.warning("Missing feature columns %s. Generating dummy dataset.", missing_cols)
        df = generate_dummy_dataset()

    features = df[feature_cols].to_numpy(dtype=float)
    labels = df["risk"].to_numpy(dtype=int)

    if features.size == 0 or labels.size == 0:
        logger.warning("Dataset contains no usable rows. Generating dummy dataset.")
        df = generate_dummy_dataset()
        features = df[feature_cols].to_numpy(dtype=float)
        labels = df["risk"].to_numpy(dtype=int)
    return features, labels


def generate_dummy_dataset(rows: int = 50) -> pd.DataFrame:
    logger.info("Generating dummy dataset with %s samples.", rows)
    rng = np.random.default_rng(42)
    data = {
        "object_cat": rng.integers(0, 4, size=rows),
        "dist": rng.random(size=rows),
        "motion": rng.integers(0, 2, size=rows),
        "speed": rng.random(size=rows),
        "audio": rng.random(size=rows),
        "risk": rng.integers(0, 3, size=rows),
    }
    return pd.DataFrame(data)


def normalize_features(
    features: np.ndarray, scaler: Optional[Dict[str, np.ndarray]] = None
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    if scaler is None:
        min_vals = features.min(axis=0)
        max_vals = features.max(axis=0)
        ranges = np.where(max_vals - min_vals == 0, 1.0, max_vals - min_vals)
        scaler = {"min": min_vals, "range": ranges}
    else:
        min_vals = scaler["min"]
        ranges = scaler["range"]

    normalized = (features - min_vals) / ranges
    normalized = np.clip(normalized, 0.0, 1.0)
    return normalized, scaler


def quantum_kernel(
    x1: np.ndarray,
    x2: np.ndarray,
) -> np.ndarray:
    logger.debug("Building quantum kernel for shapes %s and %s", x1.shape, x2.shape)
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    if x1.ndim == 1:
        x1 = x1.reshape(1, -1)
    if x2.ndim == 1:
        x2 = x2.reshape(1, -1)
    kernel = np.zeros((x1.shape[0], x2.shape[0]), dtype=float)
    for i, row_a in enumerate(x1):
        for j, row_b in enumerate(x2):
            state_a = quantum_state(row_a)
            state_b = quantum_state(row_b)
            kernel[i, j] = _state_fidelity(state_a, state_b)
    return kernel


def train_qml_model(
    features: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    logger.info("Splitting dataset (test_size=%s).", test_size)
    if len(features) < 2:
        X_train, X_test = features, np.empty((0, features.shape[1]))
        y_train, y_test = labels, np.empty((0,), dtype=int)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels if len(np.unique(labels)) > 1 else None,
        )

    logger.info("Building training kernel matrix (%s samples).", len(X_train))
    K_train = quantum_kernel(X_train, X_train)

    logger.info("Training quantum SVM.")
    svm = SVC(kernel="precomputed")
    svm.fit(K_train, y_train)

    accuracy = None
    if len(X_test) > 0:
        logger.info("Evaluating on test split (%s samples).", len(X_test))
        K_test = quantum_kernel(X_test, X_train)
        predictions = svm.predict(K_test)
        accuracy = accuracy_score(y_test, predictions)
        logger.info("Test accuracy: %.4f", accuracy)

    return {
        "model": svm,
        "train_features": X_train,
        "accuracy": accuracy,
    }


def predict(
    features: np.ndarray,
    model_bundle: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    if model_bundle is None:
        model_bundle = load_model()

    scaler = model_bundle["scaler"]
    normalized, _ = normalize_features(np.asarray(features, dtype=float), scaler)
    train_features = model_bundle["train_features"]
    model: SVC = model_bundle["model"]
    kernel = quantum_kernel(normalized, train_features)
    return model.predict(kernel)


def save_model(
    model: SVC,
    scaler: Dict[str, np.ndarray],
    train_features: np.ndarray,
    model_path: Path = MODEL_PATH,
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "train_features": train_features,
        },
        model_path,
    )
    logger.info("Saved model artifacts to %s", model_path)


def load_model(model_path: Path = MODEL_PATH) -> Dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}. Train the model first.")
    return joblib.load(model_path)


def main() -> None:
    logger.info("Loading dataset from %s", DATASET_PATH)
    features, labels = load_dataset(DATASET_PATH)

    logger.info("Normalizing features.")
    normalized_features, scaler = normalize_features(features)

    result = train_qml_model(normalized_features, labels)

    scaler_bundle = {"min": scaler["min"], "range": scaler["range"]}
    save_model(result["model"], scaler_bundle, result["train_features"], MODEL_PATH)

    logger.info(
        "Training completed. Accuracy: %s",
        f"{result['accuracy']:.4f}" if result["accuracy"] is not None else "N/A",
    )


if __name__ == "__main__":
    main()

