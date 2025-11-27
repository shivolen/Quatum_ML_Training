from __future__ import annotations

import base64
import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import cv2
import numpy as np
import requests

import os


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("dataset.extract_features")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CSV_HEADER = ["object_cat", "dist", "motion", "speed", "audio", "risk"]

MOVING_KEYWORDS = {"car", "bike", "motorcycle", "bus", "scooter", "person", "dog"}
STATIC_KEYWORDS = {
    "pole",
    "bench",
    "chair",
    "table",
    "wall",
    "door",
    "stairs",
    "railing",
    "sink",
    "tree",
    "pillar",
}

BOX_PATTERN = re.compile(
    r"([A-Za-z0-9_\-\s]+)\s*[:,]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)"
)


@dataclass
class Detection:
    label: str
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def height(self) -> float:
        return abs(self.y2 - self.y1)


def get_dataset_paths() -> tuple[Path, Path]:
    base_dir = Path(__file__).resolve().parents[1]
    dataset_dir = base_dir / "dataset"
    images_dir = dataset_dir / "images"
    features_path = dataset_dir / "features.csv"
    images_dir.mkdir(parents=True, exist_ok=True)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    return images_dir, features_path


def list_image_files(images_dir: Path) -> Iterable[Path]:
    for image_path in sorted(images_dir.glob("*")):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            yield image_path


def load_image(image_path: Path) -> tuple[np.ndarray, bytes]:
    matrix = cv2.imread(str(image_path))
    if matrix is None:
        raise ValueError(f"Unable to decode image: {image_path}")
    data = image_path.read_bytes()
    return matrix, data


def ensure_csv_header(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header_line = ",".join(CSV_HEADER)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        csv_path.write_text(header_line + "\n", encoding="utf-8")
        return

    with csv_path.open("r", encoding="utf-8") as existing:
        contents = existing.read()
    stripped = contents.lstrip("\ufeff")
    if stripped.startswith(header_line):
        return

    with csv_path.open("w", encoding="utf-8", newline="") as destination:
        destination.write(header_line + "\n")
        destination.write(stripped if stripped else "")


def call_gemini(image_bytes: bytes) -> Optional[str]:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    gemini_api_url = os.getenv(
        "GEMINI_API_URL",
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-001:generateContent",
    )
    
    if not gemini_api_key:
        logger.error("GEMINI_API_KEY is not configured. Set it as an environment variable or in .env file.")
        return None

    img_base64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": (
                            "Detect objects and provide bounding boxes in the format: "
                            "name,x1,y1,x2,y2. Only include detections you are confident about."
                        )
                    },
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": img_base64,
                        }
                    },
                ]
            }
        ],
        "generationConfig": {"temperature": 0.1, "topP": 1, "topK": 32, "maxOutputTokens": 256},
    }

    try:
        response = requests.post(
            gemini_api_url,
            params={"key": gemini_api_key},
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=45,
        )
        if response.status_code != 200:
            logger.error("Gemini API failed (%s): %s", response.status_code, response.text)
            return None
        body = response.json()
    except requests.exceptions.RequestException as exc:
        logger.error("Gemini API request error: %s", exc)
        return None
    except ValueError as exc:
        logger.error("Failed to decode Gemini response: %s", exc)
        return None

    return extract_text_from_response(body)


def extract_text_from_response(body: dict) -> Optional[str]:
    candidates = body.get("candidates") or []
    texts: List[str] = []
    for candidate in candidates:
        content = candidate.get("content") or {}
        for part in content.get("parts", []):
            text = part.get("text")
            if text:
                texts.append(text.strip())
    combined = "\n".join(texts).strip()
    return combined if combined else None


def parse_bounding_boxes(response_text: str) -> List[Detection]:
    detections = extract_boxes_from_json_like(response_text)
    if detections:
        return detections

    matches = BOX_PATTERN.findall(response_text)
    parsed: List[Detection] = []
    for match in matches:
        label = match[0].strip()
        try:
            coords = [float(value) for value in match[1:]]
        except ValueError:
            logger.debug("Skipping malformed coordinates in line: %s", match)
            continue
        parsed.append(Detection(label, *coords))
    return parsed


def extract_boxes_from_json_like(response_text: str) -> List[Detection]:
    candidates = [response_text]
    candidates.extend(match.strip() for match in re.findall(r"```(?:json)?(.*?)```", response_text, re.DOTALL))

    for snippet in candidates:
        try:
            data = json.loads(snippet)
        except json.JSONDecodeError:
            continue
        detections: List[Detection] = []
        if isinstance(data, dict):
            data = data.get("detections") or data.get("objects") or data.get("items")
        if isinstance(data, list):
            for entry in data:
                label = entry.get("name") or entry.get("label")
                coords = entry.get("box") or entry.get("bbox") or entry
                if not label or not isinstance(coords, (list, tuple)) or len(coords) != 4:
                    continue
                try:
                    x1, y1, x2, y2 = (float(value) for value in coords)
                except (TypeError, ValueError):
                    continue
                detections.append(Detection(str(label), x1, y1, x2, y2))
        if detections:
            return detections
    return []


def categorize_object(label: str) -> int:
    normalized = label.lower()
    if not normalized.strip() or normalized == "none":
        return 0
    if any(keyword in normalized for keyword in MOVING_KEYWORDS):
        return 3
    if any(keyword in normalized for keyword in STATIC_KEYWORDS):
        return 2
    return 1


def detection_to_row(detection: Detection, image_height: int) -> List[str]:
    if image_height <= 0:
        image_height = 1
    bbox_height = detection.height
    distance = max(0.0, min(1.0, bbox_height / float(image_height)))
    category = categorize_object(detection.label)
    return [str(category), f"{distance:.4f}", "0", "1", "0", ""]


def append_rows(csv_path: Path, rows: Iterable[List[str]]) -> None:
    with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(rows)


def main() -> None:
    images_dir, features_path = get_dataset_paths()
    ensure_csv_header(features_path)

    image_files = list(list_image_files(images_dir))
    if not image_files:
        logger.warning("No images found in %s.", images_dir)
        return

    total_objects = 0
    for image_path in image_files:
        logger.info("Processing image: %s", image_path.name)
        try:
            matrix, image_bytes = load_image(image_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load image %s: %s", image_path.name, exc)
            continue

        image_height = matrix.shape[0] if matrix is not None else 0
        response_text = call_gemini(image_bytes)
        if not response_text:
            logger.warning("Skipping %s due to empty Gemini response.", image_path.name)
            continue

        detections = parse_bounding_boxes(response_text)
        if not detections:
            logger.info("No objects found in %s", image_path.name)
            continue

        rows = [detection_to_row(det, image_height) for det in detections]
        append_rows(features_path, rows)
        total_objects += len(rows)
        logger.info("Logged %s objects for %s", len(rows), image_path.name)

    logger.info("Feature extraction complete. Total objects logged: %s", total_objects)


if __name__ == "__main__":
    main()

