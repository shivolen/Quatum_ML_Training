from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import os


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("dataset.extract_frames")

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi"}
BLACK_FRAME_THRESHOLD = 8.0  # average pixel intensity
DUPLICATE_DIFF_THRESHOLD = 1.5  # mean absolute difference between frames


def get_dataset_paths() -> tuple[Path, Path]:
    base_dir = Path(__file__).resolve().parents[1]
    dataset_dir = base_dir / "dataset"
    videos_dir = dataset_dir / "videos"
    images_dir = dataset_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    return videos_dir, images_dir


def list_video_files(video_dir: Path) -> Iterable[Path]:
    for video_path in sorted(video_dir.glob("*")):
        if video_path.is_file() and video_path.suffix.lower() in VIDEO_EXTENSIONS:
            yield video_path


def is_black_frame(frame: np.ndarray) -> bool:
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        return False
    return float(np.mean(gray)) <= BLACK_FRAME_THRESHOLD


def is_duplicate_frame(frame: np.ndarray, last_frame: Optional[np.ndarray]) -> bool:
    if last_frame is None:
        return False
    if frame.shape != last_frame.shape:
        return False
    diff = cv2.absdiff(frame, last_frame)
    mean_diff = float(np.mean(diff))
    return mean_diff <= DUPLICATE_DIFF_THRESHOLD


def save_frame(frame: np.ndarray, output_dir: Path, video_stem: str, frame_number: int) -> bool:
    filename = f"{video_stem}_frame_{frame_number:04d}.jpg"
    output_path = output_dir / filename
    try:
        success = cv2.imwrite(str(output_path), frame)
        if not success:
            logger.error("Failed to write frame to %s", output_path)
        return success
    except cv2.error as exc:
        logger.error("OpenCV error while saving frame %s: %s", output_path, exc)
        return False


def extract_frames_from_video(video_path: Path, output_dir: Path, interval: int) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Could not open video: %s", video_path)
        return 0

    video_stem = video_path.stem.replace(" ", "_")
    frame_index = 0
    saved_frames = 0
    last_saved_frame: Optional[np.ndarray] = None

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            if frame_index % interval == 0:
                if frame is None:
                    logger.warning("Skipping empty frame %s at index %s", video_path.name, frame_index)
                elif is_black_frame(frame):
                    logger.debug("Skipping black frame %s at index %s", video_path.name, frame_index)
                elif is_duplicate_frame(frame, last_saved_frame):
                    logger.debug("Skipping duplicate frame %s at index %s", video_path.name, frame_index)
                else:
                    frame_number = saved_frames + 1
                    if save_frame(frame, output_dir, video_stem, frame_number):
                        saved_frames += 1
                        last_saved_frame = frame
            frame_index += 1
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed while processing video %s: %s", video_path.name, exc)
    finally:
        cap.release()

    return saved_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames from all videos in dataset/videos.")
    parser.add_argument(
        "--interval",
        type=int,
        default=int(os.environ.get("FRAME_INTERVAL", 10)) if "FRAME_INTERVAL" in os.environ else 10,
        help="Extract one frame every N frames (default: 10).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.interval <= 0:
        logger.error("Interval must be a positive integer.")
        return

    videos_dir, images_dir = get_dataset_paths()
    if not videos_dir.exists():
        logger.error("Video directory %s does not exist.", videos_dir)
        return

    video_files = list(list_video_files(videos_dir))
    if not video_files:
        logger.warning("No video files found in %s.", videos_dir)
        return

    total_frames = 0
    for video_path in video_files:
        logger.info("Processing video: %s", video_path.name)
        extracted = extract_frames_from_video(video_path, images_dir, args.interval)
        logger.info("Extracted %s frames from %s", extracted, video_path.name)
        total_frames += extracted

    logger.info("Frame extraction complete. Total frames extracted: %s", total_frames)


if __name__ == "__main__":
    main()

