import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

try:
    import cv2  # type: ignore import
except ImportError as exc:
    raise ImportError("OpenCV is required for dataset utilities. Please install opencv-python.") from exc


def load_video_frames(
    video_path: str,
    resize: Optional[Tuple[int, int]] = None,
    stride: int = 1,
) -> List[Tuple[int, Any]]:
    """Load frames from a video while optionally resizing and skipping frames."""
    if stride < 1:
        raise ValueError("stride must be a positive integer")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    frames: List[Tuple[int, Any]] = []
    frame_id = 0
    sampled_id = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            if sampled_id % stride == 0:
                if resize is not None:
                    frame = cv2.resize(frame, resize)
                frames.append((frame_id, frame))
                frame_id += 1
            sampled_id += 1
    finally:
        capture.release()

    return frames


def iter_video_frames(
    video_path: str,
    resize: Optional[Tuple[int, int]] = None,
    stride: int = 1,
) -> Iterator[Tuple[int, Any]]:
    """A generator variant of load_video_frames to reduce memory usage."""
    if stride < 1:
        raise ValueError("stride must be a positive integer")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    sampled_id = 0
    frame_id = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            if sampled_id % stride == 0:
                if resize is not None:
                    frame = cv2.resize(frame, resize)
                yield frame_id, frame
                frame_id += 1
            sampled_id += 1
    finally:
        capture.release()


def load_object_images(
    obj_folder: str,
    resize: Optional[Tuple[int, int]] = (224, 224),
) -> List[Any]:
    """Load the query object images from a folder."""
    folder = Path(obj_folder)
    if not folder.exists():
        raise FileNotFoundError(f"Object folder does not exist: {obj_folder}")

    images: List[Any] = []
    for image_path in sorted(folder.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        if resize is not None:
            image = cv2.resize(image, resize)
        images.append(image)

    if not images:
        raise ValueError(f"No object images found in: {obj_folder}")

    return images


def load_annotations(json_path: str) -> Dict[str, Any]:
    """Load annotations from a JSON file."""
    with open(json_path, "r", encoding="utf-8") as file:
        return json.load(file)


def list_video_sample_dirs(root_dir: str) -> List[Path]:
    """Return sorted sample directories within the dataset root."""
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root_dir}")

    sample_dirs = [path for path in root.iterdir() if path.is_dir()]
    sample_dirs.sort()
    if not sample_dirs:
        raise ValueError(f"No sample directories found under: {root_dir}")
    return sample_dirs
