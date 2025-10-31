from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    import cv2  # type: ignore import
except ImportError as exc:
    raise ImportError("OpenCV is required for visualization. Please install opencv-python.") from exc

from dataset_utils import list_video_sample_dirs
from utils import extract_video_id

Color = tuple[int, int, int]


def load_submission(submission_path: str) -> Dict[str, Dict[int, List[Dict[str, float]]]]:
    with open(submission_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    video_frames: Dict[str, Dict[int, List[Dict[str, float]]]] = {}
    for entry in data:
        video_id = entry.get("video_id")
        if not video_id:
            continue
        
        frames = defaultdict(list)
        
        for group in entry.get("annotations", []):
            for bbox in group.get("bboxes", []):
                frame_idx = int(bbox.get("frame", -1))
                if frame_idx < 0:
                    continue
                frames[frame_idx].append(bbox)
                
        for group in entry.get("detections", []):
            for bbox in group.get("bboxes", []):
                frame_idx = int(bbox.get("frame", -1))
                if frame_idx < 0:
                    continue
                frames[frame_idx].append(bbox)
                
        video_frames[video_id] = frames
        
    return video_frames


def find_videos(dataset_root: str, video_ext: str) -> Dict[str, Path]:
    video_map: Dict[str, Path] = {}
    for sample_dir in list_video_sample_dirs(dataset_root):
        for candidate in sample_dir.glob(f"*{video_ext}"):
            video_id = extract_video_id(str(candidate), prefer_parent=True)
            video_map[video_id] = candidate
            break
    return video_map


def draw_boxes(frame, boxes: Iterable[Dict[str, float]], color: Color, thickness: int, text_color: Color) -> None:
    height, width = frame.shape[:2]
    for bbox in boxes:
        x1 = int(round(bbox.get("x1", 0)))
        y1 = int(round(bbox.get("y1", 0)))
        x2 = int(round(bbox.get("x2", 0)))
        y2 = int(round(bbox.get("y2", 0)))
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        if "score" in bbox:
            text = f"{bbox['score']:.2f}"
            cv2.putText(frame, text, (x1 + 2, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
        if "track_id" in bbox:
            text = f"ID:{int(bbox['track_id'])}"
            cv2.putText(frame, text, (x1 + 2, y1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)


def visualize_video(
    video_path: Path,
    frames: Dict[int, List[Dict[str, float]]],
    show: bool,
    output_path: Optional[Path],
    color: Color,
    thickness: int,
    text_color: Color,
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    writer = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") if output_path is not None else None
    window_name = f"viz::{video_path.name}"

    frame_index = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            annotations = frames.get(frame_index, [])
            if annotations:
                draw_boxes(frame, annotations, color=color, thickness=thickness, text_color=text_color)

            if output_path is not None:
                if writer is None:
                    height, width = frame.shape[:2]
                    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height), isColor=True)
                writer.write(frame)

            if show:
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in {27, ord("q")}:
                    break

            frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyWindow(window_name)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize submission bounding boxes on videos")
    parser.add_argument("--submission", type=str, required=True, help="Path to submission JSON")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="data/public_test/samples",
        help="Root directory containing sample video folders",
    )
    parser.add_argument("--video-ext", type=str, default=".mp4", help="Expected video extension")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional directory to save rendered videos")
    parser.add_argument("--show", action="store_true", help="Display videos with overlays in a window")
    parser.add_argument("--color", type=str, default="0,255,0", help="Bounding box color as R,G,B")
    parser.add_argument("--text-color", type=str, default="255,255,255", help="Overlay text color as R,G,B")
    parser.add_argument("--thickness", type=int, default=2, help="Bounding box line thickness")
    return parser


def parse_color(value: str) -> Color:
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Color must be provided as R,G,B")
    r, g, b = (int(part) for part in parts)
    for component in (r, g, b):
        if not 0 <= component <= 255:
            raise argparse.ArgumentTypeError("Color components must be in [0, 255]")
    return (b, g, r)


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    submission = load_submission(args.submission)
    video_map = find_videos(args.dataset_root, args.video_ext)

    if args.output_dir is not None:
        output_root = Path(args.output_dir)
    else:
        output_root = None

    box_color = parse_color(args.color)
    text_color = parse_color(args.text_color)

    missing_videos: List[str] = []
    for video_id, frames in submission.items():
        video_path = video_map.get(video_id)
        if video_path is None:
            missing_videos.append(video_id)
            continue
        output_path = output_root / f"{video_id}_vis.mp4" if output_root is not None else None
        visualize_video(
            video_path,
            frames,
            show=args.show,
            output_path=output_path,
            color=box_color,
            thickness=args.thickness,
            text_color=text_color,
        )

    if missing_videos:
        print("Skipped videos without matches:", ", ".join(sorted(missing_videos)))


if __name__ == "__main__":
    main()
