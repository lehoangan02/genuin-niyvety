from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


def extract_video_id(video_path: str, *, prefer_parent: bool = False) -> str:
    path = Path(video_path)
    if prefer_parent:
        parent = path.parent
        if parent != path and parent.name:
            return parent.name
    if path.suffix:
        return path.stem
    return path.name


def save_predictions(
    video_id: str,
    detections: List[Dict[str, float]],
    output_json: str,
) -> None:
    output_path = Path(output_json)
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    found = False
    for entry in data:
        if entry.get("video_id") == video_id:
            entry["detections"] = [{"bboxes": detections}]
            found = True
            break

    if not found:
        data.append({"video_id": video_id, "detections": [{"bboxes": detections}]})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def xyxy_to_xywh(box: Dict[str, float]) -> Dict[str, float]:
    width = box["x2"] - box["x1"]
    height = box["y2"] - box["y1"]
    return {"x": box["x1"], "y": box["y1"], "width": width, "height": height}
