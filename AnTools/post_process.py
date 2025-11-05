# 🔗 File paths
INPUT_TXT  = "gt.txt"          # input text file
OUTPUT_JSON = "gt_converted.json"  # output JSON file
# ================================================================

import json

def convert_txt_to_json(input_path, output_path):
    videos = {}

    with open(input_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 10:
                continue

            video_id = parts[0]
            frame_file = parts[4]
            has_obj = int(parts[5])
            x1, y1, x2, y2 = map(float, parts[6:])
            frame_num = int(frame_file.split("frame")[-1].split("_")[0])

            # Only include frames where object exists
            if has_obj == 0:
                continue

            bbox = {
                "frame": frame_num,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            }

            videos.setdefault(video_id, []).append(bbox)

    # build final JSON structure
    json_data = []
    for vid, boxes in videos.items():
        json_data.append({
            "video_id": vid,
            "detections": [
                {"bboxes": boxes}
            ]
        })

    with open(output_path, "w") as f:
        json.dump(json_data, f, indent=4)
    print(f"✅ Converted {len(json_data)} videos → saved to {output_path}")


if __name__ == "__main__":
    convert_txt_to_json(INPUT_TXT, OUTPUT_JSON)
