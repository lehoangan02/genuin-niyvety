#!/usr/bin/env python3
# ================================================================
# Split label.txt into train/val text files only (no image copying)
# ================================================================

import os
import random

# ===== CONFIG =====
LABEL_PATH = "label.txt"        # path to label.txt
OUTPUT_DIR = "dataset"          # output folder
SPLIT_RATIO = 0.8               # 80% train, 20% val
SEED = 42                       # random seed for reproducibility
# ===================

random.seed(SEED)

# Read all lines
with open(LABEL_PATH, "r") as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]

# Group lines by video ID
videos = {}
for line in lines:
    parts = line.split()
    if len(parts) < 10:
        continue
    video_id = parts[0]
    videos.setdefault(video_id, []).append(line)

video_ids = list(videos.keys())
random.shuffle(video_ids)

# Split by ratio
split_index = int(len(video_ids) * SPLIT_RATIO)
train_videos = video_ids[:split_index]
val_videos = video_ids[split_index:]

print(f"Total videos: {len(video_ids)}")
print(f"→ Train: {len(train_videos)} videos")
print(f"→ Val:   {len(val_videos)} videos")

# Prepare output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

train_txt = os.path.join(OUTPUT_DIR, "label_train.txt")
val_txt = os.path.join(OUTPUT_DIR, "label_val.txt")

train_lines, val_lines = [], []

# Split lines into train/val sets
for vid, vlines in videos.items():
    if vid in train_videos:
        train_lines.extend(vlines)
    else:
        val_lines.extend(vlines)

# Write results
with open(train_txt, "w") as f:
    f.write("\n".join(train_lines))
with open(val_txt, "w") as f:
    f.write("\n".join(val_lines))

print(f"\n✅ Done! Labels saved in '{OUTPUT_DIR}/'")
print(f"   - Train labels: {len(train_lines)} lines")
print(f"   - Val labels:   {len(val_lines)} lines")

# make the test_list.txt file that is the same as val file but without labels
test_txt = os.path.join(OUTPUT_DIR, "test_list.txt")
with open(test_txt, "w") as f:
    for line in val_lines:
        parts = line.split()
        test_line = " ".join(parts[:5])  # keep only video_id, q1, q2, q3, frame_name
        f.write(test_line + "\n")
