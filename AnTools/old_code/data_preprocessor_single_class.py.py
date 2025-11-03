import json
import os
import cv2  # OpenCV for video processing
import random
import shutil

# --- Configuration ---

# 1. Base directory of your dataset (where 'annotations' and 'samples' are)
BASE_DIR = "./train" # Current directory

# 2. Input directories
ANNOTATIONS_FILE = os.path.join(BASE_DIR, "annotations", "annotations.json")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")

# 3. Output directory for the new YOLOv8 dataset
OUTPUT_DIR = os.path.join(BASE_DIR, "yolov8_dataset_single_class")

# 4. Train/Validation split ratio (e.g., 0.8 = 80% train, 20% val)
TRAIN_SPLIT_RATIO = 0.8

# 5. Define the single class
CLASS_ID = 0
CLASS_NAME = "object"

# --- Helper Functions ---

def normalize_bbox(x1, y1, x2, y2, W, H):
    """
    Converts absolute (x1, y1, x2, y2) coordinates to normalized YOLO format 
    (x_center, y_center, width, height).
    """
    w_abs = x2 - x1
    h_abs = y2 - y1
    x_center_abs = x1 + (w_abs / 2)
    y_center_abs = y1 + (h_abs / 2)

    x_center_norm = x_center_abs / W
    y_center_norm = y_center_abs / H
    w_norm = w_abs / W
    h_norm = h_abs / H

    # Clamp values to [0.0, 1.0]
    x_center_norm = max(0.0, min(1.0, x_center_norm))
    y_center_norm = max(0.0, min(1.0, y_center_norm))
    w_norm = max(0.0, min(1.0, w_norm))
    h_norm = max(0.0, min(1.0, h_norm))

    return x_center_norm, y_center_norm, w_norm, h_norm

def setup_directories(base_output_dir):
    """
    Creates the required directory structure for YOLOv8.
    """
    if os.path.exists(base_output_dir):
        print(f"Warning: Output directory '{base_output_dir}' already exists. Deleting it.")
        shutil.rmtree(base_output_dir)
    
    os.makedirs(os.path.join(base_output_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(base_output_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(base_output_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(base_output_dir, "labels", "val"), exist_ok=True)
    print(f"Created directory structure at '{base_output_dir}'")

# --- Main Script ---

def main():
    print(f"Starting YOLOv8 dataset preprocessing for a SINGLE class: '{CLASS_NAME}'")
    setup_directories(OUTPUT_DIR)

    # 1. Discover video paths
    video_paths = {}

    print(f"Scanning for videos in '{SAMPLES_DIR}'...")
    for video_id in os.listdir(SAMPLES_DIR):
        video_folder_path = os.path.join(SAMPLES_DIR, video_id)
        if os.path.isdir(video_folder_path):
            video_file_path = os.path.join(video_folder_path, "drone_video.mp4")
            if os.path.exists(video_file_path):
                video_paths[video_id] = video_file_path
            else:
                print(f"Warning: 'drone_video.mp4' not found in {video_folder_path}")

    if not video_paths:
        print(f"Error: No video files found in {SAMPLES_DIR}. Exiting.")
        return
    
    print(f"Found {len(video_paths)} videos to process.")

    # 2. Load annotations
    print(f"Loading annotations from '{ANNOTATIONS_FILE}'...")
    try:
        with open(ANNOTATIONS_FILE, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Annotation file not found at {ANNOTATIONS_FILE}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {ANNOTATIONS_FILE}")
        return

    # Ensure data is a list of records
    if not isinstance(data, list):
        if 'video_id' in data and 'annotations' in data:
            data = [data]
        else:
            print("Error: Unknown JSON structure. Expected a list of video records.")
            return

    total_frames_saved = 0
    total_labels_saved = 0

    # 3. Process each video
    for video_record in data:
        video_id = video_record.get("video_id")
        if not video_id:
            print("Skipping record with missing 'video_id'")
            continue

        if video_id not in video_paths:
            print(f"Warning: video_id '{video_id}' from JSON not found in 'samples' directory. Skipping.")
            continue

        video_file_path = video_paths[video_id]
        
        # All annotations will use the same CLASS_ID
        print(f"\nProcessing video: {video_id} (All detections will be Class ID: {CLASS_ID})")

        # 4. Build a quick-lookup map for frames in this video
        frame_annotation_map = {}
        for interval in video_record.get("annotations", []):
            for bbox in interval.get("bboxes", []):
                frame_num = bbox.get("frame")
                if frame_num is not None:
                    if frame_num not in frame_annotation_map:
                        frame_annotation_map[frame_num] = []
                    frame_annotation_map[frame_num].append(bbox)
        
        if not frame_annotation_map:
            print("...No annotations found for this video. Skipping.")
            continue

        # 5. Open video and get dimensions
        cap = cv2.VideoCapture(video_file_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_file_path}")
            continue
        
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if W == 0 or H == 0:
            print(f"Error: Could not get dimensions for {video_file_path}. Skipping.")
            cap.release()
            continue

        print(f"...Video dimensions: {W}x{H}")

        # 6. Extract frames, normalize labels, and save
        frame_count = -1
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break # End of video
            
            frame_count += 1
            
            # Check if this frame has an annotation
            if frame_count in frame_annotation_map:
                bboxes = frame_annotation_map[frame_count]
                
                # Decide if this frame goes to train or val
                set_folder = "train" if random.random() < TRAIN_SPLIT_RATIO else "val"
                
                # Define file paths
                img_name = f"{video_id}_frame_{frame_count:05d}.jpg"
                label_name = f"{video_id}_frame_{frame_count:05d}.txt"
                
                img_path = os.path.join(OUTPUT_DIR, "images", set_folder, img_name)
                label_path = os.path.join(OUTPUT_DIR, "labels", set_folder, label_name)
                
                # Save the image
                cv2.imwrite(img_path, frame)
                total_frames_saved += 1
                
                # Save the label file
                yolo_lines = []
                for bbox in bboxes:
                    x1, y1 = bbox.get("x1"), bbox.get("y1")
                    x2, y2 = bbox.get("x2"), bbox.get("y2")
                    
                    if not all([x1 is not None, y1 is not None, x2 is not None, y2 is not None]):
                        print(f"Warning: Corrupt bbox data for {video_id} frame {frame_count}")
                        continue

                    # Normalize
                    x_c, y_c, w, h = normalize_bbox(x1, y1, x2, y2, W, H)
                    
                    # Format: [class_id] [x_center] [y_center] [width] [height]
                    # We use the global CLASS_ID (0) for everything.
                    yolo_lines.append(f"{CLASS_ID} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
                    total_labels_saved += 1

                if yolo_lines:
                    with open(label_path, 'w') as f:
                        f.writelines(yolo_lines)
        
        cap.release()
        print(f"...Done processing {video_id}.")

    # 7. Generate dataset.yaml
    yaml_path = os.path.join(OUTPUT_DIR, "dataset.yaml")
    yaml_content = f"""
# Path to dataset root directory
path: {os.path.abspath(OUTPUT_DIR)}

# Train/val/test sets
train: images/train
val: images/val
# test: (optional)

# Classes
nc: 1
names: ['{CLASS_NAME}']
"""
    
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print("\n--- Preprocessing Complete ---")
    print(f"Total images saved: {total_frames_saved}")
    print(f"Total labels (bboxes) saved: {total_labels_saved}")
    print(f"Dataset YAML file created at: {yaml_path}")
    print("\nReady to train!")
    print(f"Use this command: yolo train model=yolov8n.pt data={yaml_path} epochs=100 imgsz=640")

if __name__ == "__main__":
    main()
