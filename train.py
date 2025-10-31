import os
import torch  # Import torch to check for hardware
from ultralytics import YOLO

def start_training():
    """
    Loads the YOLO model and starts the training process using the
    dataset prepared by the data_preprocessor_single_class.py script.
    """
    
    # --- Configuration ---

    # 1. Path to the dataset.yaml file created by your preprocessor
    # This assumes train.py is in the same directory as 'yolov8_dataset_single_class'
    base_dir = "." # Current directory
    dataset_yaml_path = os.path.join(base_dir, "./train/yolov8_dataset_single_class", "dataset.yaml")

    # 2. Model selection
    # We use 'yolov8n.pt' (Nano) as it's the best choice for real-time
    # inference on NVIDIA Jetson devices.
    model_name = 'yolov8n.pt'

    # 3. Training parameters
    training_epochs = 150
    image_size = 640
    batch_size = 16 # Adjust this based on your GPU's VRAM (e.g., 8, 16, 32)
    
    # 4. Device selection
    # 'auto' was causing an error. We will now manually detect the best
    # available device in the correct priority order.
    if torch.cuda.is_available():
        device_to_use = "cuda"
    elif torch.backends.mps.is_available():
        device_to_use = "mps"
    else:
        device_to_use = "cpu"
    
    # --- End of Configuration ---
    
    # Check if the dataset configuration file exists
    if not os.path.exists(dataset_yaml_path):
        print(f"Error: Dataset YAML file not found at '{dataset_yaml_path}'")
        print("Please make sure you have run the 'data_preprocessor_single_class.py' script first.")
        return

    try:
        # Load the pre-trained model
        model = YOLO(model_name)
        
        print(f"Starting training with model: {model_name}")
        print(f"Using dataset: {dataset_yaml_path}")
        print(f"Epochs: {training_epochs}, Image Size: {image_size}, Batch Size: {batch_size}")
        print(f"Auto-detected device: {device_to_use}")

        # Start the training process
        model.train(
            data=dataset_yaml_path,
            epochs=training_epochs,
            imgsz=image_size,
            batch=batch_size,
            project="runs/train",  # Saves results to 'runs/train/exp'
            name="yolov8_drone_object_detector", # Subfolder name for this run
            device=device_to_use # Specify the device for training
        )
        
        print("\n--- Training Complete ---")
        print(f"Your trained model and results are saved in 'runs/train/yolov8_drone_object_detector'")
        print(f"The best model weights are at 'runs/train/yolov8_drone_object_detector/weights/best.pt'")

    except Exception as e:
        print(f"\nAn error occurred during training: {e}")

if __name__ == "__main__":
    # Ensure you have the 'ultralytics' and 'torch' packages installed:
    # pip install ultralytics torch
    start_training()

