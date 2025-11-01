import os
import shutil

base_dir = "./DATA/samples"
output_dir = "./DATA/frames"
os.makedirs(output_dir, exist_ok=True)

for subfolder in os.listdir(base_dir):
    subfolder_path = os.path.join(base_dir, subfolder, "frames")
    if os.path.isdir(subfolder_path):
        for file_name in os.listdir(subfolder_path):
            file_path = os.path.join(subfolder_path, file_name)
            if os.path.isfile(file_path):
                shutil.move(file_path, os.path.join(output_dir, file_name))

print("All frames moved to DATA/frames")
