import os
import shutil

base_dir = "DATA/samples"
output_dir = "DATA/images"
os.makedirs(output_dir, exist_ok=True)

for subfolder in os.listdir(base_dir):
    subfolder_path = os.path.join(base_dir, subfolder, "object_images")
    if os.path.isdir(subfolder_path):
        for idx, file_name in enumerate(sorted(os.listdir(subfolder_path)), start=1):
            file_path = os.path.join(subfolder_path, file_name)
            if os.path.isfile(file_path):
                ext = os.path.splitext(file_name)[1]  # keep original extension
                new_name = f"{subfolder}_Img{idx}{ext}"
                shutil.move(file_path, os.path.join(output_dir, new_name))

print("All images renamed and moved to DATA/images")
