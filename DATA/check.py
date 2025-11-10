# ===== CONFIG =====
input_file = "label_train.txt"        # your original txt file
output_file = "label_train.txt_filtered.txt"    # file to save filtered lines
# ===================

with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
    for line in f_in:
        # Skip empty lines
        if not line.strip():
            continue
        
        # Write only lines that do NOT contain '00000'
        if "0 0 0 0 0" not in line:
            f_out.write(line)
