input_file = "label_train.txt"
output_file = "test_list.txt"

with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
    for line in f_in:
        parts = line.strip().split()
        f_out.write(" ".join(parts[:5]) + "\n")
