import random

def shuffle_file(input_path, output_path, seed=None):
    if seed is not None:
        random.seed(seed)

    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()  # keeps all lines without '\n'

    random.shuffle(lines)

    with open(output_path, 'w', encoding='utf-8') as f:
        for i, line in enumerate(lines):
            if i < len(lines) - 1:
                f.write(line + '\n')
            else:
                f.write(line)  # last line — no newline

if __name__ == "__main__":
    input_file = "label_train.txt"
    output_file = "shuffled.txt"
    shuffle_file(input_file, output_file)
    print(f"✅ Shuffled lines saved to {output_file}")
