# check if there is any duplicate lines in a txt file and show which lines are duplicated
def has_duplicate_lines(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        unique_lines = set(lines)
        if len(lines) != len(unique_lines):
            print("Duplicate lines found:")
            for line in unique_lines:
                if lines.count(line) > 1:
                    print(f"'{line.strip()}' - {lines.count(line)} times")
            return True
        return False
# Example usage:
# file_path = 'path/to/your/file.txt'
# print(has_duplicate_lines(file_path))

if __name__ == "__main__":
    file_path = 'label.txt'  # Replace with your file path
    if has_duplicate_lines(file_path):
        print("The file has duplicate lines.")
    else:
        print("The file has no duplicate lines.")