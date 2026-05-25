"""
extract_frontal.py
Filters MTFL training.txt to keep only frontal images (pose == 1).

Usage:
    python extract_frontal.py <training.txt> <output.txt>
"""
import sys

def extract_frontal(input_path, output_path):
    count = 0
    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 14:
                continue
            pose = parts[-1]  # last field is pose
            if pose == '1':
                fout.write(stripped + '\n')
                count += 1
    print(f"Done. {count} frontal images written to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_frontal.py <training.txt> <output.txt>")
        sys.exit(1)
    extract_frontal(sys.argv[1], sys.argv[2])