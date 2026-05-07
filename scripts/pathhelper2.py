from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description="Rename JSON files by trimming a suffix.")
parser.add_argument("--root", type=str, default=".", help="Root directory to search for JSON files.")
parser.add_argument("--suffix", type=str, default="annotation", help="Suffix to trim")
args = parser.parse_args()
root = Path(args.root)
suffix = args.suffix

tail = f"_{suffix}.json"
for path in root.rglob(f"*{tail}"):
    new_path = path.with_name(path.name[:-len(tail)] + ".json")
    if new_path.exists():
        print(f"skipping, already exists:{new_path}")
        continue
    path.rename(new_path)
    print(f"Renamed: {path} -> {new_path}")

