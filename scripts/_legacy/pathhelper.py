from pathlib import Path
#take argument from command line for root path, default to current directory
import argparse
parser = argparse.ArgumentParser(description="Rename JSON files by adding a prefix.")
parser.add_argument("--root", type=str, default=".", help="Root directory to search for JSON files.")
parser.add_argument("--prefix", type=str, default="cmuh", help="Prefix to add to JSON file names.")
args = parser.parse_args()
root = Path(args.root)
prefix = args.prefix

for path in root.rglob("*.json"):
    if path.name.startswith(prefix):
        #remove _ from "cmuh_", i.e. "cmuh_1234.json" -> "cmuh1234.json"
        new_name = path.name.replace("_", "", 1)
        new_path = path.with_name(new_name)
    else:
        new_path = path.with_name(prefix + path.name)
    if new_path.exists():
        print(f"skipping, already exists:{new_path}")
        continue
    path.rename(new_path)
    print(f"Renamed: {path} -> {new_path}")

