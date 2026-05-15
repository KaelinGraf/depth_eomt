#!/usr/bin/env python3
"""
Prune frame_* directories that are missing required files.

A frame is considered complete when it contains:
  - rgb.png
  - a *instance_raw*.{png,jpg,jpeg} file
  - a *scene_info*.json file

Usage:
    python3 clean_dataset.py /path/to/dataset_root             # dry-run, report only
    python3 clean_dataset.py /path/to/dataset_root --delete    # actually remove
    python3 clean_dataset.py /path/to/dataset_root --delete -v # verbose
"""

import argparse
import shutil
import sys
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg"}


def missing_files(frame_dir: Path) -> list[str]:
    if not frame_dir.is_dir():
        return ["<not a directory>"]

    files = list(frame_dir.iterdir())
    missing = []

    if not any(f.name == "rgb.png" for f in files):
        missing.append("rgb.png")
    if not any("instance_raw" in f.name and f.suffix in IMG_EXTS for f in files):
        missing.append("*instance_raw*.{png,jpg,jpeg}")
    if not any("scene_info" in f.name and f.suffix == ".json" for f in files):
        missing.append("*scene_info*.json")

    return missing


def scan(split_dir: Path):
    frames = sorted(p for p in split_dir.rglob("frame_*") if p.is_dir())
    bad = [(f, m) for f in frames if (m := missing_files(f))]
    return frames, bad


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_root", type=Path, help="Path containing train/ and val/ subdirs")
    parser.add_argument("--delete", action="store_true", help="Actually delete incomplete frames (default: dry-run)")
    parser.add_argument("-v", "--verbose", action="store_true", help="List every kept frame too")
    args = parser.parse_args()

    root: Path = args.dataset_root
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        sys.exit(1)

    splits = [d for d in ("train", "val") if (root / d).is_dir()]
    if not splits:
        print(f"note: no train/ or val/ under {root}; scanning the root directly.")
        splits = ["."]

    total_frames = 0
    total_bad = 0

    for split in splits:
        split_dir = root if split == "." else root / split
        frames, bad = scan(split_dir)

        print(f"\n[{split}] {len(bad)} incomplete / {len(frames)} total")
        if args.verbose:
            bad_set = {p for p, _ in bad}
            for f in frames:
                if f not in bad_set:
                    print(f"  ok   {f}")
        for frame, missing in bad:
            print(f"  miss {frame}  (missing: {', '.join(missing)})")

        if args.delete:
            for frame, _ in bad:
                shutil.rmtree(frame)
            if bad:
                print(f"  deleted {len(bad)} frames")

        total_frames += len(frames)
        total_bad += len(bad)

    print(f"\nsummary: {total_bad} incomplete / {total_frames} total")
    if total_bad and not args.delete:
        print("dry-run only — re-run with --delete to remove them.")


if __name__ == "__main__":
    main()
