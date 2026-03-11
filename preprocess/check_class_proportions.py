"""Print class (urlId) proportions for each split of the supervised dataset.

Usage:
    python preprocess/check_class_proportions.py [--pc] [--phone]
    (defaults to --pc if neither is given)
"""
import argparse
import json
import pathlib
import sys
from collections import Counter


def build_label_map(json_dir: pathlib.Path) -> dict:
    """Return {sessionId_i: urlId} by parsing only the 'loms' key of each JSON."""
    label_map: dict = {}
    json_files = list(json_dir.glob("*.json"))
    print(f"Parsing {len(json_files)} JSON files in {json_dir} …", flush=True)
    for jf in json_files:
        try:
            # Use raw text to avoid loading large 'timeline' key into memory
            data = json.loads(jf.read_bytes())
        except Exception as e:
            print(f"  [WARN] Could not parse {jf.name}: {e}", file=sys.stderr)
            continue
        loms = data.get("loms", {})
        if not isinstance(loms, dict):
            continue
        for i, lom in enumerate(loms.values()):
            uid = lom.get("urlId")
            if uid is not None:
                label_map[f"{jf.stem}_{i}"] = uid
    print(f"  → {len(label_map)} label entries built.\n", flush=True)
    return label_map


def count_split(png_dir: pathlib.Path, label_map: dict):
    counts: Counter = Counter()
    unknown = 0
    for p in png_dir.glob("*.png"):
        lbl = label_map.get(p.stem)
        if lbl:
            counts[lbl] += 1
        else:
            unknown += 1
    return counts, unknown


def print_split(split_name: str, counts: Counter, unknown: int):
    total = sum(counts.values()) + unknown
    if total == 0:
        print(f"  (empty)\n")
        return
    bar_width = 35
    print(f"{'Class':<22} {'N':>6}  {'%':>6}  Distribution")
    print("-" * 72)
    for cls, n in sorted(counts.items(), key=lambda x: -x[1]):
        bar = "█" * int(bar_width * n / total)
        print(f"  {cls:<20} {n:>6}  {100*n/total:>5.1f}%  {bar}")
    if unknown:
        print(f"  {'(no label)':<20} {unknown:>6}  {100*unknown/total:>5.1f}%")
    print(f"\n  Total: {total} images\n")


def analyse(json_dir: pathlib.Path, png_base: pathlib.Path, device: str):
    print(f"\n{'='*72}")
    print(f"  Device: {device.upper()}")
    print(f"  JSON:   {json_dir}")
    print(f"  PNGs:   {png_base}")
    print(f"{'='*72}\n")

    if not json_dir.exists():
        print(f"[ERROR] JSON dir not found: {json_dir}", file=sys.stderr)
        return
    if not png_base.exists():
        print(f"[ERROR] PNG base dir not found: {png_base}", file=sys.stderr)
        return

    label_map = build_label_map(json_dir)

    for split in ["train", "validation", "test"]:
        png_dir = png_base / split
        if not png_dir.exists():
            continue
        counts, unknown = count_split(png_dir, label_map)
        total = sum(counts.values()) + unknown
        print(f"── {split.upper()}  ({total} images) " + "─" * 40)
        print_split(split, counts, unknown)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pc", action="store_true", default=False)
    parser.add_argument("--phone", action="store_true", default=False)
    args = parser.parse_args()

    root = pathlib.Path(__file__).parent.parent

    # Default to PC if nothing specified
    if not args.pc and not args.phone:
        args.pc = True

    if args.pc:
        analyse(
            json_dir=root / "dataset_supervised" / "dataset_supervised_pc" / "json",
            png_base=root / "dataset_supervised" / "dataset_supervised_pc" / "png_scaled",
            device="pc",
        )

    if args.phone:
        analyse(
            json_dir=root / "dataset_supervised" / "dataset_supervised_phone" / "json",
            png_base=root / "dataset_supervised" / "dataset_supervised_phone" / "png_scaled",
            device="phone",
        )


if __name__ == "__main__":
    main()
