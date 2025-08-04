#!/usr/bin/env python3
"""
Compute width/height statistics for bbox_ref and bbox in a jsonl dataset.

Each jsonl line must contain:
  - "bbox_ref": [x1, y1, x2, y2]
  - "bbox"    : [x1, y1, x2, y2]
  - (other fields are ignored)

Outputs summary stats to stdout.

Usage examples
--------------
# 用默认文件
python bbox_stats.py

# 指定文件
python bbox_stats.py ./result/new_run.jsonl
"""

import argparse
from pathlib import Path
import json
import numpy as np
from textwrap import indent

# ------------------------- helpers ---------------------------------
def wh_from_bbox(bbox):
    """Return (width, height) given [x1,y1,x2,y2]."""
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1), max(0.0, y2 - y1)

def describe(arr, name):
    """Pretty-print statistics for a 1-D array."""
    if len(arr) == 0:
        print(f"{name}: 0 samples")
        return
    stats = {
        "samples": len(arr),
        "mean"   : np.mean(arr),
        "std"    : np.std(arr, ddof=1),
        "min"    : np.min(arr),
        "25%"    : np.percentile(arr, 25),
        "50%"    : np.median(arr),
        "75%"    : np.percentile(arr, 75),
        "max"    : np.max(arr)
    }
    print(f"{name}:")
    for k, v in stats.items():
        print(f"  {k:<6} = {v:8.3f}")
    print()

# ------------------------- main ------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="BBox width/height stats")
    ap.add_argument(
        "jsonl",
        type=Path,
        nargs="?",
        default=Path("./results/lora_infer_backup.jsonl"),
        help="Input jsonl file (default: %(default)s)"
    )
    return ap.parse_args()

def main():
    args = parse_args()
    w_ref, h_ref = [], []
    w_pred, h_pred = [], []

    for ln, line in enumerate(args.jsonl.read_text().splitlines(), 1):
        try:
            data = json.loads(line)
            wr, hr = wh_from_bbox(data["bbox_ref"])
            wp, hp = wh_from_bbox(data["bbox"])
            w_ref.append(wr);  h_ref.append(hr)
            w_pred.append(wp); h_pred.append(hp)
        except Exception as e:
            print(f"* Skip line {ln}: {e}")

    print("\n=== bbox_ref (GT) ===")
    describe(np.array(w_ref), "Width")
    describe(np.array(h_ref), "Height")

    print("=== bbox (Prediction) ===")
    describe(np.array(w_pred), "Width")
    describe(np.array(h_pred), "Height")

if __name__ == "__main__":
    main()
