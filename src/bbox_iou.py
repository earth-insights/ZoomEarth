#!/usr/bin/env python3
"""
Compute Intersection-over-Union (IoU) for a jsonl file.

Each jsonl line must contain:
  - "bbox_ref": [x1, y1, x2, y2]   # ground-truth box
  - "bbox":     [x1, y1, x2, y2]   # predicted box
  - "question_id": str or int      # only used to keep track of samples

Outputs:
  • Per-sample IoU printed (optional CSV dump)
  • Aggregate stats: mean IoU, median IoU,
    hit-rates at 0.25 / 0.50 / 0.75
"""

import json
import argparse
from pathlib import Path
from statistics import mean, median

def get_crop_area(bbox, min_size=512):
    """
    最终输出统一为 min_size × min_size：
    - 若 bbox 边长 < min_size → 在原图中扩展，必要时平移避免越界；
    - 若 bbox 边长 >= min_size → 对 bbox 区域缩放并中心裁剪。
    """
    x1, y1, x2, y2 = map(int, bbox)
    width, height = x2 - x1, y2 - y1

    if width < min_size or height < min_size:
        # 中心点
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # 初步计算边界
        new_x1 = center_x - min_size // 2
        new_y1 = center_y - min_size // 2
        new_x2 = new_x1 + min_size
        new_y2 = new_y1 + min_size

        # 最后确保框不越界
        new_x1 = max(0, new_x1)
        new_y1 = max(0, new_y1)

        return [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]

    else:
        return bbox

def _fix_order(box):
    """Ensure (x1, y1) is top-left, (x2, y2) bottom-right."""
    x1, y1, x2, y2 = box
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

def iou(box_a, box_b):
    """Compute IoU between two boxes given as [x1, y1, x2, y2]."""
    x1a, y1a, x2a, y2a = _fix_order(box_a)
    x1b, y1b, x2b, y2b = _fix_order(box_b)

    inter_x1, inter_y1 = max(x1a, x1b), max(y1a, y1b)
    inter_x2, inter_y2 = min(x2a, x2b), min(y2a, y2b)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, (x2a - x1a)) * max(0.0, (y2a - y1a))
    area_b = max(0.0, (x2b - x1b)) * max(0.0, (y2b - y1b))

    union = area_a + area_b - inter_area
    return 0.0 if union == 0 else inter_area / union

def parse_args():
    p = argparse.ArgumentParser(description="Compute IoU stats.")
    p.add_argument(
        "-j", "--jsonl",
        type=Path,
        default=Path("./results/lora_infer_backup.jsonl"),
        help="Input jsonl file (default: %(default)s)"
    )
    p.add_argument("-o", "--out_csv", type=Path,
                   help="Optional path to save per-sample IoU as CSV")
    return p.parse_args()

def main():
    args = parse_args()
    ious = []
    lines = args.jsonl.read_text().rstrip().splitlines()

    for ln, line in enumerate(lines, 1):
        try:
            data = json.loads(line)
            iou_val = iou(get_crop_area(data["bbox_ref"]), get_crop_area(data["bbox"]))
            ious.append((data["question_id"], iou_val))
        except Exception as e:
            print(f"[Line {ln}] skipped due to error: {e}")

    if not ious:
        print("No valid samples found.")
        return

    # --- Aggregate statistics ---
    vals = [v for _, v in ious]
    mean_iou   = mean(vals)
    median_iou = median(vals)
    hit_25 = sum(v >= 0.25 for v in vals) / len(vals)
    hit_50 = sum(v >= 0.50 for v in vals) / len(vals)
    hit_75 = sum(v >= 0.75 for v in vals) / len(vals)

    print(f"Samples : {len(vals)}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Median  : {median_iou:.4f}")
    print("Hit-rates:")
    print(f"  IoU ≥ 0.25 : {hit_25:.2%}")
    print(f"  IoU ≥ 0.50 : {hit_50:.2%}")
    print(f"  IoU ≥ 0.75 : {hit_75:.2%}")

    # --- Optional CSV dump ---
    if args.out_csv:
        import csv
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["question_id", "iou"])
            writer.writerows(ious)
        print(f"Per-sample IoU written to: {args.out_csv}")

if __name__ == "__main__":
    main()
