# 标注格式转换：YOLO 分割多边形 → YOLO 检测框
# 本数据集 labels 是 "类别 x1 y1 x2 y2 x3 y3 ..." 的多边形轮廓，
# 检测任务只需要外接矩形 (xmin ymin w h)，取多边形坐标的 min/max 即可。
# 原始多边形标注保留在 labels/，转换结果输出到 labels_bbox/
import glob
import os
from pathlib import Path

SRC_DIR = Path("E:/YOLO/raw_dataset/labels")
DST_DIR = Path("E:/YOLO/raw_dataset/labels_bbox")

def main():
    DST_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(SRC_DIR.glob("*.txt"))
    total_boxes = 0
    skipped = []
    for f in files:
        lines_out = []
        for line in f.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = parts[0]
            vals = list(map(float, parts[1:]))
            if len(vals) % 2 != 0:
                vals = vals[:-1]  # 奇数个坐标时丢弃末尾
            xs = vals[0::2]
            ys = vals[1::2]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            w, h = xmax - xmin, ymax - ymin
            if w <= 0 or h <= 0:
                continue
            cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
            lines_out.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            total_boxes += 1
        if lines_out:
            (DST_DIR / f.name).write_text("\n".join(lines_out))
        else:
            skipped.append(f.name)
    print(f"转换文件数: {len(files) - len(skipped)} | 检测框总数: {total_boxes} | 无有效框: {len(skipped)}")

if __name__ == "__main__":
    main()
