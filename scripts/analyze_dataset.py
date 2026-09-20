# -*- coding: utf-8 -*-
"""
数据集目标统计分析（论文改进动机依据）
统计内容：
  1. 目标框归一化宽/高分布（验证小目标占比）
  2. 目标面积分布（按 COCO 小/中/大目标标准分类）
  3. 长宽比分布（验证细长形目标占比）
  4. 每图目标数分布（目标密集度）
输出：E:/YOLO/analysis/ 下的 PNG 图表 + 统计摘要 analysis_summary.txt
"""
from pathlib import Path
import collections

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
LABEL_DIRS = [ROOT / "datasets/tunnel/labels/train", ROOT / "datasets/tunnel/labels/val"]
OUT = ROOT / "analysis"
OUT.mkdir(exist_ok=True)

widths, heights, areas, ratios, per_img = [], [], [], [], []
for d in LABEL_DIRS:
    for f in d.glob("*.txt"):
        n = 0
        for line in f.read_text().strip().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            w, h = float(p[3]), float(p[4])
            if w <= 0 or h <= 0:
                continue
            widths.append(w); heights.append(h)
            areas.append(w * h)
            ratios.append(w / h if h > 0 else 0)
            n += 1
        if n:
            per_img.append(n)

N = len(areas)
small = sum(1 for a in areas if a < 0.01)          # COCO: area < 32^2 px ≈ 0.0025 @640px，此处用归一化面积近似
mid = sum(1 for a in areas if 0.01 <= a < 0.1)
large = sum(1 for a in areas if a >= 0.1)
tiny_wh = sum(1 for w, h in zip(widths, heights) if w < 0.2 and h < 0.2)
elong = sum(1 for r in ratios if r >= 3 or r <= 1/3)

lines = []
lines.append(f"目标框总数: {N}")
lines.append(f"每图目标数: 均值 {sum(per_img)/len(per_img):.2f}, 最大 {max(per_img)}, 单目标图占比 {sum(1 for x in per_img if x==1)/len(per_img)*100:.1f}%")
lines.append(f"面积分布(归一化): 小(<0.01) {small} ({small/N*100:.1f}%) | 中(0.01-0.1) {mid} ({mid/N*100:.1f}%) | 大(>=0.1) {large} ({large/N*100:.1f}%)")
lines.append(f"宽高均 <0.2 的框: {tiny_wh} ({tiny_wh/N*100:.1f}%)  ← 小目标证据")
lines.append(f"长宽比 >=3 或 <=1/3: {elong} ({elong/N*100:.1f}%)  ← 细长形证据")
lines.append(f"宽中位数 {sorted(widths)[N//2]:.3f}, 高中位数 {sorted(heights)[N//2]:.3f}")

plt.rcParams["font.family"] = ["Microsoft YaHei"]
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Tunnel Leakage Dataset - Target Statistics (train+val)", fontsize=14)

axes[0,0].hist(widths, bins=40, color="#4C78A8", edgecolor="white")
axes[0,0].axvline(0.2, color="red", ls="--", lw=1, label="w=0.2")
axes[0,0].set_title("Normalized Width"); axes[0,0].set_xlabel("w (norm)"); axes[0,0].legend()

axes[0,1].hist(heights, bins=40, color="#F58518", edgecolor="white")
axes[0,1].axvline(0.2, color="red", ls="--", lw=1, label="h=0.2")
axes[0,1].set_title("Normalized Height"); axes[0,1].set_xlabel("h (norm)"); axes[0,1].legend()

axes[1,0].hist(areas, bins=40, color="#54A24B", edgecolor="white")
axes[1,0].axvline(0.01, color="red", ls="--", lw=1, label="area=0.01 (small)")
axes[1,0].set_title("Normalized Area"); axes[1,0].set_xlabel("w*h"); axes[1,0].set_yscale("log"); axes[1,0].legend()

axes[1,1].hist([min(r, 10) for r in ratios], bins=40, color="#B279A2", edgecolor="white")
axes[1,1].axvline(3, color="red", ls="--", lw=1, label="ratio=3")
axes[1,1].set_title("Aspect Ratio (capped at 10)"); axes[1,1].set_xlabel("w/h"); axes[1,1].legend()

plt.tight_layout()
plt.savefig(OUT / "target_statistics.png", dpi=150)
lines.append("图表已保存: analysis/target_statistics.png")

(OUT / "analysis_summary.txt").write_text("\n".join(lines), encoding="utf-8")
print("DONE")
