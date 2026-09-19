# 数据集划分脚本：将原始数据按 8:1:1 划分为 train/val/test
# 用法:
#   1. 把下载的数据集解压，确保结构为:
#      raw_dataset/images/*.jpg
#      raw_dataset/labels/*.txt
#   2. python scripts/split_dataset.py
import random
import shutil
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "raw_dataset"
DST = Path(__file__).resolve().parent.parent / "datasets" / "tunnel"
RATIO = (0.8, 0.1, 0.1)  # train / val / test
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def main():
    imgs = sorted(p for p in (SRC / "images").iterdir() if p.suffix.lower() in IMG_EXTS)
    if not imgs:
        raise SystemExit(f"[错误] 在 {SRC/'images'} 下没找到图片，请检查解压路径")

    random.seed(42)  # 固定种子，保证每次划分一致
    random.shuffle(imgs)

    n = len(imgs)
    splits = {
        "train": imgs[: int(n * RATIO[0])],
        "val": imgs[int(n * RATIO[0]) : int(n * (RATIO[0] + RATIO[1]))],
        "test": imgs[int(n * (RATIO[0] + RATIO[1])) :],
    }

    missing_label = 0
    for name, files in splits.items():
        img_dir = DST / "images" / name
        lbl_dir = DST / "labels" / name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(f, img_dir / f.name)
            lbl = SRC / "labels_bbox" / (f.stem + ".txt")  # 用转换后的检测框标注
            if lbl.exists():
                shutil.copy2(lbl, lbl_dir / lbl.name)
            else:
                missing_label += 1  # 无标注=纯背景图，YOLO 允许

    print(f"总数 {n} | train {len(splits['train'])} | val {len(splits['val'])} | test {len(splits['test'])}")
    print(f"无标注图片 {missing_label} 张（作为背景图保留）")
    print(f"已输出到: {DST}")


if __name__ == "__main__":
    main()
