"""漏检/误检样本分析：为下一项改进提供直接证据。

流程：
  1. 用指定权重在测试集上推理（conf=0.25）；
  2. 将预测框与 GT 按 IoU>=0.5 贪心匹配，统计 FN（漏检）与 FP（误检）；
  3. 输出：FN/FP 数量统计、FN 框的尺寸与长宽比分布（和全体 GT 对比）、
     FN 框在图片中的位置分布（判断是否集中在边缘）、最严重漏检图的可视化。

用法：
    PYTHONIOENCODING=utf-8 python scripts/missed_case_analysis.py --model <best.pt> [--out analysis]
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

IMG_DIR = ROOT / "datasets" / "tunnel" / "images" / "test"
LBL_DIR = ROOT / "datasets" / "tunnel" / "labels" / "test"


def load_gt(label_path: Path, w: int, h: int) -> np.ndarray:
    """读取 YOLO 归一化标签，返回像素坐标 xyxy (n,4)。空文件返回 (0,4) 数组。"""
    if not label_path.exists() or label_path.stat().st_size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    boxes = []
    for line in label_path.read_text().strip().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cx, cy, bw, bh = (float(v) for v in p[1:5])
        boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(boxes, dtype=np.float32)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (n,4), b: (m,4) -> (n,m) IoU。"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "runs" / "detect" / "runs" / "imp1_wiou" / "weights" / "best.pt"))
    ap.add_argument("--out", default=str(ROOT / "analysis" / "missed_cases"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--topk", type=int, default=8, help="可视化最严重漏检图的数量")
    args = ap.parse_args()

    from ultralytics import YOLO  # noqa: E402

    model = YOLO(args.model)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted(IMG_DIR.glob("*.jpg")) + sorted(IMG_DIR.glob("*.png"))
    total_gt = total_fn = total_fp = 0
    fn_shapes, all_shapes = [], []          # (w, h) 像素
    fn_cx, fn_cy = [], []                   # 归一化中心位置
    fp_confs, fp_shapes = [], []
    per_img = []                            # (img_name, n_fn, n_fp, n_gt)

    for img_path in imgs:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gts = load_gt(LBL_DIR / f"{img_path.stem}.txt", w, h)

        res = model.predict(source=img, conf=args.conf, imgsz=640, device=0, verbose=False)[0]
        preds = res.boxes.xyxy.cpu().numpy() if len(res.boxes) else np.zeros((0, 4), np.float32)
        confs = res.boxes.conf.cpu().numpy() if len(res.boxes) else np.zeros((0,), np.float32)

        ious = iou_matrix(gts, preds)
        gt_matched = np.zeros(len(gts), bool)
        pred_matched = np.zeros(len(preds), bool)
        if len(gts) and len(preds):
            for gi in np.argsort(-ious.max(axis=1)):
                pi = int(np.argmax(ious[gi]))
                if not pred_matched[pi] and ious[gi, pi] >= args.iou_thr:
                    gt_matched[gi] = pred_matched[pi] = True

        fn_idx = np.where(~gt_matched)[0]
        fp_idx = np.where(~pred_matched)[0]
        total_gt += len(gts)
        total_fn += len(fn_idx)
        total_fp += len(fp_idx)

        for b in gts:
            all_shapes.append((float(b[2] - b[0]), float(b[3] - b[1])))
        for gi in fn_idx:
            b = gts[gi]
            fn_shapes.append((float(b[2] - b[0]), float(b[3] - b[1])))
            fn_cx.append(float((b[0] + b[2]) / 2 / w))
            fn_cy.append(float((b[1] + b[3]) / 2 / h))
        for pi in fp_idx:
            fp_confs.append(float(confs[pi]))
            b = preds[pi]
            fp_shapes.append((float(b[2] - b[0]), float(b[3] - b[1])))
        per_img.append((img_path.name, len(fn_idx), len(fp_idx), len(gts)))

        # 前 topk 张最严重漏检图的伪标注可视化（延后统一排序后再画，见下）
        if len(per_img) <= 400:
            vis = img.copy()
            for gi, b in enumerate(gts):
                x1, y1, x2, y2 = map(int, b)
                color = (0, 220, 60) if gt_matched[gi] else (0, 165, 255)  # TP绿 / FN橙
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                tag = "GT" if gt_matched[gi] else "MISSED"
                cv2.putText(vis, tag, (x1, max(14, y1 - 4)), 0, 0.5, color, 1)
            for pi, b in enumerate(preds):
                if pred_matched[pi]:
                    continue
                x1, y1, x2, y2 = map(int, b)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)  # FP红
                cv2.putText(vis, f"FP {confs[pi]:.2f}", (x1, max(14, y1 - 4)), 0, 0.5, (0, 0, 255), 1)
            cv2.imwrite(str(out_dir / f"_{img_path.name}"), vis)

    # 可视化图：按 FN 数排序，把最严重的 topk 张重命名（不删除其余图，避免批量删除保护拦截）
    per_img.sort(key=lambda t: (-t[1], -t[2]))
    for name, nfn, nfp, ngt in per_img[: args.topk]:
        f = out_dir / f"_{name}"
        if f.exists():
            f.rename(out_dir / f"top_{nfn}FN_{nfp}FP_{name}")

    # FN 归因统计
    def shape_bucket(bw, bh, W=640):
        area = bw * bh / (W * W)
        if area < 0.01:
            return "small(<1%)"
        if area < 0.1:
            return "mid(1-10%)"
        return "large(>=10%)"

    def aspect_bucket(bw, bh):
        r = bw / max(bh, 1e-6)
        if r > 3:
            return "wide(>3)"
        if r < 1 / 3:
            return "tall(<1/3)"
        return "normal"

    def dist(values, buckets):
        c = Counter(buckets(*v) for v in values)
        n = max(len(values), 1)
        return {k: round(c.get(k, 0) / n * 100, 1) for k in buckets.__defaults__ or ["small(<1%)", "mid(1-10%)", "large(>=10%)"]}

    def aspect_dist(values):
        c = Counter(aspect_bucket(*v) for v in values)
        n = max(len(values), 1)
        return {k: round(c.get(k, 0) / n * 100, 1) for k in ["tall(<1/3)", "normal", "wide(>3)"]}

    report = {
        "model": args.model,
        "conf_threshold": args.conf,
        "iou_threshold": args.iou_thr,
        "test_images": len(imgs),
        "total_gt_boxes": total_gt,
        "total_FN": total_fn,
        "total_FP": total_fp,
        "FN_rate_pct": round(total_fn / max(total_gt, 1) * 100, 2),
        "FP_per_img": round(total_fp / max(len(imgs), 1), 3),
        "FN_area_dist_pct": dist(fn_shapes, shape_bucket),
        "ALL_area_dist_pct": dist(all_shapes, shape_bucket),
        "FN_aspect_dist_pct": aspect_dist(fn_shapes),
        "ALL_aspect_dist_pct": aspect_dist(all_shapes),
        "FN_center_x_mean": round(float(np.mean(fn_cx)), 3) if fn_cx else None,
        "FN_center_y_mean": round(float(np.mean(fn_cy)), 3) if fn_cy else None,
        "FP_conf_mean": round(float(np.mean(fp_confs)), 3) if fp_confs else None,
        "worst_images": per_img[: args.topk],
    }
    (ROOT / "analysis" / "missed_analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))

    print(f"测试图: {len(imgs)}  GT框: {total_gt}  FN: {total_fn} ({report['FN_rate_pct']}%)  FP: {total_fp}")
    print(f"FN 面积分布:   {report['FN_area_dist_pct']}   (全体GT: {report['ALL_area_dist_pct']})")
    print(f"FN 长宽比分布: {report['FN_aspect_dist_pct']}   (全体GT: {report['ALL_aspect_dist_pct']})")
    print(f"FN 中心位置均值: x={report['FN_center_x_mean']}, y={report['FN_center_y_mean']} (0~1, 0.5=居中)")
    print(f"FP 平均置信度: {report['FP_conf_mean']}")
    print("最严重漏检图:", [(n, f) for n, f, _, _ in per_img[: args.topk]])
    print(f"可视化已保存: {out_dir}")


if __name__ == "__main__":
    main()
