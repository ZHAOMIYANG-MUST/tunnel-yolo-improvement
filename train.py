# 基线训练脚本：YOLOv8n 隧道病害检测
# 用法: python train.py
from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")  # 自动下载 COCO 预训练权重
    model.train(
        data="configs/tunnel.yaml",
        epochs=300,        # 冒烟测试时改为 5
        imgsz=640,
        batch=16,          # 显存不足报错时降到 8 或 4
        seed=42,           # 固定随机种子，保证消融实验可复现
        device=0,          # 无 GPU 用 "cpu"（仅适合冒烟测试）
        workers=4,         # Windows 下若报 DataLoader 错误改为 0
        project="runs",
        name="baseline",
        exist_ok=True,
    )
    # 训练完成后在验证集上评估并打印指标
    metrics = model.val()
    print("\n===== 基线指标（写入实验记录）=====")
    print(f"Precision   : {metrics.box.mp:.4f}")
    print(f"Recall      : {metrics.box.mr:.4f}")
    print(f"mAP@0.5     : {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")

if __name__ == "__main__":
    main()
