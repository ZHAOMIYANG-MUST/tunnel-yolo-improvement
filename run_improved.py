"""带自定义模块注册的训练/评估入口（改进实验专用）。

用法（在 E:\\YOLO 目录下执行）：

    # 训练（改进②：WIoU 损失 + CBAM 注意力）
    python run_improved.py train model=configs/yolov8n-cbam.yaml data=configs/tunnel.yaml \
        epochs=300 imgsz=640 batch=16 device=0 workers=4 seed=42 project=runs name=imp2_cbam

    # 测试集评估
    python run_improved.py val model=runs/detect/runs/imp2_cbam/weights/best.pt \
        data=configs/tunnel.yaml split=test imgsz=640 device=0 name=imp2_cbam_test

为什么要用它而不是 yolo 命令行：CBAM 是自定义模块，必须先把类注册进 ultralytics 的
命名空间，模型 yaml（构建时）与权重文件（反序列化时）才能被正确解析。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plugins import register_attention  # noqa: E402

register_attention()


def _cast(value: str):
    """把命令行里的字符串转成 int / float / bool / None / str。"""
    low = value.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"none", "null"}:
        return None
    for caster in (int, float):
        try:
            return caster(value)
        except ValueError:
            pass
    return value


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 1

    mode, items = sys.argv[1], sys.argv[2:]
    kwargs: dict = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            print(f"参数格式应为 key=value，收到：{item}")
            return 1
        kwargs[key] = _cast(value)

    from ultralytics import YOLO

    weights = kwargs.pop("model", None)
    if not weights:
        print("必须提供 model=...")
        return 1

    model = YOLO(weights)
    if mode == "train":
        model.train(**kwargs)
    elif mode == "val":
        model.val(**kwargs)
    elif mode == "predict":
        model.predict(**kwargs)
    else:
        print(f"未知模式：{mode}（支持 train / val / predict）")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
