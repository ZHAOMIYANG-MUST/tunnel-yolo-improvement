"""生成带自定义注意力模块的初始化权重，保证改进实验与基线的初始化条件一致。

基线用 ``model=yolov8n.pt``（COCO 预训练）。插入注意力模块改动了网络结构，官方权重
无法直接整体加载，因此这里按"层号重映射 + 同名同形状"逐层迁移主干与颈部权重
（检测头因类别数不同需重新初始化），另存一份初始化权重供训练使用——
这样改进实验与基线在"是否使用预训练"这一点上完全对齐，符合消融实验的唯一变量原则。

用法：
    python scripts/make_init_weights.py                              # 默认 CBAM 版
    python scripts/make_init_weights.py --cfg configs/yolov8n-ca.yaml --out weights/yolov8n-ca-init.pt
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plugins import register_attention  # noqa: E402

register_attention()

import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.tasks import load_checkpoint  # noqa: E402
from ultralytics.utils.torch_utils import intersect_dicts  # noqa: E402

PRETRAINED = ROOT / "yolov8n.pt"

# 新增注意力层在新结构里的层号（与本项目两个改进 yaml 的插入位置一致）
NEW_MODULE_LAYERS = (10, 17, 21, 25)
DETECT_LAYER = 26  # 检测头

# 官方 yolov8n 层号 -> 插入注意力后的层号。
# 插入模块会让后续层号整体后移，而权重是按 "model.<层号>.<参数名>" 命名的，
# 若不做重映射，颈部及检测头的预训练权重会因为“名字对不上”而全部丢失（初始化条件与基线不一致）。
LAYER_MAP = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 11,  # Upsample
    11: 12,  # Concat
    12: 13,  # C2f
    13: 14,  # Upsample
    14: 15,  # Concat
    15: 16,  # C2f  (P3)
    16: 18,  # Conv
    17: 19,  # Concat
    18: 20,  # C2f  (P4)
    19: 22,  # Conv
    20: 23,  # Concat
    21: 24,  # C2f  (P5)
    22: 26,  # Detect
}


def remap_layer_indices(state_dict: dict, mapping: dict) -> dict:
    """把权重里的层号按映射表改写，使官方 yolov8n 权重能对上新结构。"""
    remapped = {}
    for key, tensor in state_dict.items():
        parts = key.split(".")
        if len(parts) > 2 and parts[0] == "model" and parts[1].isdigit():
            target = mapping.get(int(parts[1]))
            if target is None:
                continue
            remapped[".".join(["model", str(target), *parts[2:]])] = tensor
        else:
            remapped[key] = tensor
    return remapped


def main() -> None:
    ap = argparse.ArgumentParser(description="生成改进模型的初始化权重")
    ap.add_argument("--cfg", default="configs/yolov8n-cbam.yaml", help="模型 yaml（相对项目根目录）")
    ap.add_argument("--out", default=None, help="输出权重路径（默认 weights/<cfg名>-init.pt）")
    args = ap.parse_args()

    cfg = ROOT / args.cfg
    if not cfg.exists():
        raise FileNotFoundError(f"找不到模型配置：{cfg}")
    out = ROOT / args.out if args.out else ROOT / "weights" / f"{cfg.stem}-init.pt"

    if not PRETRAINED.exists():
        raise FileNotFoundError(f"缺少预训练权重：{PRETRAINED}（首次运行会自动下载）")

    base, _ = load_checkpoint(str(PRETRAINED))

    model = YOLO(str(cfg)).model
    state = model.state_dict()
    remapped = remap_layer_indices(base.float().state_dict(), LAYER_MAP)
    transferred = intersect_dicts(remapped, state)
    model.load_state_dict(transferred, strict=False)

    new_prefixes = tuple(f"model.{i}." for i in NEW_MODULE_LAYERS)
    head_prefix = f"model.{DETECT_LAYER}."
    new_keys = [k for k in state if k.startswith(new_prefixes)]
    head_keys = [k for k in state if k.startswith(head_prefix)]
    other_missing = [
        k for k in state if k not in transferred and k not in head_keys and not k.startswith(new_prefixes)
    ]

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model}, out)

    print(f"模型结构        : {cfg.name}")
    print(f"总参数量        : {sum(p.numel() for p in model.parameters()):,}")
    print(f"权重迁移        : {len(transferred)}/{len(state)} 项（主干+颈部，来自 COCO 预训练）")
    print(f"检测头重新初始化: {len(head_keys)} 项（共享层已迁移，类别分支按 nc=1 重建）")
    print(f"新增模块随机init: {len(new_keys)} 项（注意力模块，无预训练权重）")
    print(f"其他未迁移      : {len(other_missing)} 项 {other_missing if other_missing else ''}")
    print(f"已保存          : {out}")


if __name__ == "__main__":
    main()
