"""结构诊断脚本：对比官方 yolov8n 与改进结构的逐层参数，定位差异来源。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plugins import register_attention  # noqa: E402

register_attention(verbose=False)

from ultralytics.nn.tasks import DetectionModel, yaml_model_load  # noqa: E402

OFFICIAL = ROOT / "venv" / "Lib" / "site-packages" / "ultralytics" / "cfg" / "models" / "v8" / "yolov8.yaml"

cfg = dict(yaml_model_load(OFFICIAL))
cfg["scale"] = "n"
base = DetectionModel(cfg, nc=1, verbose=False)

ours = DetectionModel(yaml_model_load(ROOT / "configs" / "yolov8n-cbam.yaml"), nc=1, verbose=False)

LAYER_MAP = {**{i: i for i in range(10)}, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15, 15: 16, 16: 18, 17: 19, 18: 20, 19: 22, 20: 23, 21: 24, 22: 26}


def layers(model):
    out = {}
    for i, layer in enumerate(model.model):
        out[i] = (layer.type, sum(p.numel() for p in layer.parameters()))
    return out


b, o = layers(base), layers(ours)
print(f"官方 yolov8n(nc=1) 总参数 {sum(p.numel() for p in base.parameters()):,}")
print(f"改进结构(nc=1)    总参数 {sum(p.numel() for p in ours.parameters()):,}")
print()
print(f"{'官方':>4} {'类型':<12}{'参数':>9}   ->  {'改进':>4} {'类型':<12}{'参数':>9}   一致")
for j in sorted(b):
    k = LAYER_MAP.get(j)
    bt, bp = b[j]
    if k is None or k not in o:
        print(f"{j:>4} {bt:<12}{bp:>9}   ->   (无对应层)")
        continue
    ot, op = o[k]
    flag = "OK" if bp == op else f"差异 {op - bp:+d}"
    print(f"{j:>4} {bt:<12}{bp:>9}   ->  {k:>4} {ot:<12}{op:>9}   {flag}")

extra = [i for i in sorted(o) if i not in LAYER_MAP.values()]
print()
print("改进结构独有层：", [(i, o[i]) for i in extra])
