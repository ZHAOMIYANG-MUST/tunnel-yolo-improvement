# 源码补丁说明

本仓库的 WIoU v3 损失改进通过修改 ultralytics 包源码实现（而非独立模块），补丁与备份在此存档，保证可复现。

## 文件

- `backup/loss.py.orig` —— 修改前的原始文件（ultralytics 8.4.155 `ultralytics/utils/loss.py`）
- `wiou_loss.patch` —— WIoU v3 修改的 unified diff

## 应用方法

在装有 **ultralytics 8.4.155** 的环境里执行：

```bash
cd <venv>/Lib/site-packages/ultralytics/utils/
cp loss.py loss.py.bak          # 先备份
patch loss.py < 项目根/docs/patches/wiou_loss.patch
```

或在 Linux/macOS 下（路径为 `venv/lib/python3.x/site-packages/ultralytics/utils/`）。

> 若你的 ultralytics 版本不同，`BboxLoss.forward` 的实现可能不一致，补丁不能直接套用——
> 此时请参照 patch 内容手动修改：核心是在 `BboxLoss.forward` 中按环境变量
> `YOLO_IOU_TYPE=wiou` 切换损失计算，WIoU v3 公式见实验记录-消融实验.md。

## 使用

- 默认（不设环境变量）：行为与原版完全一致（CIoU），基线可复现
- `YOLO_IOU_TYPE=wiou`：启用 WIoU v3（α=1.9、δ=3、momentum=0.1，可用 `WIOU_ALPHA`/`WIOU_DELTA`/`WIOU_MOMENTUM` 调节）
