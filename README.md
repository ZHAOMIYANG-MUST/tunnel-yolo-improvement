# 基于改进 YOLOv8 的隧道衬砌渗漏检测 —— 基线复现阶段交付说明

> 毕业设计：隧道衬砌病害（渗漏水）目标检测
> 阶段：基线复现（已完成）
> 更新日期：2026-09-19

---

## 一、本阶段工作概述

本阶段完成了毕业设计的技术调研与基线模型复现，具体包括：

1. **文献调研**：调研了隧道衬砌病害检测领域的深度学习方法（以 YOLO 系列改进为主），梳理了现有研究的痛点与改进方向，形成调研报告（见 `调研报告.html`）。
2. **数据准备**：采用 OpenConstruction 平台公开的 Tunnel Leakage Defect Dataset（2351 张隧道渗漏图像），将原始的多边形分割标注转换为检测框标注，并按 8:1:1 划分训练/验证/测试集。
3. **环境搭建与基线训练**：在本机（RTX 5060 Laptop GPU）上基于 Ultralytics 框架复现 YOLOv8n 基线，300 轮完整训练，并在独立测试集上评估。
4. **实验记录**：全过程（环境、数据、参数、结果、问题与解决）整理为实验记录文档，见 `docs/实验记录-基线复现.md`。

## 二、基线实验结果

**模型**：YOLOv8n（预训练权重初始化），训练配置：epochs=300, imgsz=640, batch=16, seed=42

| 评估集 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| 验证集（235 张图 / 273 框） | 0.909 | 0.929 | **0.947** | **0.726** |
| 测试集（236 张图 / 263 框，未参与训练） | 0.871 | 0.928 | **0.929** | **0.723** |

- 模型规格：3.01M 参数，8.1 GFLOPs，单张推理 5.0ms（约 200 FPS）
- 训练与测试集指标接近，无明显过拟合；召回率高于精确率，符合病害检测"不漏检优先"的应用需求
- 该结果为后续改进实验（注意力机制、小目标检测层、损失函数改进等）的对照基准

## 三、交付物清单

### 1. 调研报告（重点交付物）

- **`调研报告.html`** —— 技术调研报告（网页版，浏览器直接打开）
  内容按四个核心问题组织：隧道病害检测存在什么问题 → 问题产生的技术根源 → 现有文献中的改进方法 → 本课题接下来的研究计划。其中所有关键数据均标注了文献出处，文末附参考文献列表（含 DOI）。

### 2. 实验记录

- **`docs/实验记录-基线复现.md`** —— 基线复现完整实验记录
  含实验环境（硬件/软件版本）、数据集来源与预处理、训练配置、验证集与测试集结果、复现命令、过程中遇到的问题及解决方案。

### 3. 训练产物（`runs/detect/runs/baseline/`）

| 文件 | 说明 |
|---|---|
| `weights/best.pt` | 最优模型权重（按验证集 mAP 选出） |
| `weights/last.pt` | 最后一轮权重 |
| `results.csv` | 300 轮逐轮训练指标（损失、P、R、mAP） |
| `results.png` | 训练过程曲线总览 |
| `BoxPR_curve.png` | PR 曲线 |
| `BoxF1_curve.png` | F1-置信度曲线 |
| `confusion_matrix.png` | 混淆矩阵（含归一化版本） |
| `val_batch*_pred.jpg` 等 | 验证集预测可视化（预测框 vs 真实框对比） |

测试集独立评估产物在 `runs/detect/runs/baseline_test/`，训练全程日志在根目录 `baseline_log.txt`。

### 4. 代码与配置

| 文件 | 说明 |
|---|---|
| `train.py` | 基线训练脚本（固定随机种子，保证实验可复现） |
| `configs/tunnel.yaml` | 数据集配置文件（路径、类别定义） |
| `scripts/polygon_to_bbox.py` | 标注格式转换：多边形分割标注 → 检测框标注 |
| `scripts/split_dataset.py` | 数据集划分脚本（按文件名确定性排序，8:1:1 划分） |

### 5. 数据集

- `datasets/tunnel/` —— 划分后的数据集：train 1880 / val 235 / test 236 张
- `raw_dataset/` —— 原始数据与转换后的标注
- 数据来源：OpenConstruction "Tunnel Leakage Defect Dataset"（Jia et al., *Advanced Engineering Informatics*, 2026, doi:10.1016/j.aei.2026.104955），开源协议 GPL-3.0

### 6. 其他文档

- `毕设完整执行手册.md` —— 毕设整体执行计划（文献清单、16 周时间表、实验设计、论文结构、答辩准备）

## 四、复现步骤

```bash
# 1. 激活 Python 虚拟环境
E:\YOLO\venv\Scripts\activate

# 2.（可选）重新划分数据集
python scripts/split_dataset.py

# 3. 基线训练（约 1.8 小时）
yolo detect train data=configs/tunnel.yaml model=yolov8n.pt epochs=300 imgsz=640 batch=16 device=0 workers=4 seed=42 project=runs name=baseline

# 4. 在测试集上评估
yolo detect val model=runs/detect/runs/baseline/weights/best.pt data=configs/tunnel.yaml split=test
```

环境依赖：Python 3.13、PyTorch 2.11.0+cu128（RTX 50 系显卡需 cu128 版本）、Ultralytics 8.4.155。

## 五、下一阶段计划

在基线基础上进行模型改进实验（核心创新部分）：

1. 改进损失函数（WIoU / Shape-IoU，针对病害目标形状不规则的特点）
2. 增加 P2 小目标检测层（针对细小病害漏检问题）
3. 引入注意力机制（针对隧道低光照、背景干扰问题）

每个改进单独消融验证（统一 seed=42 与训练配置），最终形成改进模型并撰写论文。

---

*如需查看详细实验过程与问题记录，请阅读 `docs/实验记录-基线复现.md`；如需了解选题依据与研究现状，请打开 `调研报告.html`。*
