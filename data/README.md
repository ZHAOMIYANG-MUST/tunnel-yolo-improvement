# 数据集获取与准备说明

本仓库不含数据集文件（约 1.4GB，超出 GitHub 建议体积），请按以下步骤自行准备：

## 1. 下载数据集

- 数据集：**Tunnel Leakage Defect Dataset**（OpenConstruction 平台）
- 来源：https://www.openconstruction.org/ 搜索 "Tunnel Leakage"
- 论文：Jia et al., *Advanced Engineering Informatics*, 2026, doi:10.1016/j.aei.2026.104955
- 协议：GPL-3.0，详情页提供百度网盘下载链接

## 2. 放置与预处理

下载得到 `Leakage.zip` 后放到项目根目录，然后：

```bash
# 解压（图片进 raw_dataset/images/，多边形标注进 raw_dataset/labels/）
python -c "import zipfile; zipfile.ZipFile('Leakage.zip').extractall('raw_dataset')"

# 多边形分割标注 → 检测框标注（生成 raw_dataset/labels_bbox/）
python scripts/polygon_to_bbox.py

# 按 8:1:1 划分 train/val/test（确定性划分，结果可复现）
python scripts/split_dataset.py
```

完成后目录结构：

```
datasets/tunnel/
├── images/{train,val,test}     # 1880 / 235 / 236 张
└── labels/{train,val,test}     # 对应 YOLO 检测框标注
```

## 3. 训练

```bash
yolo detect train data=configs/tunnel.yaml model=yolov8n.pt epochs=300 imgsz=640 batch=16 device=0 workers=4 seed=42 project=runs name=baseline
```
