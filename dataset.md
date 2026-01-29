# 数据集格式说明

本文档说明 YOLOv5 训练所需的数据集格式要求。

## 目录结构

```
my_dataset/
├── data.yaml           # 数据集配置文件
├── train/
│   ├── images/         # 训练图片
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   └── labels/         # 训练标签
│       ├── img001.txt
│       ├── img002.txt
│       └── ...
└── valid/
    ├── images/         # 验证图片
    │   ├── img101.jpg
    │   └── ...
    └── labels/         # 验证标签
        ├── img101.txt
        └── ...
```

## 配置文件 (data.yaml)

```yaml
# 数据集根路径 (可选，默认为 data.yaml 所在目录)
path: /path/to/my_dataset

# 训练和验证集路径 (相对于 path)
train: train/images
val: valid/images

# 类别数量
nc: 1

# 类别名称列表
names: ['drone']
```

### 多类别示例

```yaml
path: /path/to/dataset
train: train/images
val: valid/images

nc: 3
names: ['drone', 'bird', 'airplane']
```

## 标签格式 (YOLO格式)

每张图片对应一个同名的 `.txt` 标签文件。

### 格式说明

每行一个目标，格式为：
```
<class_id> <x_center> <y_center> <width> <height>
```

| 字段 | 说明 | 取值范围 |
|------|------|----------|
| class_id | 类别索引 | 0, 1, 2, ... (从0开始) |
| x_center | 边界框中心 X 坐标 | 0.0 ~ 1.0 (相对于图像宽度) |
| y_center | 边界框中心 Y 坐标 | 0.0 ~ 1.0 (相对于图像高度) |
| width | 边界框宽度 | 0.0 ~ 1.0 (相对于图像宽度) |
| height | 边界框高度 | 0.0 ~ 1.0 (相对于图像高度) |

### 示例

假设图片尺寸为 1920x1080，无人机边界框为 (100, 200) 到 (300, 400)：

```
# 计算归一化坐标
x_center = (100 + 300) / 2 / 1920 = 0.104
y_center = (200 + 400) / 2 / 1080 = 0.278
width = (300 - 100) / 1920 = 0.104
height = (400 - 200) / 1080 = 0.185

# 标签内容 (drone.txt)
0 0.104 0.278 0.104 0.185
```

### 多目标示例

```
0 0.104 0.278 0.104 0.185
0 0.521 0.463 0.083 0.139
1 0.750 0.300 0.120 0.200
```

## 图片要求

### 支持格式
- JPG / JPEG
- PNG
- BMP
- WEBP

### 建议规格
- 分辨率：640x640 或更高
- 文件大小：< 5MB
- 颜色空间：RGB

## 数据集划分建议

| 数据集 | 比例 | 用途 |
|--------|------|------|
| train | 70-80% | 模型训练 |
| valid | 15-20% | 训练时验证 |
| test | 5-10% | 最终测试 (可选) |

## 标注工具推荐

### 1. LabelImg (本地)
```bash
pip install labelImg
labelImg
```
- 选择 YOLO 格式输出
- 快捷键：W=创建框，D=下一张，A=上一张

### 2. Roboflow (在线)
- 网址：https://roboflow.com
- 支持在线标注、数据增强、格式转换
- 可直接导出 YOLOv5 格式

### 3. CVAT (在线/本地)
- 网址：https://cvat.ai
- 支持团队协作
- 导出时选择 YOLO 1.1 格式

## 数据增强建议

训练时 YOLOv5 会自动应用数据增强，也可在 `hyp.yaml` 中自定义：

```yaml
# 色彩增强
hsv_h: 0.015    # 色调 ±1.5%
hsv_s: 0.7      # 饱和度 ±70%
hsv_v: 0.4      # 明度 ±40%

# 几何变换
degrees: 10     # 旋转 ±10度
translate: 0.1  # 平移 ±10%
scale: 0.5      # 缩放 ±50%
shear: 0.0      # 剪切

# 翻转
flipud: 0.0     # 上下翻转概率 (无人机不建议)
fliplr: 0.5     # 左右翻转概率

# 高级增强
mosaic: 1.0     # 马赛克拼接
mixup: 0.0      # 图像混合
```

## 快速验证数据集

```bash
# 检查数据集配置
python scripts/finetune.py --data path/to/data.yaml --epochs 1

# 或使用 YOLOv5 内置检查
cd yolov5
python -c "from utils.dataloaders import *; check_dataset('path/to/data.yaml')"
```

## 常见问题

### Q: 标签和图片名不匹配
确保标签文件名与图片文件名相同（仅扩展名不同）：
- `img001.jpg` → `img001.txt` ✓
- `img001.jpg` → `image001.txt` ✗

### Q: 没有目标的图片
可以保留空的 `.txt` 文件，或不创建标签文件（作为负样本）。

### Q: 坐标超出范围
所有坐标值必须在 0.0 ~ 1.0 之间，超出会导致训练错误。

### Q: 类别数量不匹配
确保 `data.yaml` 中的 `nc` 与 `names` 列表长度一致，且标签中的 class_id < nc。
