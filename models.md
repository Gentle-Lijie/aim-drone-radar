# 无人机检测模型指南

本文档介绍三个基于 YOLOv5 的无人机检测项目，包括下载链接、使用方法和微调指南。

---

## Benchmark 评估结果

测试环境：RTX 5060 Laptop GPU (8GB), PyTorch 2.10.0+cu128

### 速度对比 (平均值)
| 模型 | 平均延迟(ms) | FPS | 参数量 |
|------|-------------|-----|--------|
| DroneDetection | 8.5 | 117 | 7.0M |
| Model3 | 8.9 | 112 | 7.2M |
| AntiUAV | 9.1 | 111 | 7.0M |

### 准确率对比 - AntiUAV 验证集 (241张)
| 模型 | Precision | Recall | F1 |
|------|-----------|--------|-----|
| DroneDetection | 0.930 | 0.494 | 0.645 |
| Model3 | 0.842 | 0.286 | 0.427 |
| **AntiUAV** | **0.975** | **0.983** | **0.979** |

### 准确率对比 - DroneDetection 验证集 (292张)
| 模型 | Precision | Recall | F1 |
|------|-----------|--------|-----|
| **DroneDetection** | **0.843** | **0.606** | **0.705** |
| Model3 | 0.610 | 0.503 | 0.552 |
| AntiUAV | 0.034 | 0.021 | 0.026 |

### 交叉验证总结
| 模型 | 自身数据集 F1 | 跨数据集 F1 | 泛化能力 |
|------|--------------|-------------|----------|
| DroneDetection | 0.705 | 0.645 | ⭐⭐⭐ 良好 |
| Model3 | - | 0.43~0.55 | ⭐⭐ 一般 |
| AntiUAV | 0.979 | 0.026 | ⭐ 较差 |

### 结论
- **泛化能力最强**: DroneDetection（跨数据集仍保持 F1>0.6）
- **特定场景最优**: AntiUAV（在蜂群检测场景 F1=97.9%）
- **推荐策略**: 根据实际部署场景选择模型，或用目标场景数据微调

---

## 1. Anti-UAV 实时无人机蜂群检测

### 下载链接
- **仓库地址**: https://github.com/blurryface-1/Anti-UAV---Drone-Swarm-Detection
- **ZIP下载**: https://github.com/blurryface-1/Anti-UAV---Drone-Swarm-Detection/archive/refs/heads/main.zip
- **数据集下载**: https://app.roboflow.com/ds/GCfAhcDsm5?key=M3f5GiSxHE

### 项目简介
使用 YOLOv5 检测无人机群体，精度达 99.1%，召回率 97.5%。训练数据包含 2,485 张图片（2,244 训练 + 241 验证）。

### 使用方法

```bash
# 克隆仓库
git clone https://github.com/blurryface-1/Anti-UAV---Drone-Swarm-Detection.git
cd Anti-UAV---Drone-Swarm-Detection

# 安装依赖
pip install -r requirements.txt

# 推理检测
python detect.py --weights best.pt --source your_image.jpg
python detect.py --weights best.pt --source your_video.mp4
```

### 微调方法

```bash
# 1. 准备数据集（YOLO格式）
# 目录结构:
# dataset/
#   ├── images/
#   │   ├── train/
#   │   └── val/
#   └── labels/
#       ├── train/
#       └── val/

# 2. 创建数据配置文件 data.yaml
# data.yaml 示例:
# train: dataset/images/train
# val: dataset/images/val
# nc: 1  # 类别数
# names: ['drone']

# 3. 下载预训练权重
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt

# 4. 开始训练
python train.py --img 640 --batch 8 --epochs 80 --data data.yaml --weights yolov5s.pt

# 5. 训练完成后，最佳权重保存在 runs/train/exp/weights/best.pt
```

---

## 2. Advanced Aerial Drone Detection System

### 下载链接
- **仓库地址**: https://github.com/Ayushkumawat/Advanced-Aerial-Drone-Detection-System
- **ZIP下载**: https://github.com/Ayushkumawat/Advanced-Aerial-Drone-Detection-System/archive/refs/heads/main.zip
- **数据集下载**: https://universe.roboflow.com/drone-detection-ehdcs/drone-dataset-by-ayushkumawat
- **预训练权重**: 仓库中包含 `best.pt`

### 项目简介
提供现成的 `best.pt` 模型权重，支持实时检测和交互式边界框调整，适合快速原型开发和演示。训练数据包含 1,400 张多种类型无人机图片。

### 使用方法

```bash
# 克隆仓库
git clone https://github.com/Ayushkumawat/Advanced-Aerial-Drone-Detection-System.git
cd Advanced-Aerial-Drone-Detection-System

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 运行检测脚本
python Advanced_Drone_Detection.py
```

### 操作说明
- 脚本将打开默认摄像头的实时视频
- 点击并拖动鼠标可创建检测边界矩形
- 拖动矩形角点可调整边界
- 当无人机进入边界时显示警告
- 按 `q` 退出程序

### 微调方法

```bash
# 1. 从 Roboflow 下载数据集
# 访问: https://universe.roboflow.com/drone-detection-ehdcs/drone-dataset-by-ayushkumawat
# 导出为 YOLOv5 格式

# 2. 克隆 YOLOv5 官方仓库
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt

# 3. 使用现有 best.pt 作为预训练权重进行微调
python train.py --img 640 --batch 16 --epochs 100 \
    --data path/to/your/data.yaml \
    --weights path/to/best.pt \
    --name drone_finetune

# 4. 调整置信度阈值（在代码中修改）
# 修改 conf 参数来控制检测灵敏度
```

---

## 3. Drone Detection Using YOLOv5

### 下载链接
- **仓库地址**: https://github.com/akashthakur4553/Drone_Detection_Using_YOLOv5
- **ZIP下载**: https://github.com/akashthakur4553/Drone_Detection_Using_YOLOv5/archive/refs/heads/main.zip

### 项目简介
带有 Streamlit Web 界面的无人机检测系统，支持多目标检测、目标追踪和检测通知功能。

### 使用方法

```bash
# 克隆仓库
git clone https://github.com/akashthakur4553/Drone_Detection_Using_YOLOv5.git
cd Drone_Detection_Using_YOLOv5

# 安装依赖
pip install -r requirements.txt

# 运行 Streamlit 应用
streamlit run final_trial.py
```

### 主要功能
- 实时无人机检测
- 多目标同时检测
- 目标轨迹追踪
- 检测到无人机时发送通知（含置信度）

### 微调方法

```bash
# 1. 准备自定义数据集
# 使用 LabelImg 或 Roboflow 进行图像标注
# 确保标注格式为 YOLO 格式 (txt文件，每行: class x_center y_center width height)

# 2. 安装 YOLOv5
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt

# 3. 配置数据集 (创建 custom_data.yaml)
# custom_data.yaml:
# path: /path/to/dataset
# train: images/train
# val: images/val
# nc: 1
# names: ['drone']

# 4. 训练模型
python train.py --img 640 --batch 16 --epochs 50 \
    --data custom_data.yaml \
    --weights yolov5s.pt \
    --name drone_custom

# 5. 将训练好的权重替换到项目中
cp runs/train/drone_custom/weights/best.pt /path/to/Drone_Detection_Using_YOLOv5/
```

---

## 通用微调技巧

### 数据标注工具
- **LabelImg**: https://github.com/tzutalin/labelImg
- **Roboflow**: https://roboflow.com (在线标注和数据增强)
- **CVAT**: https://github.com/opencv/cvat

### 数据增强建议
```yaml
# 在 hyp.yaml 中配置数据增强参数
hsv_h: 0.015  # 色调
hsv_s: 0.7    # 饱和度
hsv_v: 0.4    # 明度
degrees: 10   # 旋转角度
translate: 0.1  # 平移
scale: 0.5    # 缩放
shear: 0.0    # 剪切
flipud: 0.5   # 上下翻转概率
fliplr: 0.5   # 左右翻转概率
mosaic: 1.0   # 马赛克增强
```

### 训练参数调优
| 参数 | 说明 | 建议值 |
|------|------|--------|
| `--img` | 输入图像尺寸 | 640 或 1280 |
| `--batch` | 批次大小 | 根据显存调整 (8/16/32) |
| `--epochs` | 训练轮数 | 50-300 |
| `--weights` | 预训练权重 | yolov5s.pt / yolov5m.pt |
| `--patience` | 早停轮数 | 50 |

### 模型选择
| 模型 | 大小 | 速度 | 精度 |
|------|------|------|------|
| YOLOv5n | 最小 | 最快 | 较低 |
| YOLOv5s | 小 | 快 | 中等 |
| YOLOv5m | 中 | 中等 | 较高 |
| YOLOv5l | 大 | 较慢 | 高 |
| YOLOv5x | 最大 | 最慢 | 最高 |

### 快速克隆命令

```bash
# 一键克隆所有仓库
git clone https://github.com/blurryface-1/Anti-UAV---Drone-Swarm-Detection.git
git clone https://github.com/Ayushkumawat/Advanced-Aerial-Drone-Detection-System.git
git clone https://github.com/akashthakur4553/Drone_Detection_Using_YOLOv5.git
```
