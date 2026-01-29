# AIM Radar 部署指南

本文档介绍如何部署和使用 AIM Radar 无人机检测系统。

## 目录

- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [模型选择](#模型选择)
- [视频检测](#视频检测)
- [模型微调](#模型微调)
- [性能优化](#性能优化)
- [常见问题](#常见问题)

---

## 环境要求

### 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| GPU | GTX 1060 6GB | RTX 3060 / RTX 5060 |
| CPU | 4核 | 8核以上 |
| 内存 | 8GB | 16GB以上 |
| 存储 | 10GB | 50GB SSD |

### 软件要求

- Python 3.10+
- CUDA 11.8+ (GPU加速)
- Windows 10/11 或 Linux

---

## 快速开始

### 1. 克隆项目

```bash
git clone <repository-url>
cd aim-radar
```

### 2. 创建虚拟环境

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 3. 安装依赖

```bash
# RTX 50系列 (Blackwell架构)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# RTX 30/40系列
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 其他依赖
pip install opencv-python pyyaml tqdm
```

### 4. 验证环境

```bash
python scripts/test_cuda.py
```

预期输出：
```
CUDA 可用: True
GPU: NVIDIA GeForce RTX XXXX
NMS 测试: 通过
```

---

## 模型选择

### 可用模型

| 模型 | 路径 | FPS | 特点 |
|------|------|-----|------|
| DroneDetection | `models/DroneDetection/best.pt` | 117 | 泛化能力强 |
| AntiUAV | `runs/yolov5/antiuav/weights/best.pt` | 111 | 蜂群检测精度高 |
| Model3 | `models/model3.pt` | 112 | 备选模型 |

### 选择建议

- **通用场景**: 使用 DroneDetection
- **比赛蜂群检测**: 使用 AntiUAV
- **自定义场景**: 基于 DroneDetection 微调

---

## 视频检测

### 摄像头实时检测

```bash
python scripts/detect_video.py --source 0
```

### 检测视频文件

```bash
python scripts/detect_video.py --source path/to/video.mp4
```

### 保存检测结果

```bash
python scripts/detect_video.py --source video.mp4 --save --output result.mp4
```

### 完整参数

```bash
python scripts/detect_video.py \
    --source 0                              # 视频源 (0=摄像头, 或视频路径)
    --weights models/DroneDetection/best.pt # 模型权重
    --conf 0.5                              # 置信度阈值
    --output output.mp4                     # 输出路径
    --save                                  # 保存视频
    --no-show                               # 不显示窗口
```

### 快捷键

| 按键 | 功能 |
|------|------|
| `q` | 退出 |
| `s` | 截图 |
| `p` | 暂停/继续 |

---

## 模型微调

### 1. 准备数据集

参考 [dataset.md](dataset.md) 准备数据集：

```
my_dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
└── valid/
    ├── images/
    └── labels/
```

### 2. 配置 data.yaml

```yaml
path: /path/to/my_dataset
train: train/images
val: valid/images
nc: 1
names: ['drone']
```

### 3. 开始微调

```bash
python scripts/finetune.py \
    --data path/to/data.yaml \
    --weights models/DroneDetection/best.pt \
    --epochs 50 \
    --batch 16 \
    --name my_drone_model
```

### 4. 微调参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data` | (必填) | 数据集配置文件 |
| `--weights` | DroneDetection | 预训练权重 |
| `--epochs` | 50 | 训练轮数 |
| `--batch` | 16 | 批次大小 (显存不足时减小) |
| `--imgsz` | 640 | 输入图像大小 |
| `--freeze` | 10 | 冻结前N层 (加速收敛) |
| `--patience` | 30 | 早停轮数 |
| `--lr` | 0.001 | 初始学习率 |

### 5. 使用微调后的模型

```bash
python scripts/detect_video.py \
    --weights runs/finetune/my_drone_model/weights/best.pt \
    --source video.mp4
```

---

## 性能优化

### GPU 内存优化

```bash
# 减小批次大小
python scripts/finetune.py --batch 8

# 减小图像尺寸
python scripts/finetune.py --imgsz 480
```

### 推理加速

```python
# 使用半精度 (FP16)
model.half()

# 设置推理模式
model.eval()
```

### 多摄像头部署

```python
# 使用线程池处理多路视频
from concurrent.futures import ThreadPoolExecutor

sources = [0, 1, 2]  # 多个摄像头
with ThreadPoolExecutor(max_workers=len(sources)) as executor:
    futures = [executor.submit(process_video, src) for src in sources]
```

---

## 常见问题

### Q1: CUDA 不可用

**症状**: `torch.cuda.is_available()` 返回 False

**解决方案**:
1. 确认安装了 NVIDIA 驱动: `nvidia-smi`
2. 安装正确版本的 PyTorch:
   ```bash
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```

### Q2: RTX 50系列 "no kernel image" 错误

**症状**: `RuntimeError: CUDA error: no kernel image is available`

**解决方案**:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### Q3: 内存不足 (OOM)

**症状**: `RuntimeError: CUDA out of memory`

**解决方案**:
```bash
# 减小批次大小
python scripts/finetune.py --batch 4 --workers 2
```

### Q4: 摄像头打不开

**症状**: `无法打开视频源: 0`

**解决方案**:
1. 确认摄像头连接正常
2. 尝试其他摄像头ID: `--source 1`
3. Windows: 检查相机权限设置

### Q5: 检测效果不好

**建议**:
1. 降低置信度阈值: `--conf 0.3`
2. 使用针对性训练的模型
3. 用目标场景数据微调模型

---

## 项目结构

```
aim-radar/
├── models/                    # 模型文件
│   ├── DroneDetection/
│   │   └── best.pt           # 推荐模型
│   ├── AntiUAV/
│   └── model3.pt
├── scripts/
│   ├── detect_video.py       # 视频检测
│   ├── finetune.py           # 模型微调
│   ├── train_yolov5.py       # 完整训练
│   ├── benchmark_models.py   # 性能测试
│   └── test_cuda.py          # 环境验证
├── runs/                      # 训练输出
├── yolov5/                    # YOLOv5 仓库 (自动克隆)
├── CLAUDE.md                  # 开发指南
├── DEPLOY.md                  # 部署指南 (本文件)
├── dataset.md                 # 数据集格式
├── models.md                  # 模型说明
└── progress.md                # 开发进度
```

---

## 联系与支持

- 问题反馈: 提交 GitHub Issue
- 模型下载: 参考 [models.md](models.md)
- 数据标注: 参考 [dataset.md](dataset.md)
