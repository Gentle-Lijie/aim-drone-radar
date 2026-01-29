# AIM Radar

基于 **YOLOv5** 的机甲大师 (RoboMaster) 比赛赛场视频流无人机实时识别系统。

> **重要**: 本项目统一使用 YOLOv5 模型格式，不支持 YOLOv8/v11。

## 功能特性

- 实时视频流无人机目标检测
- 基于 **YOLOv5** 的高效目标识别
- 支持 NVIDIA CUDA GPU 加速
- 支持 Intel CPU 优化 (Linux)

## 环境要求

- Python 3.10+
- NVIDIA GPU (推荐 RTX 系列) + CUDA Toolkit
- Windows / Linux

## 快速开始

### 1. 创建虚拟环境

```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows CMD
.\.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

### 2. 安装 PyTorch (CUDA 版本)

根据你的 CUDA 版本选择对应的安装命令：

```bash
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 3. 安装项目依赖

```bash
pip install -r requirements.txt
```

### 4. 验证 CUDA 可用性

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## 项目结构

```
aim-radar/
├── requirements.txt    # 项目依赖
├── CUDA_Setup.md       # CUDA 配置指南
└── README.md           # 项目说明
```

## CUDA 配置

详细的 CUDA 环境配置和故障排查请参考 [CUDA_Setup.md](CUDA_Setup.md)。

## 技术栈

- **深度学习框架**: PyTorch
- **目标检测**: YOLOv5 (唯一支持格式)
- **视觉处理**: OpenCV
- **GPU 加速**: CUDA
- **可视化**: Matplotlib, Streamlit

## License

MIT
