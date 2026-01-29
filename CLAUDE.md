# CLAUDE.md

本文件为 Claude Code 提供项目上下文。

## 项目概述

AIM Radar - 基于 YOLO 的机甲大师 (RoboMaster) 比赛赛场视频流无人机实时识别系统。

## 技术栈

- **深度学习框架**: PyTorch, YOLOv5
- **目标检测**: YOLOv5 (唯一支持的模型格式)
- **GPU 加速**: NVIDIA CUDA

## 重要约束

- **模型格式**: 项目只使用 YOLOv5 模型，不使用 YOLOv8/v11
- **训练输出**: 所有训练必须输出 YOLOv5 格式的 .pt 文件
- **兼容性**: 确保所有脚本与 YOLOv5 模型兼容

## 硬件配置

- **GPU**: NVIDIA RTX 5060 (8GB VRAM)
- **CPU**: Intel i9-14900 (24核)
- **RAM**: 32GB

## CUDA 配置

详细的 CUDA 环境配置请参考 [CUDA_Setup.md](CUDA_Setup.md)。

关键要点：
- 确保 `nvidia-smi` 可用
- 安装对应 CUDA 版本的 PyTorch（推荐 cu121 或 cu124）
- 验证 `torch.cuda.is_available()` 返回 True

## 模型目录结构

```
models/
├── AntiUAV/           # 反无人机数据集（需要训练）
│   ├── train/images/  # 2244 张训练图片
│   ├── valid/images/  # 241 张验证图片
│   └── data.yaml      # 数据集配置
├── DroneDetection/    # 无人机检测数据集（已有 best.pt）
│   ├── train/images/
│   ├── valid/images/
│   ├── test/images/
│   ├── data.yaml
│   └── best.pt        # 已训练模型
└── model3.pt          # 独立模型文件
```

## 常用命令

### 训练 YOLOv5 模型
```bash
python scripts/train_yolov5.py --data models/AntiUAV/data.yaml --epochs 100 --weights yolov5s.pt
```

### 运行基准测试
```bash
python scripts/benchmark.py
```

### 验证 CUDA
```bash
python scripts/test_cuda.py
```

## 开发指南

1. 使用虚拟环境 `.venv`
2. 优先使用 GPU 加速
3. 训练输出保存在 `runs/` 目录
4. **重要**: 所有模型训练必须在外部命令行窗口执行，不能在 Claude 后台运行（训练时间长，需要用户监控）

## 进度跟踪

实时进度请查看 [progress.md](progress.md)。

每次操作后应更新 progress.md：
- 记录已完成的任务
- 更新当前进行中的任务
- 添加测试结果和日志
