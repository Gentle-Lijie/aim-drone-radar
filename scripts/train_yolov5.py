#!/usr/bin/env python
"""
YOLOv5 模型训练脚本
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# 抑制 FutureWarning
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"


def main():
    parser = argparse.ArgumentParser(description="训练 YOLOv5 目标检测模型")
    parser.add_argument("--data", type=str, default="models/AntiUAV/data.yaml", help="数据集配置")
    parser.add_argument("--weights", type=str, default="yolov5s.pt", help="预训练权重 (yolov5n/s/m/l/x)")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch", type=int, default=16, help="批次大小")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图像大小")
    parser.add_argument("--device", type=str, default="0", help="设备 (cpu 或 GPU编号)")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers 数量")
    parser.add_argument("--project", type=str, default="runs/yolov5", help="输出目录")
    parser.add_argument("--name", type=str, default="train", help="实验名称")

    args = parser.parse_args()

    print("=" * 50)
    print("YOLOv5 训练")
    print("=" * 50)
    print(f"数据集: {args.data}")
    print(f"预训练权重: {args.weights}")
    print(f"轮数: {args.epochs}")
    print(f"批次: {args.batch}")
    print(f"图像大小: {args.imgsz}")
    print(f"设备: {args.device}")
    print(f"输出: {args.project}/{args.name}")
    print("=" * 50)

    # 检查 yolov5 是否已克隆
    yolov5_dir = Path(__file__).parent.parent / "yolov5"

    if not yolov5_dir.exists():
        print("\n正在克隆 YOLOv5 仓库...")
        subprocess.run([
            "git", "clone", "https://github.com/ultralytics/yolov5.git",
            str(yolov5_dir)
        ], check=True)

        print("安装 YOLOv5 依赖...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r",
            str(yolov5_dir / "requirements.txt")
        ], check=True)

    # 构建训练命令
    train_script = yolov5_dir / "train.py"
    data_path = Path(args.data).absolute()

    cmd = [
        sys.executable, "-W", "ignore::FutureWarning", str(train_script),
        "--data", str(data_path),
        "--weights", args.weights,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch),
        "--img", str(args.imgsz),
        "--device", args.device,
        "--workers", str(args.workers),
        "--project", args.project,
        "--name", args.name,
        "--exist-ok",
    ]

    print(f"\n执行命令: {' '.join(cmd)}\n")

    # 运行训练
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\n" + "=" * 50)
        print("训练完成!")
        print(f"最佳模型: {args.project}/{args.name}/weights/best.pt")
        print("=" * 50)
    else:
        print(f"\n训练失败，返回码: {result.returncode}")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
