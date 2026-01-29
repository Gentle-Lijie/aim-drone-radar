#!/usr/bin/env python
"""
使用 Ultralytics YOLO 训练无人机检测模型
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="训练 YOLO 目标检测模型")
    parser.add_argument("--data", type=str, default="models/AntiUAV/data.yaml", help="数据集配置")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="预训练模型 (yolo11n/s/m/l/x)")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch", type=int, default=16, help="批次大小")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图像大小")
    parser.add_argument("--device", type=str, default="0", help="设备 (cpu 或 GPU编号)")
    parser.add_argument("--project", type=str, default="runs", help="输出目录")
    parser.add_argument("--name", type=str, default="antiuav", help="实验名称")

    args = parser.parse_args()

    print("=" * 50)
    print("YOLO 无人机检测训练")
    print("=" * 50)
    print(f"数据集: {args.data}")
    print(f"模型: {args.model}")
    print(f"轮数: {args.epochs}")
    print(f"批次: {args.batch}")
    print(f"图像大小: {args.imgsz}")
    print(f"设备: {args.device}")
    print("=" * 50)

    # 加载模型
    model = YOLO(args.model)

    # 开始训练
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        verbose=True,
    )

    print("\n训练完成!")
    print(f"最佳模型: {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
