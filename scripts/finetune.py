#!/usr/bin/env python
"""
YOLOv5 模型微调脚本
基于 DroneDetection 预训练模型进行微调
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# 抑制警告
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning"


def check_dataset(data_yaml: Path) -> bool:
    """检查数据集配置是否有效"""
    import yaml

    if not data_yaml.exists():
        print(f"[错误] 数据集配置文件不存在: {data_yaml}")
        return False

    with open(data_yaml) as f:
        config = yaml.safe_load(f)

    required_keys = ["train", "val", "nc", "names"]
    for key in required_keys:
        if key not in config:
            print(f"[错误] data.yaml 缺少必要字段: {key}")
            return False

    # 检查路径
    base_path = data_yaml.parent
    if "path" in config:
        base_path = Path(config["path"])

    train_path = base_path / config["train"]
    val_path = base_path / config["val"]

    if not train_path.exists():
        print(f"[错误] 训练集路径不存在: {train_path}")
        return False

    if not val_path.exists():
        print(f"[错误] 验证集路径不存在: {val_path}")
        return False

    train_images = list(train_path.glob("*.jpg")) + list(train_path.glob("*.png"))
    val_images = list(val_path.glob("*.jpg")) + list(val_path.glob("*.png"))

    print(f"[数据集] 训练集: {len(train_images)} 张图片")
    print(f"[数据集] 验证集: {len(val_images)} 张图片")
    print(f"[数据集] 类别数: {config['nc']}")
    print(f"[数据集] 类别名: {config['names']}")

    return True


def main():
    parser = argparse.ArgumentParser(description="YOLOv5 模型微调")
    parser.add_argument("--data", type=str, required=True, help="数据集配置文件 (data.yaml)")
    parser.add_argument("--weights", type=str, default="models/DroneDetection/best.pt",
                        help="预训练权重路径 (默认: DroneDetection)")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch", type=int, default=16, help="批次大小")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图像大小")
    parser.add_argument("--device", type=str, default="0", help="设备 (cpu 或 GPU编号)")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--project", type=str, default="runs/finetune", help="输出目录")
    parser.add_argument("--name", type=str, default="exp", help="实验名称")
    parser.add_argument("--patience", type=int, default=30, help="早停轮数")
    parser.add_argument("--freeze", type=int, default=10, help="冻结前N层 (0=不冻结)")
    parser.add_argument("--lr", type=float, default=0.001, help="初始学习率")
    parser.add_argument("--resume", action="store_true", help="从上次中断处继续训练")

    args = parser.parse_args()

    print("=" * 60)
    print("YOLOv5 模型微调")
    print("=" * 60)

    # 检查数据集
    data_path = Path(args.data).absolute()
    if not check_dataset(data_path):
        sys.exit(1)

    # 检查预训练权重
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"[错误] 预训练权重不存在: {weights_path}")
        sys.exit(1)

    print(f"\n[配置]")
    print(f"  预训练权重: {args.weights}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  批次大小: {args.batch}")
    print(f"  图像大小: {args.imgsz}")
    print(f"  冻结层数: {args.freeze}")
    print(f"  学习率: {args.lr}")
    print(f"  早停轮数: {args.patience}")
    print(f"  输出目录: {args.project}/{args.name}")
    print("=" * 60)

    # 检查 yolov5 目录
    yolov5_dir = Path(__file__).parent.parent / "yolov5"
    if not yolov5_dir.exists():
        print("\n正在克隆 YOLOv5 仓库...")
        subprocess.run([
            "git", "clone", "https://github.com/ultralytics/yolov5.git",
            str(yolov5_dir)
        ], check=True)

    # 构建训练命令
    train_script = yolov5_dir / "train.py"

    cmd = [
        sys.executable, "-W", "ignore::FutureWarning", str(train_script),
        "--data", str(data_path),
        "--weights", str(weights_path.absolute()),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch),
        "--img", str(args.imgsz),
        "--device", args.device,
        "--workers", str(args.workers),
        "--project", args.project,
        "--name", args.name,
        "--patience", str(args.patience),
        "--exist-ok",
        "--hyp", str(yolov5_dir / "data/hyps/hyp.finetune.yaml"),
    ]

    # 冻结层
    if args.freeze > 0:
        cmd.extend(["--freeze", str(args.freeze)])

    # 继续训练
    if args.resume:
        cmd.append("--resume")

    print(f"\n执行命令:\n{' '.join(cmd)}\n")

    # 运行训练
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\n" + "=" * 60)
        print("微调完成!")
        print(f"最佳模型: {args.project}/{args.name}/weights/best.pt")
        print(f"最终模型: {args.project}/{args.name}/weights/last.pt")
        print("=" * 60)
    else:
        print(f"\n训练失败，返回码: {result.returncode}")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
