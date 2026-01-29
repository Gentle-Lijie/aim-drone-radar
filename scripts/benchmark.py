#!/usr/bin/env python
"""
模型基准测试脚本 (纯 PyTorch)
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def check_env():
    """检查环境"""
    print("=" * 60)
    print("环境检测")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    print(f"torchvision: {torchvision.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  显存: {props.total_memory / 1024**3:.1f} GB")
    print()
    return torch.cuda.is_available()


def find_models(models_dir: Path) -> list:
    """查找模型文件"""
    models = []
    for pt_file in models_dir.glob("*.pt"):
        models.append({"name": pt_file.stem, "path": pt_file, "type": "standalone"})
    for subdir in models_dir.iterdir():
        if subdir.is_dir():
            best_pt = subdir / "best.pt"
            if best_pt.exists():
                models.append({"name": subdir.name, "path": best_pt, "type": "trained"})
    return models


def benchmark_fasterrcnn(device, imgsz=640, warmup=10, runs=100):
    """测试 Faster R-CNN"""
    model = fasterrcnn_resnet50_fpn_v2(weights=None, num_classes=2)
    model.to(device).eval()

    dummy = torch.rand(1, 3, imgsz, imgsz).to(device)

    # 预热
    print(f"  预热 ({warmup} 次)...", end=" ", flush=True)
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    print("完成")

    # 测试
    print(f"  推理测试 ({runs} 次)...", end=" ", flush=True)
    latencies = []
    with torch.no_grad():
        for _ in range(runs):
            start = time.perf_counter()
            model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)
    print("完成")

    latencies.sort()
    return {
        "avg_ms": sum(latencies) / len(latencies),
        "min_ms": latencies[0],
        "max_ms": latencies[-1],
        "p50_ms": latencies[len(latencies) // 2],
        "p95_ms": latencies[int(len(latencies) * 0.95)],
        "fps": 1000 / (sum(latencies) / len(latencies)),
    }


def main():
    parser = argparse.ArgumentParser(description="模型基准测试")
    parser.add_argument("--models-dir", type=str, default="models", help="模型目录")
    parser.add_argument("--device", type=str, default="0", help="设备 (cpu 或 GPU编号)")
    parser.add_argument("--imgsz", type=int, default=640, help="图像尺寸")
    parser.add_argument("--warmup", type=int, default=10, help="预热次数")
    parser.add_argument("--runs", type=int, default=100, help="测试次数")

    args = parser.parse_args()

    cuda_ok = check_env()
    if args.device != "cpu" and not cuda_ok:
        print("CUDA 不可用，使用 CPU")
        args.device = "cpu"

    device = torch.device("cpu" if args.device == "cpu" else f"cuda:{args.device}")
    print(f"测试设备: {device}\n")

    # 列出模型
    models_dir = Path(args.models_dir)
    models = find_models(models_dir)
    print("=" * 60)
    print(f"找到 {len(models)} 个模型文件:")
    for m in models:
        print(f"  {m['name']}: {m['path']}")
    print()

    results = []

    # Faster R-CNN 基准测试
    print("=" * 60)
    print("Faster R-CNN 基准测试")
    print("-" * 60)
    bench = benchmark_fasterrcnn(device, args.imgsz, args.warmup, args.runs)
    bench["name"] = "FasterRCNN-ResNet50-FPN"
    bench["type"] = "torchvision"
    results.append(bench)

    # 打印结果
    print("\n")
    print("=" * 60)
    print("测试结果")
    print("=" * 60)
    print(f"{'模型':<30} {'平均(ms)':<10} {'P50(ms)':<10} {'P95(ms)':<10} {'FPS':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<30} {r['avg_ms']:<10.2f} {r['p50_ms']:<10.2f} {r['p95_ms']:<10.2f} {r['fps']:<10.1f}")

    # 保存结果
    output = Path("benchmark_results.json")
    with open(output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {output}")


if __name__ == "__main__":
    main()
