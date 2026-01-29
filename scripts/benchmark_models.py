#!/usr/bin/env python
"""
多模型 Benchmark - 评估多个 YOLOv5 模型的推理速度和准确率
"""

import argparse
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List

# 抑制警告
os.environ["YOLO_VERBOSE"] = "False"
warnings.filterwarnings("ignore", category=FutureWarning)

import torch


def get_default_models() -> Dict[str, str]:
    """获取默认模型路径"""
    base = Path(__file__).parent.parent
    return {
        "DroneDetection": str(base / "models/DroneDetection/best.pt"),
        "Model3": str(base / "models/model3.pt"),
    }


def load_yolov5_model(model_path: str):
    """加载 YOLOv5 模型"""
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False, verbose=False)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model


def benchmark_speed(model, images_dir: Path, warmup: int = 10, runs: int = 100) -> Dict:
    """测试推理速度"""
    images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    if not images:
        return {"error": "No images found"}

    # 循环使用图片
    test_images = (images * ((runs + warmup) // len(images) + 1))[:runs + warmup]

    # 预热
    for img in test_images[:warmup]:
        model(str(img))

    # 测速
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latencies = []

    for img in test_images[warmup:]:
        start = time.perf_counter()
        model(str(img))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)  # ms

    latencies.sort()
    return {
        "avg_ms": sum(latencies) / len(latencies),
        "p50_ms": latencies[len(latencies) // 2],
        "p95_ms": latencies[int(len(latencies) * 0.95)],
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "fps": 1000 / (sum(latencies) / len(latencies)),
    }


def benchmark_accuracy(model, images_dir: Path, labels_dir: Path) -> Dict:
    """测试准确率 - 计算 mAP"""
    from collections import defaultdict
    import numpy as np

    images = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    if not images:
        return {"error": "No images found"}

    all_preds = []
    all_labels = []
    tp, fp, fn = 0, 0, 0
    iou_threshold = 0.5

    for img_path in images:
        # 获取预测
        results = model(str(img_path))
        pred_boxes = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, cls]

        # 获取标签
        label_path = labels_dir / (img_path.stem + ".txt")
        gt_boxes = []
        if label_path.exists():
            # 获取图像尺寸 (兼容新旧 API)
            if hasattr(results, 'ims'):
                img_h, img_w = results.ims[0].shape[:2]
            elif hasattr(results, 'imgs'):
                img_h, img_w = results.imgs[0].shape[:2]
            else:
                from PIL import Image
                img = Image.open(img_path)
                img_w, img_h = img.size
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls, cx, cy, bw, bh = map(float, parts[:5])
                        x1 = (cx - bw / 2) * img_w
                        y1 = (cy - bh / 2) * img_h
                        x2 = (cx + bw / 2) * img_w
                        y2 = (cy + bh / 2) * img_h
                        gt_boxes.append([x1, y1, x2, y2, int(cls)])

        # 简单匹配计算 TP/FP/FN
        matched_gt = set()
        for pred in pred_boxes:
            px1, py1, px2, py2, conf, pcls = pred
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(gt_boxes):
                gx1, gy1, gx2, gy2, gcls = gt
                # 计算 IoU
                ix1 = max(px1, gx1)
                iy1 = max(py1, gy1)
                ix2 = min(px2, gx2)
                iy2 = min(py2, gy2)
                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                inter = iw * ih
                union = (px2-px1)*(py2-py1) + (gx2-gx1)*(gy2-gy1) - inter
                iou = inter / union if union > 0 else 0

                if iou > best_iou and gt_idx not in matched_gt:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_gt_idx)
            else:
                fp += 1

        fn += len(gt_boxes) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def run_benchmark(models: Dict[str, str], images_dir: Path, labels_dir: Path = None,
                  speed_only: bool = False, runs: int = 100) -> List[Dict]:
    """运行完整 benchmark"""
    results = []

    for name, model_path in models.items():
        print(f"\n{'='*50}")
        print(f"评估模型: {name}")
        print(f"路径: {model_path}")
        print("="*50)

        if not Path(model_path).exists():
            print(f"[跳过] 模型文件不存在")
            results.append({"name": name, "error": "Model not found"})
            continue

        try:
            print("加载模型...")
            model = load_yolov5_model(model_path)
            result = {"name": name, "path": model_path}

            # 速度测试
            print("\n[速度测试]")
            speed = benchmark_speed(model, images_dir, runs=runs)
            result["speed"] = speed
            if "error" not in speed:
                print(f"  平均延迟: {speed['avg_ms']:.2f} ms")
                print(f"  P50 延迟: {speed['p50_ms']:.2f} ms")
                print(f"  P95 延迟: {speed['p95_ms']:.2f} ms")
                print(f"  FPS: {speed['fps']:.1f}")
            else:
                print(f"  错误: {speed['error']}")

            # 准确率测试
            if not speed_only and labels_dir and labels_dir.exists():
                print("\n[准确率测试]")
                accuracy = benchmark_accuracy(model, images_dir, labels_dir)
                result["accuracy"] = accuracy
                if "error" not in accuracy:
                    print(f"  Precision: {accuracy['precision']:.4f}")
                    print(f"  Recall: {accuracy['recall']:.4f}")
                    print(f"  F1: {accuracy['f1']:.4f}")
                    print(f"  TP: {accuracy['tp']}, FP: {accuracy['fp']}, FN: {accuracy['fn']}")
                else:
                    print(f"  错误: {accuracy['error']}")

            results.append(result)

        except Exception as e:
            print(f"[错误] {e}")
            results.append({"name": name, "error": str(e)})

    return results


def print_summary(results: List[Dict]):
    """打印汇总表格"""
    print("\n" + "="*80)
    print("Benchmark 汇总")
    print("="*80)

    # 速度表格
    print("\n[速度对比]")
    print(f"{'模型':<20} {'平均(ms)':<12} {'P50(ms)':<12} {'P95(ms)':<12} {'FPS':<10}")
    print("-"*70)
    for r in results:
        if "error" in r:
            print(f"{r['name']:<20} 错误: {r['error'][:40]}")
            continue
        s = r.get("speed", {})
        if "error" in s:
            print(f"{r['name']:<20} {s['error']:<12}")
        else:
            print(f"{r['name']:<20} {s['avg_ms']:<12.2f} {s['p50_ms']:<12.2f} {s['p95_ms']:<12.2f} {s['fps']:<10.1f}")

    # 准确率表格
    has_accuracy = any("accuracy" in r and "error" not in r.get("accuracy", {}) for r in results)
    if has_accuracy:
        print("\n[准确率对比]")
        print(f"{'模型':<20} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print("-"*60)
        for r in results:
            if "error" in r:
                continue
            a = r.get("accuracy", {})
            if "error" in a:
                print(f"{r['name']:<20} {a.get('error', 'N/A'):<12}")
            elif a:
                print(f"{r['name']:<20} {a['precision']:<12.4f} {a['recall']:<12.4f} {a['f1']:<12.4f}")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="YOLOv5 多模型 Benchmark 评估")
    parser.add_argument("--models", type=str, nargs="+", help="模型路径列表")
    parser.add_argument("--names", type=str, nargs="+", help="模型名称列表")
    parser.add_argument("--images", type=str, default="models/AntiUAV/valid/images", help="测试图片目录")
    parser.add_argument("--labels", type=str, default=None, help="标签目录 (用于准确率测试)")
    parser.add_argument("--speed-only", action="store_true", help="只测试速度")
    parser.add_argument("--runs", type=int, default=100, help="速度测试次数")

    args = parser.parse_args()

    # 获取模型
    if args.models:
        if args.names and len(args.names) == len(args.models):
            models = dict(zip(args.names, args.models))
        else:
            models = {f"Model_{i}": p for i, p in enumerate(args.models)}
    else:
        models = get_default_models()

    images_dir = Path(args.images)
    if not images_dir.exists():
        print(f"错误: 图片目录不存在 - {images_dir}")
        return

    # 自动检测标签目录
    if args.labels:
        labels_dir = Path(args.labels)
    else:
        labels_dir = images_dir.parent.parent / "labels" / images_dir.name
        if not labels_dir.exists():
            labels_dir = images_dir.with_name("labels")

    print("="*50)
    print("YOLOv5 多模型 Benchmark")
    print("="*50)
    print(f"设备: {'CUDA - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"模型数量: {len(models)}")
    print(f"测试图片: {images_dir}")
    print(f"标签目录: {labels_dir if labels_dir.exists() else '未找到'}")
    print(f"速度测试次数: {args.runs}")

    results = run_benchmark(
        models=models,
        images_dir=images_dir,
        labels_dir=labels_dir if labels_dir.exists() else None,
        speed_only=args.speed_only,
        runs=args.runs,
    )

    print_summary(results)


if __name__ == "__main__":
    main()
