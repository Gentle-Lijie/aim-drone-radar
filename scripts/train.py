#!/usr/bin/env python
"""
使用 Faster R-CNN 训练目标检测模型 (纯 PyTorch)
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import yaml


class UAVDataset(Dataset):
    """YOLO 格式数据集"""

    def __init__(self, images_dir, labels_dir, transforms=None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transforms = transforms
        self.images = sorted(list(self.images_dir.glob("*.jpg")) + list(self.images_dir.glob("*.png")))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # 加载 YOLO 格式标签
        label_path = self.labels_dir / (img_path.stem + ".txt")
        boxes = []
        labels = []

        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls, cx, cy, bw, bh = map(float, parts[:5])
                        # YOLO 格式转 xyxy
                        x1 = (cx - bw / 2) * w
                        y1 = (cy - bh / 2) * h
                        x2 = (cx + bw / 2) * w
                        y2 = (cy + bh / 2) * h
                        boxes.append([x1, y1, x2, y2])
                        labels.append(int(cls) + 1)  # 0 是背景

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        if self.transforms:
            img = self.transforms(img)
        else:
            img = torchvision.transforms.ToTensor()(img)

        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


def get_model(num_classes):
    """获取预训练的 Faster R-CNN 模型"""
    model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0
    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        if (i + 1) % 10 == 0:
            print(f"  Epoch {epoch}, Batch {i+1}/{len(data_loader)}, Loss: {losses.item():.4f}", flush=True)

    return total_loss / len(data_loader)


def main():
    parser = argparse.ArgumentParser(description="训练 Faster R-CNN 目标检测模型")
    parser.add_argument("--data", type=str, default="models/AntiUAV/data.yaml", help="数据集配置")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--batch", type=int, default=4, help="批次大小")
    parser.add_argument("--lr", type=float, default=0.005, help="学习率")
    parser.add_argument("--device", type=str, default="0", help="设备 (cpu 或 GPU编号)")
    parser.add_argument("--output", type=str, default=None, help="输出路径")

    args = parser.parse_args()

    # 解析数据集配置
    data_path = Path(args.data)
    with open(data_path) as f:
        data_cfg = yaml.safe_load(f)

    base_path = Path(data_cfg.get("path", data_path.parent))
    train_images = base_path / data_cfg["train"]
    train_labels = train_images.parent.parent / "labels" / train_images.parent.name
    if not train_labels.exists():
        train_labels = train_images.with_name("labels")

    num_classes = data_cfg["nc"] + 1  # +1 for background
    class_names = data_cfg["names"]

    print("=" * 50, flush=True)
    print("Faster R-CNN 训练", flush=True)
    print("=" * 50, flush=True)
    print(f"数据集: {args.data}", flush=True)
    print(f"训练图片: {train_images}", flush=True)
    print(f"类别数: {num_classes - 1} ({class_names})", flush=True)
    print(f"轮数: {args.epochs}", flush=True)
    print(f"批次: {args.batch}", flush=True)
    print(f"设备: {args.device}", flush=True)
    print("=" * 50, flush=True)

    # 设备
    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device}")
    print(f"使用设备: {device}")

    # 数据集
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    dataset = UAVDataset(train_images, train_labels, transforms)
    data_loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, collate_fn=collate_fn, num_workers=0)
    print(f"训练样本数: {len(dataset)}")

    # 模型
    model = get_model(num_classes)
    model.to(device)

    # 优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 训练
    best_loss = float("inf")
    output_dir = Path(args.output) if args.output else base_path
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        start = time.time()
        loss = train_one_epoch(model, optimizer, data_loader, device, epoch)
        lr_scheduler.step()
        elapsed = time.time() - start
        print(f"Epoch {epoch} 完成, Loss: {loss:.4f}, 耗时: {elapsed:.1f}s")

        # 保存最佳模型
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), output_dir / "best.pt")
            print(f"保存最佳模型: {output_dir / 'best.pt'}")

    print(f"\n训练完成! 最佳模型: {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
