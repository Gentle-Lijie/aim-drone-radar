"""
增强训练脚本 - 从审查结果构建数据集 + 配置超参数 + 生成训练命令

功能:
  1. 从审查结果构建数据集 (正样本/负样本)
  2. 配置训练超参数 (预设模板或自定义)
  3. 生成训练命令 (在外部终端执行)

用法:
    python scripts/train_enhanced.py
"""

import json
import os
import random
import shutil
import sys
import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent.parent
YOLOV5_PATH = BASE_DIR / "yolov5"
INFERENCE_DIR = BASE_DIR / "inference_results"
DATASETS_DIR = BASE_DIR / "datasets"

# 图片后缀
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

# Excel 列名 (来自 batch_inference.py)
COL_FOLDER = '文件夹'
COL_MODEL = '模型'
COL_IMAGE = '图片'
COL_DETECTED = '检测到'
COL_CONFIDENCE = '置信度'
COL_XMIN = 'x_min'
COL_YMIN = 'y_min'
COL_XMAX = 'x_max'
COL_YMAX = 'y_max'

# 分类文件夹
CATEGORIES = ['Confirmed', 'FalsePositive', 'BadDetection', 'Detected', 'Empty', 'Missed']

# 超参数预设
HYP_PRESETS = {
    'balanced': {
        'name': '均衡模式 (默认)',
        'description': '标准配置，适合常规训练',
        'params': {
            'cls_pw': 1.0,
            'obj_pw': 1.0,
            'fl_gamma': 0.0,
            'cls': 0.5,
            'obj': 1.0,
            'box': 0.05,
            'label_smoothing': 0.0,
        },
        'image_weights': False,
    },
    'high_recall': {
        'name': '高召回模式 (减少漏检)',
        'description': '提高 obj 权重，优先减少漏检',
        'params': {
            'cls_pw': 1.0,
            'obj_pw': 2.0,
            'fl_gamma': 0.0,
            'cls': 0.5,
            'obj': 1.5,
            'box': 0.05,
            'label_smoothing': 0.0,
        },
        'image_weights': False,
    },
    'high_precision': {
        'name': '高精确模式 (减少误报)',
        'description': '使用 Focal Loss + 分类权重，减少误报',
        'params': {
            'cls_pw': 1.5,
            'obj_pw': 1.0,
            'fl_gamma': 1.5,
            'cls': 0.5,
            'obj': 1.0,
            'box': 0.05,
            'label_smoothing': 0.0,
        },
        'image_weights': True,
    },
}


# ========== 工具函数 ==========

def input_with_default(prompt, default):
    """带默认值的输入"""
    val = input(f"{prompt} [{default}]: ").strip()
    return val if val else str(default)


def input_choice(prompt, options, default=None):
    """选择菜单"""
    for i, opt in enumerate(options, 1):
        marker = " (默认)" if default is not None and i == default else ""
        print(f"  {i}. {opt}{marker}")
    while True:
        val = input(f"{prompt}: ").strip()
        if not val and default is not None:
            return default - 1
        try:
            idx = int(val) - 1
            if 0 <= idx < len(options):
                return idx
        except ValueError:
            pass
        print(f"  请输入 1-{len(options)}")


def scan_run_directory(run_dir):
    """扫描 run 目录，返回 {competition: {model: {category: [images]}}}"""
    structure = {}
    for competition_dir in sorted(run_dir.iterdir()):
        if not competition_dir.is_dir() or competition_dir.name.startswith('.'):
            continue
        # 跳过非赛区目录 (如文件)
        if not any((competition_dir / m).is_dir() for m in os.listdir(competition_dir)
                   if (competition_dir / m).is_dir()):
            continue

        comp_name = competition_dir.name
        structure[comp_name] = {}

        for model_dir in sorted(competition_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name
            structure[comp_name][model_name] = {}

            for cat_dir in sorted(model_dir.iterdir()):
                if not cat_dir.is_dir():
                    continue
                cat_name = cat_dir.name
                images = [f for f in cat_dir.iterdir()
                          if f.suffix.lower() in IMG_EXTENSIONS]
                if images:
                    structure[comp_name][model_name][cat_name] = images

    return structure


def load_excel_bbox(run_dir, model_name):
    """从 detection_results.xlsx 加载指定模型的 bbox 数据
    返回: {原始文件名: (x_min, y_min, x_max, y_max, img_w, img_h)}
    """
    excel_path = run_dir / "detection_results.xlsx"
    if not excel_path.exists():
        print(f"  警告: 未找到 {excel_path}")
        return {}

    df = pd.read_excel(excel_path, sheet_name='全部结果')
    # 只取指定模型且有检测结果的行
    mask = (df[COL_MODEL] == model_name) & (df[COL_DETECTED] == '是')
    model_df = df[mask]

    bbox_map = {}
    for _, row in model_df.iterrows():
        img_name = row[COL_IMAGE]
        x_min = row[COL_XMIN]
        y_min = row[COL_YMIN]
        x_max = row[COL_XMAX]
        y_max = row[COL_YMAX]
        if x_min == 0 and y_min == 0 and x_max == 0 and y_max == 0:
            continue
        bbox_map[img_name] = (x_min, y_min, x_max, y_max)

    return bbox_map


def pixel_to_yolo(x_min, y_min, x_max, y_max, img_w, img_h):
    """像素坐标转 YOLO 归一化格式: class x_center y_center width height"""
    x_center = ((x_min + x_max) / 2.0) / img_w
    y_center = ((y_min + y_max) / 2.0) / img_h
    width = (x_max - x_min) / img_w
    height = (y_max - y_min) / img_h
    # 确保值在 [0, 1] 范围
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))
    return x_center, y_center, width, height


def get_image_size(img_path):
    """获取图片尺寸 (宽, 高)，不依赖 OpenCV"""
    try:
        from PIL import Image
        with Image.open(img_path) as img:
            return img.size  # (width, height)
    except Exception:
        pass
    try:
        import cv2
        import numpy as np
        img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            h, w = img.shape[:2]
            return (w, h)
    except Exception:
        pass
    return None


def extract_original_name(filename):
    """从带置信度前缀的文件名提取原始文件名
    如 '0.92_TDTVsI Hiter_BO5_3_173.jpg' → 'TDTVsI Hiter_BO5_3_173.jpg'
    """
    name = filename
    # 匹配 0.XX_ 前缀
    if len(name) > 5 and name[0] == '0' and name[1] == '.':
        parts = name.split('_', 1)
        if len(parts) == 2:
            try:
                float(parts[0])
                return parts[1]
            except ValueError:
                pass
    return name


# ========== 功能 1: 构建数据集 ==========

def build_dataset():
    """从审查结果构建数据集"""
    print()
    print("=" * 50)
    print("从审查结果构建数据集")
    print("=" * 50)

    # 选择 run 目录
    runs = sorted(INFERENCE_DIR.iterdir()) if INFERENCE_DIR.exists() else []
    runs = [r for r in runs if r.is_dir()]
    if not runs:
        print("错误: 未找到推理结果目录")
        return None

    print("\n可用的 run 目录:")
    for i, r in enumerate(runs, 1):
        print(f"  {i}. {r.name}")
    idx = input_choice("选择 run 目录", [r.name for r in runs], default=len(runs))
    run_dir = runs[idx]
    print(f"\n已选择: {run_dir.name}")

    # 扫描目录结构
    print("\n扫描目录结构...")
    structure = scan_run_directory(run_dir)
    if not structure:
        print("错误: 未找到有效的赛区/模型/分类目录")
        return None

    # 打印统计
    print("\n目录统计:")
    all_models = set()
    for comp_name, models in structure.items():
        print(f"\n  {comp_name}:")
        for model_name, categories in models.items():
            all_models.add(model_name)
            cats_info = []
            for cat_name, imgs in categories.items():
                cats_info.append(f"{cat_name}={len(imgs)}")
            print(f"    {model_name}: {', '.join(cats_info)}")

    # 选择源模型
    model_list = sorted(all_models)
    print(f"\n可用模型: {', '.join(model_list)}")
    idx = input_choice("选择源模型", model_list, default=1)
    source_model = model_list[idx]
    print(f"\n已选择模型: {source_model}")

    # 收集该模型下所有分类和图片
    model_categories = {}  # {category: [img_paths]}
    for comp_name, models in structure.items():
        if source_model in models:
            for cat_name, imgs in models[source_model].items():
                if cat_name not in model_categories:
                    model_categories[cat_name] = []
                model_categories[cat_name].extend(imgs)

    print(f"\n{source_model} 模型各分类图片数:")
    for cat, imgs in sorted(model_categories.items()):
        print(f"  {cat}: {len(imgs)} 张")

    # 加载 Excel bbox 数据
    print(f"\n加载 detection_results.xlsx 中 {source_model} 的 bbox 数据...")
    bbox_map = load_excel_bbox(run_dir, source_model)
    print(f"  已加载 {len(bbox_map)} 条 bbox 记录")

    # 交互式选择每个分类的角色
    print("\n===  为每个分类指定角色  ===")
    print("  正样本: 包含目标，生成 YOLO 标签 (从 Excel 读取 bbox)")
    print("  负样本: 不包含目标，生成空标签文件 (背景样本)")
    print("  跳过:   不加入数据集")
    print()

    role_options = ['正样本', '负样本', '跳过']

    # 默认角色
    default_roles = {
        'Confirmed': 0,   # 正样本
        'FalsePositive': 1,  # 负样本
        'BadDetection': 2,   # 跳过
        'Detected': 2,       # 跳过
        'Empty': 2,          # 跳过
        'Missed': 2,         # 跳过
    }

    category_roles = {}  # {category: 'positive'|'negative'|'skip'}
    role_map = {0: 'positive', 1: 'negative', 2: 'skip'}

    for cat_name, imgs in sorted(model_categories.items()):
        default = default_roles.get(cat_name, 3)
        if default > 2:
            default = 3  # 跳过
        print(f"{cat_name} ({len(imgs)} 张):")
        idx = input_choice("  角色", role_options, default=default + 1)
        category_roles[cat_name] = role_map[idx]
        print()

    # 统计
    positive_images = []
    negative_images = []
    for cat_name, role in category_roles.items():
        if role == 'positive':
            positive_images.extend(model_categories.get(cat_name, []))
        elif role == 'negative':
            negative_images.extend(model_categories.get(cat_name, []))

    if not positive_images and not negative_images:
        print("错误: 没有选择任何图片")
        return None

    print(f"\n样本统计:")
    print(f"  正样本: {len(positive_images)} 张")
    print(f"  负样本: {len(negative_images)} 张")
    total = len(positive_images) + len(negative_images)
    if total > 0:
        print(f"  正负比例: {len(positive_images)/total*100:.1f}% / {len(negative_images)/total*100:.1f}%")

    # 验证/训练划分比例
    val_ratio = float(input_with_default("\n验证集比例", "0.2"))
    val_ratio = max(0.05, min(0.5, val_ratio))

    # 创建数据集目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_dir = DATASETS_DIR / f"robomaster_{timestamp}"
    train_img_dir = dataset_dir / "train" / "images"
    train_lbl_dir = dataset_dir / "train" / "labels"
    val_img_dir = dataset_dir / "val" / "images"
    val_lbl_dir = dataset_dir / "val" / "labels"

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n输出目录: {dataset_dir}")
    print("正在构建数据集...")

    # 合并并打乱
    all_items = []
    for img_path in positive_images:
        all_items.append(('positive', img_path))
    for img_path in negative_images:
        all_items.append(('negative', img_path))
    random.shuffle(all_items)

    # 划分 train/val
    val_count = max(1, int(len(all_items) * val_ratio))
    val_items = all_items[:val_count]
    train_items = all_items[val_count:]

    stats = {
        'total': len(all_items),
        'train_positive': 0, 'train_negative': 0,
        'val_positive': 0, 'val_negative': 0,
        'bbox_found': 0, 'bbox_missing': 0,
    }

    def process_item(role, img_path, img_dir, lbl_dir, split):
        """处理单个图片: 复制图片 + 生成标签"""
        # 避免文件名冲突: 加入赛区前缀
        # 从路径提取赛区信息: run_dir/competition/model/category/image.jpg
        rel = img_path.relative_to(run_dir)
        parts = rel.parts
        if len(parts) >= 3:
            comp_short = parts[0].replace("robomaster_", "").replace(" ", "_")[:20]
            safe_name = f"{comp_short}_{img_path.name}"
        else:
            safe_name = img_path.name

        # 确保唯一性
        dst_img = img_dir / safe_name
        counter = 0
        while dst_img.exists():
            counter += 1
            stem = Path(safe_name).stem
            suffix = Path(safe_name).suffix
            dst_img = img_dir / f"{stem}_{counter}{suffix}"
            safe_name = dst_img.name

        # 复制图片
        shutil.copy2(str(img_path), str(dst_img))

        # 生成标签文件
        lbl_name = Path(safe_name).stem + ".txt"
        dst_lbl = lbl_dir / lbl_name

        if role == 'positive':
            # 尝试从 Excel 获取 bbox
            original_name = extract_original_name(img_path.name)
            bbox = bbox_map.get(original_name) or bbox_map.get(img_path.name)

            if bbox:
                x_min, y_min, x_max, y_max = bbox
                # 获取图片尺寸
                size = get_image_size(dst_img)
                if size:
                    img_w, img_h = size
                    xc, yc, w, h = pixel_to_yolo(x_min, y_min, x_max, y_max, img_w, img_h)
                    with open(dst_lbl, 'w') as f:
                        f.write(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                    stats['bbox_found'] += 1
                else:
                    # 无法获取尺寸，写空标签
                    dst_lbl.touch()
                    stats['bbox_missing'] += 1
            else:
                # 没有 bbox 数据，写空标签
                dst_lbl.touch()
                stats['bbox_missing'] += 1

            stats[f'{split}_positive'] += 1
        else:
            # 负样本: 空标签文件
            dst_lbl.touch()
            stats[f'{split}_negative'] += 1

    # 处理训练集
    for role, img_path in train_items:
        process_item(role, img_path, train_img_dir, train_lbl_dir, 'train')

    # 处理验证集
    for role, img_path in val_items:
        process_item(role, img_path, val_img_dir, val_lbl_dir, 'val')

    # 生成 data.yaml
    data_yaml = {
        'path': str(dataset_dir.resolve()).replace('\\', '/'),
        'train': 'train/images',
        'val': 'val/images',
        'nc': 1,
        'names': ['drone'],
    }
    yaml_path = dataset_dir / "data.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

    # 保存构建信息
    build_info = {
        'created': datetime.now().isoformat(),
        'run_dir': str(run_dir),
        'source_model': source_model,
        'category_roles': category_roles,
        'val_ratio': val_ratio,
        'stats': stats,
    }
    with open(dataset_dir / "build_info.json", 'w', encoding='utf-8') as f:
        json.dump(build_info, f, ensure_ascii=False, indent=2)

    # 打印结果
    print()
    print("=" * 50)
    print("数据集构建完成!")
    print("=" * 50)
    print(f"  目录: {dataset_dir}")
    print(f"  data.yaml: {yaml_path}")
    print()
    print(f"  训练集: {stats['train_positive'] + stats['train_negative']} 张")
    print(f"    正样本: {stats['train_positive']}")
    print(f"    负样本: {stats['train_negative']}")
    print(f"  验证集: {stats['val_positive'] + stats['val_negative']} 张")
    print(f"    正样本: {stats['val_positive']}")
    print(f"    负样本: {stats['val_negative']}")
    print()
    print(f"  bbox 标签: 成功 {stats['bbox_found']}, 缺失 {stats['bbox_missing']}")
    if stats['bbox_missing'] > 0:
        print(f"  (缺失的正样本将使用空标签，相当于负样本)")

    return str(yaml_path)


# ========== 功能 2: 配置超参数 ==========

def configure_hyperparams():
    """交互式配置训练超参数"""
    print()
    print("=" * 50)
    print("配置训练超参数")
    print("=" * 50)

    # 选择预设或自定义
    options = []
    preset_keys = []
    for key, preset in HYP_PRESETS.items():
        options.append(f"{preset['name']}: {preset['description']}")
        preset_keys.append(key)
    options.append("自定义: 逐项配置每个参数")

    idx = input_choice("\n选择超参数模板", options, default=1)

    if idx < len(preset_keys):
        # 使用预设
        preset = HYP_PRESETS[preset_keys[idx]]
        params = dict(preset['params'])
        image_weights = preset['image_weights']
        print(f"\n已选择: {preset['name']}")
    else:
        # 自定义
        print("\n逐项配置 (直接回车使用默认值):")
        params = {}
        params['cls_pw'] = float(input_with_default("  cls_pw (分类正样本权重)", 1.0))
        params['obj_pw'] = float(input_with_default("  obj_pw (目标正样本权重)", 1.0))
        params['fl_gamma'] = float(input_with_default("  fl_gamma (Focal Loss gamma, 0=关闭)", 0.0))
        params['cls'] = float(input_with_default("  cls (分类损失增益)", 0.5))
        params['obj'] = float(input_with_default("  obj (目标损失增益)", 1.0))
        params['box'] = float(input_with_default("  box (框回归损失增益)", 0.05))
        params['label_smoothing'] = float(input_with_default("  label_smoothing (标签平滑)", 0.0))
        iw = input_with_default("  image-weights (按类别加权采样, y/n)", "n")
        image_weights = iw.lower() in ('y', 'yes', '1')

    # 打印配置摘要
    print("\n超参数配置:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    print(f"  image-weights: {'开启' if image_weights else '关闭'}")

    # 加载基础 hyp 文件并覆盖
    base_hyp_path = YOLOV5_PATH / "data" / "hyps" / "hyp.scratch-low.yaml"
    if base_hyp_path.exists():
        with open(base_hyp_path, 'r') as f:
            hyp = yaml.safe_load(f)
    else:
        # 使用基本默认值
        hyp = {
            'lr0': 0.01, 'lrf': 0.01, 'momentum': 0.937,
            'weight_decay': 0.0005, 'warmup_epochs': 3.0,
            'warmup_momentum': 0.8, 'warmup_bias_lr': 0.1,
            'box': 0.05, 'cls': 0.5, 'cls_pw': 1.0,
            'obj': 1.0, 'obj_pw': 1.0, 'iou_t': 0.20,
            'anchor_t': 4.0, 'fl_gamma': 0.0,
            'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4,
            'degrees': 0.0, 'translate': 0.1, 'scale': 0.5,
            'shear': 0.0, 'perspective': 0.0, 'flipud': 0.0,
            'fliplr': 0.5, 'mosaic': 1.0, 'mixup': 0.0,
            'copy_paste': 0.0,
        }

    # 覆盖用户选择的参数
    for k, v in params.items():
        if k in hyp:
            hyp[k] = v

    # 保存到 datasets 目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    hyp_path = DATASETS_DIR / f"hyp_{timestamp}.yaml"
    with open(hyp_path, 'w') as f:
        yaml.dump(hyp, f, default_flow_style=False)

    print(f"\n已保存: {hyp_path}")

    return str(hyp_path), image_weights, params.get('label_smoothing', 0.0)


# ========== 功能 3: 开始训练 ==========

def start_training(data_yaml=None, hyp_yaml=None, image_weights=False, label_smoothing=0.0):
    """组装训练命令"""
    print()
    print("=" * 50)
    print("配置训练参数")
    print("=" * 50)

    # data.yaml
    if not data_yaml:
        data_yaml = input_with_default("\ndata.yaml 路径", "datasets/robomaster_*/data.yaml")
        # 尝试展开通配符
        candidates = sorted(BASE_DIR.glob(data_yaml))
        if candidates:
            data_yaml = str(candidates[-1])
            print(f"  已找到: {data_yaml}")
        elif not Path(data_yaml).is_absolute():
            data_yaml = str(BASE_DIR / data_yaml)

    # hyp.yaml
    if not hyp_yaml:
        use_custom = input_with_default("使用自定义超参数? (y/n)", "n")
        if use_custom.lower() in ('y', 'yes'):
            hyp_yaml, image_weights, label_smoothing = configure_hyperparams()

    # 其他参数
    weights = input_with_default("预训练权重", "yolov5s.pt")
    epochs = input_with_default("训练轮数", "100")
    batch = input_with_default("批次大小", "16")
    imgsz = input_with_default("图像大小", "640")
    device = input_with_default("设备 (0=GPU, cpu=CPU)", "0")
    workers = input_with_default("DataLoader workers", "4")
    project = input_with_default("输出目录", "runs/yolov5")
    name = input_with_default("实验名称", "robomaster")

    # 构建命令
    train_script = YOLOV5_PATH / "train.py"
    cmd_parts = [
        sys.executable, str(train_script),
        "--data", data_yaml,
        "--weights", weights,
        "--epochs", epochs,
        "--batch-size", batch,
        "--img", imgsz,
        "--device", device,
        "--workers", workers,
        "--project", project,
        "--name", name,
        "--exist-ok",
    ]

    if hyp_yaml:
        cmd_parts.extend(["--hyp", hyp_yaml])

    if image_weights:
        cmd_parts.append("--image-weights")

    if label_smoothing > 0:
        cmd_parts.extend(["--label-smoothing", str(label_smoothing)])

    cmd_str = " ".join(cmd_parts)

    print()
    print("=" * 60)
    print("训练命令 (请复制到外部终端执行)")
    print("=" * 60)
    print()
    print(cmd_str)
    print()
    print("=" * 60)
    print()
    print("提示:")
    print("  1. 请在外部命令行窗口执行上述命令")
    print("  2. 训练完成后，最佳模型保存在:")
    print(f"     {project}/{name}/weights/best.pt")
    print("  3. 可使用 tensorboard 查看训练过程:")
    print(f"     tensorboard --logdir {project}/{name}")

    return cmd_str


# ========== 主菜单 ==========

def main():
    print()
    print("=" * 50)
    print("  AIM Radar 增强训练工具")
    print("=" * 50)

    data_yaml = None
    hyp_yaml = None
    image_weights = False
    label_smoothing = 0.0

    while True:
        print()
        print("=== 增强训练配置 ===")
        print("1. 从审查结果构建数据集")
        print("2. 配置训练超参数")
        print("3. 开始训练")
        print("4. 退出")

        choice = input("\n选择功能 (1-4): ").strip()

        if choice == '1':
            result = build_dataset()
            if result:
                data_yaml = result
        elif choice == '2':
            result = configure_hyperparams()
            if result:
                hyp_yaml, image_weights, label_smoothing = result
        elif choice == '3':
            start_training(data_yaml, hyp_yaml, image_weights, label_smoothing)
        elif choice == '4':
            print("退出。")
            break
        else:
            print("请输入 1-4")


if __name__ == "__main__":
    main()
