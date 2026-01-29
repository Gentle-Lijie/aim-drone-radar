"""
批量推理脚本（断点续传版）
- 自动跳过已处理的图片
- 增强错误处理，单张图片出错不影响整体流程

用法:
    python scripts/batch_inference_resume.py --output inference_results/run_XXXXXXXX_XXXXXX
"""

import os
import sys
import cv2
import torch
import pandas as pd
import warnings
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from PIL import Image

# 禁用警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 修复 truncated image 问题
Image.MAX_IMAGE_PIXELS = None
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 添加 YOLOv5 到路径
YOLOV5_PATH = Path(__file__).parent.parent / "yolov5"
sys.path.insert(0, str(YOLOV5_PATH))

# 配置
BASE_DIR = Path(__file__).parent.parent
DOWNLOADS_DIR = BASE_DIR / "downloads"
OUTPUT_DIR = BASE_DIR / "inference_results"

# 三个模型
MODELS = {
    "DroneDetection": BASE_DIR / "models" / "DroneDetection" / "best.pt",
    "model3": BASE_DIR / "models" / "model3.pt",
    "AntiUAV": BASE_DIR / "runs" / "yolov5" / "antiuav" / "weights" / "best.pt",
}

IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


def load_model(model_path, device='cuda'):
    print(f"加载模型: {model_path}")
    model = torch.hub.load(str(YOLOV5_PATH), 'custom', path=str(model_path), source='local')
    model.to(device)
    model.eval()
    return model


def get_image_files(folder):
    image_dir = folder / "image"
    if not image_dir.exists():
        return []
    files = []
    for f in image_dir.iterdir():
        if f.suffix.lower() in IMG_EXTENSIONS:
            files.append(f)
    return sorted(files)


def draw_detection(image, box, conf, class_name, color=(0, 255, 0)):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    label = f"{class_name} {conf:.2f}"
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(image, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
    cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return image


def safe_read_image(img_path):
    """安全读取图片，处理 truncated 等问题"""
    try:
        # 先用 PIL 打开修复 truncated
        pil_img = Image.open(str(img_path))
        pil_img.load()
        # 转换为 OpenCV 格式
        if pil_img.mode == 'RGBA':
            pil_img = pil_img.convert('RGB')
        import numpy as np
        image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return image
    except Exception as e:
        # 回退到 OpenCV
        image = cv2.imread(str(img_path))
        return image


def run_inference(model, image_path, conf_threshold=0.25):
    try:
        results = model(str(image_path))
        detections = results.pandas().xyxy[0]

        if len(detections) == 0:
            return None, None

        detections = detections[detections['confidence'] >= conf_threshold]

        if len(detections) == 0:
            return None, None

        best_idx = detections['confidence'].idxmax()
        best_det = detections.loc[best_idx]
        return best_det, detections
    except Exception as e:
        return None, None


def process_folder(folder_name, model_name, model, output_base, results_list, device):
    folder_path = DOWNLOADS_DIR / folder_name
    image_files = get_image_files(folder_path)

    if not image_files:
        print(f"  {folder_name}: 无图片文件")
        return

    output_folder = output_base / folder_name / model_name
    output_folder.mkdir(parents=True, exist_ok=True)

    # 检查已处理的图片
    processed = set(f.name for f in output_folder.iterdir() if f.suffix.lower() in IMG_EXTENSIONS)
    remaining = [f for f in image_files if f.name not in processed]

    if not remaining:
        print(f"  {folder_name}: 已全部处理 ({len(image_files)} 张)")
        return

    print(f"  处理 {folder_name} (剩余 {len(remaining)}/{len(image_files)} 张)...")

    for img_path in tqdm(remaining, desc=f"    {model_name}", leave=False):
        try:
            # 推理
            best_det, _ = run_inference(model, img_path)

            # 安全读取图片
            image = safe_read_image(img_path)
            if image is None:
                print(f"\n    跳过无法读取: {img_path.name}")
                continue

            # 记录结果
            result_row = {
                '文件夹': folder_name,
                '模型': model_name,
                '图片': img_path.name,
                '检测到': '否' if best_det is None else '是',
                '类别': '',
                '置信度': 0.0,
                'x_min': 0, 'y_min': 0, 'x_max': 0, 'y_max': 0,
                '中心_x': 0, '中心_y': 0,
            }

            if best_det is not None:
                box = [best_det['xmin'], best_det['ymin'], best_det['xmax'], best_det['ymax']]
                image = draw_detection(image, box, best_det['confidence'], best_det['name'])
                result_row.update({
                    '类别': best_det['name'],
                    '置信度': round(float(best_det['confidence']), 4),
                    'x_min': int(best_det['xmin']),
                    'y_min': int(best_det['ymin']),
                    'x_max': int(best_det['xmax']),
                    'y_max': int(best_det['ymax']),
                    '中心_x': int((best_det['xmin'] + best_det['xmax']) / 2),
                    '中心_y': int((best_det['ymin'] + best_det['ymax']) / 2),
                })

            results_list.append(result_row)

            # 保存图片
            output_path = output_folder / img_path.name
            cv2.imwrite(str(output_path), image)

        except Exception as e:
            print(f"\n    错误 {img_path.name}: {e}")
            continue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, help='续传的输出目录，如 inference_results/run_20260129_123456')
    args = parser.parse_args()

    print("=" * 60)
    print("YOLOv5 批量推理（断点续传版）")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 检查模型
    available_models = {}
    for name, path in MODELS.items():
        if path.exists():
            available_models[name] = path
            print(f"✓ {name}")
        else:
            print(f"✗ {name} (不存在)")

    if not available_models:
        print("错误: 没有可用的模型！")
        return

    subfolders = [f.name for f in DOWNLOADS_DIR.iterdir() if f.is_dir()]
    print(f"\n数据文件夹: {len(subfolders)} 个")

    # 确定输出目录
    if args.output:
        output_base = Path(args.output)
        if not output_base.is_absolute():
            output_base = BASE_DIR / args.output
        print(f"续传目录: {output_base}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = OUTPUT_DIR / f"run_{timestamp}"
        print(f"新建目录: {output_base}")

    output_base.mkdir(parents=True, exist_ok=True)

    all_results = []

    for model_name, model_path in available_models.items():
        print(f"\n{'=' * 40}")
        print(f"模型: {model_name}")
        print(f"{'=' * 40}")

        model = load_model(model_path, device)

        for folder_name in subfolders:
            process_folder(folder_name, model_name, model, output_base, all_results, device)

        del model
        if device == 'cuda':
            torch.cuda.empty_cache()

    # 保存 Excel（合并已有数据）
    print(f"\n保存 Excel...")
    excel_path = output_base / "detection_results.xlsx"

    # 如果已有 Excel，合并数据
    if excel_path.exists():
        try:
            existing_df = pd.read_excel(excel_path, sheet_name='全部结果')
            # 去重：按 文件夹+模型+图片 去重，保留新的
            new_df = pd.DataFrame(all_results)
            combined = pd.concat([existing_df, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=['文件夹', '模型', '图片'], keep='last')
            df = combined
            print(f"  合并已有数据，共 {len(df)} 条")
        except:
            df = pd.DataFrame(all_results)
    else:
        df = pd.DataFrame(all_results)

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='全部结果', index=False)

        for model_name in available_models.keys():
            model_df = df[df['模型'] == model_name]
            model_df.to_excel(writer, sheet_name=f'{model_name}', index=False)

        stats = df.groupby(['文件夹', '模型']).agg({
            '图片': 'count',
            '检测到': lambda x: (x == '是').sum()
        }).rename(columns={'图片': '总图片数', '检测到': '检测到数量'})
        stats['检测率'] = (stats['检测到数量'] / stats['总图片数'] * 100).round(2).astype(str) + '%'
        stats.to_excel(writer, sheet_name='统计')

    print(f"✓ Excel: {excel_path}")

    # 统计
    print(f"\n{'=' * 60}")
    total = len(df)
    detected = len(df[df['检测到'] == '是'])
    print(f"总图片: {total}, 检测到: {detected}, 检测率: {detected/total*100:.2f}%")
    print(f"\n完成！{output_base}")


if __name__ == "__main__":
    main()
