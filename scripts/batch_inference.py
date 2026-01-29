"""
批量推理脚本 - 对 downloads 文件夹下的大疆数据进行 YOLOv5 推理
- 每个子文件夹用所有模型各推理一遍
- 在原图上画框并保存
- 输出坐标和概率到 Excel（只保留概率最大的检测结果）

运行前请确保安装依赖:
    pip install openpyxl tqdm pandas opencv-python

运行命令:
    python scripts/batch_inference.py
"""

import os
import sys
import cv2
import torch
import pandas as pd
import warnings

# 禁用 FutureWarning (torch.cuda.amp.autocast 弃用警告)
warnings.filterwarnings("ignore", category=FutureWarning)
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# 添加 YOLOv5 到路径
YOLOV5_PATH = Path(__file__).parent.parent / "yolov5"
sys.path.insert(0, str(YOLOV5_PATH))

# 配置
BASE_DIR = Path(__file__).parent.parent
DOWNLOADS_DIR = BASE_DIR / "downloads"
OUTPUT_DIR = BASE_DIR / "inference_results"

# 可用模型列表（三个模型）
MODELS = {
    "DroneDetection": BASE_DIR / "models" / "DroneDetection" / "best.pt",
    "model3": BASE_DIR / "models" / "model3.pt",
    "AntiUAV": BASE_DIR / "runs" / "yolov5" / "antiuav" / "weights" / "best.pt",
}

# 支持的图片格式
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


def load_model(model_path, device='cuda'):
    """加载 YOLOv5 模型"""
    print(f"加载模型: {model_path}")
    model = torch.hub.load(str(YOLOV5_PATH), 'custom', path=str(model_path), source='local')
    model.to(device)
    model.eval()
    return model


def get_image_files(folder):
    """获取文件夹下所有图片文件"""
    image_dir = folder / "image"
    if not image_dir.exists():
        print(f"警告: {image_dir} 不存在，跳过")
        return []

    files = []
    for f in image_dir.iterdir():
        if f.suffix.lower() in IMG_EXTENSIONS:
            files.append(f)
    return sorted(files)


def draw_detection(image, box, conf, class_name, color=(0, 255, 0)):
    """在图片上画检测框"""
    x1, y1, x2, y2 = map(int, box)

    # 画框
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    # 标签背景
    label = f"{class_name} {conf:.2f}"
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(image, (x1, y1 - h - 10), (x1 + w, y1), color, -1)

    # 标签文字
    cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    return image


def run_inference(model, image_path, conf_threshold=0.25):
    """
    对单张图片进行推理
    返回: (最高置信度检测结果, 所有检测结果)
    """
    results = model(str(image_path))
    detections = results.pandas().xyxy[0]  # DataFrame: xmin, ymin, xmax, ymax, confidence, class, name

    if len(detections) == 0:
        return None, None

    # 过滤低置信度
    detections = detections[detections['confidence'] >= conf_threshold]

    if len(detections) == 0:
        return None, None

    # 找出置信度最高的
    best_idx = detections['confidence'].idxmax()
    best_det = detections.loc[best_idx]

    return best_det, detections


def process_folder(folder_name, model_name, model, output_base, results_list, device):
    """处理单个文件夹的所有图片"""
    folder_path = DOWNLOADS_DIR / folder_name
    image_files = get_image_files(folder_path)

    if not image_files:
        print(f"  {folder_name}: 无图片文件")
        return

    # 创建输出目录
    output_folder = output_base / folder_name / model_name
    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"  处理 {folder_name} ({len(image_files)} 张图片)...")

    for img_path in tqdm(image_files, desc=f"    {model_name}", leave=False):
        try:
            # 推理
            best_det, all_dets = run_inference(model, img_path)

            # 读取原图
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"    警告: 无法读取 {img_path}")
                continue

            # 记录结果
            result_row = {
                '文件夹': folder_name,
                '模型': model_name,
                '图片': img_path.name,
                '检测到': '否' if best_det is None else '是',
                '类别': '',
                '置信度': 0.0,
                'x_min': 0,
                'y_min': 0,
                'x_max': 0,
                'y_max': 0,
                '中心_x': 0,
                '中心_y': 0,
            }

            if best_det is not None:
                # 画框
                box = [best_det['xmin'], best_det['ymin'], best_det['xmax'], best_det['ymax']]
                image = draw_detection(image, box, best_det['confidence'], best_det['name'])

                # 更新结果
                result_row.update({
                    '类别': best_det['name'],
                    '置信度': round(best_det['confidence'], 4),
                    'x_min': int(best_det['xmin']),
                    'y_min': int(best_det['ymin']),
                    'x_max': int(best_det['xmax']),
                    'y_max': int(best_det['ymax']),
                    '中心_x': int((best_det['xmin'] + best_det['xmax']) / 2),
                    '中心_y': int((best_det['ymin'] + best_det['ymax']) / 2),
                })

            results_list.append(result_row)

            # 保存带框的图片
            output_path = output_folder / img_path.name
            cv2.imwrite(str(output_path), image)

        except Exception as e:
            print(f"    错误处理 {img_path.name}: {e}")
            continue


def main():
    print("=" * 60)
    print("YOLOv5 批量推理脚本")
    print("=" * 60)

    # 检查 CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 检查模型
    available_models = {}
    for name, path in MODELS.items():
        if path.exists():
            available_models[name] = path
            print(f"✓ 模型可用: {name} -> {path}")
        else:
            print(f"✗ 模型不存在: {name} -> {path}")

    if not available_models:
        print("错误: 没有可用的模型！")
        return

    # 获取所有子文件夹
    subfolders = [f.name for f in DOWNLOADS_DIR.iterdir() if f.is_dir()]
    print(f"\n找到 {len(subfolders)} 个数据文件夹:")
    for sf in subfolders:
        print(f"  - {sf}")

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = OUTPUT_DIR / f"run_{timestamp}"
    output_base.mkdir(parents=True, exist_ok=True)
    print(f"\n输出目录: {output_base}")

    # 存储所有结果
    all_results = []

    # 对每个模型进行推理
    for model_name, model_path in available_models.items():
        print(f"\n{'=' * 40}")
        print(f"模型: {model_name}")
        print(f"{'=' * 40}")

        # 加载模型
        model = load_model(model_path, device)

        # 处理每个文件夹
        for folder_name in subfolders:
            process_folder(folder_name, model_name, model, output_base, all_results, device)

        # 释放显存
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()

    # 保存 Excel
    print(f"\n保存结果到 Excel...")
    df = pd.DataFrame(all_results)
    excel_path = output_base / "detection_results.xlsx"

    # 创建 Excel writer 以便添加多个 sheet
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # 总表
        df.to_excel(writer, sheet_name='全部结果', index=False)

        # 按模型分表
        for model_name in available_models.keys():
            model_df = df[df['模型'] == model_name]
            model_df.to_excel(writer, sheet_name=f'{model_name}', index=False)

        # 统计表
        stats = df.groupby(['文件夹', '模型']).agg({
            '图片': 'count',
            '检测到': lambda x: (x == '是').sum()
        }).rename(columns={'图片': '总图片数', '检测到': '检测到数量'})
        stats['检测率'] = (stats['检测到数量'] / stats['总图片数'] * 100).round(2).astype(str) + '%'
        stats.to_excel(writer, sheet_name='统计')

    print(f"✓ Excel 已保存: {excel_path}")

    # 打印统计
    print(f"\n{'=' * 60}")
    print("统计汇总")
    print(f"{'=' * 60}")
    total_images = len(df)
    detected = len(df[df['检测到'] == '是'])
    print(f"总处理图片: {total_images}")
    print(f"检测到目标: {detected}")
    print(f"总检测率: {detected/total_images*100:.2f}%")

    print(f"\n完成！结果保存在: {output_base}")


if __name__ == "__main__":
    main()
