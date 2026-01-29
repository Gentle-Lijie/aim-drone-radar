"""
按检测结果整理图片
- 移动: 按 Detected/Empty 分类
- 重命名: 添加置信度前缀

用法:
    python scripts/organize_results.py --input inference_results/run_20260129_195603
    python scripts/organize_results.py --input inference_results/run_20260129_195603 --move-only
    python scripts/organize_results.py --input inference_results/run_20260129_195603 --rename-only
"""

import argparse
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent


def find_image(input_dir, folder, model, image):
    """查找图片，支持多种可能位置"""
    # 可能的位置
    candidates = [
        input_dir / folder / model / image,                    # 原始位置
        input_dir / folder / model / "Detected" / image,       # 已移动到 Detected
        input_dir / folder / model / "Empty" / image,          # 已移动到 Empty
    ]

    for path in candidates:
        if path.exists():
            return path

    # 检查是否已重命名（带置信度前缀）
    for subdir in ["Detected", "Empty", ""]:
        search_dir = input_dir / folder / model
        if subdir:
            search_dir = search_dir / subdir
        if search_dir.exists():
            for f in search_dir.iterdir():
                # 检查是否是重命名后的文件（格式: 0.XX_原文件名）
                if f.name.endswith(image) and f.name[0].isdigit():
                    return f

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='推理结果目录')
    parser.add_argument('--move-only', action='store_true', help='只移动，不重命名')
    parser.add_argument('--rename-only', action='store_true', help='只重命名，不移动')
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_absolute():
        input_dir = BASE_DIR / args.input

    excel_path = input_dir / "detection_results.xlsx"
    if not excel_path.exists():
        print(f"错误: 找不到 {excel_path}")
        return

    # 确定执行什么操作
    do_move = not args.rename_only
    do_rename = not args.move_only

    print(f"读取 Excel: {excel_path}")
    df = pd.read_excel(excel_path, sheet_name='全部结果')
    print(f"共 {len(df)} 条记录")
    print(f"操作: {'移动' if do_move else ''} {'重命名' if do_rename else ''}")

    # 统计
    stats = {'detected': 0, 'empty': 0, 'moved': 0, 'renamed': 0, 'skipped': 0, 'error': 0}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="处理中"):
        folder = row['文件夹']
        model = row['模型']
        image = row['图片']
        is_detected = row['检测到'] == '是'
        confidence = row['置信度'] if is_detected else 0.0

        if is_detected:
            stats['detected'] += 1
        else:
            stats['empty'] += 1

        # 查找源文件
        src = find_image(input_dir, folder, model, image)
        if src is None:
            stats['error'] += 1
            continue

        # 确定目标
        current_in_subdir = src.parent.name in ["Detected", "Empty"]
        already_renamed = src.name[0].isdigit() and src.name[1] == '.' and src.name[4] == '_'

        # 计算目标路径
        if do_move and not current_in_subdir:
            # 需要移动
            target_subdir = "Detected" if is_detected else "Empty"
            dst_dir = input_dir / folder / model / target_subdir
        else:
            # 保持当前目录
            dst_dir = src.parent

        dst_dir.mkdir(parents=True, exist_ok=True)

        # 计算目标文件名
        if do_rename and not already_renamed:
            # 需要重命名
            new_name = f"{confidence:.2f}_{image}"
        else:
            # 保持当前文件名
            new_name = src.name

        dst = dst_dir / new_name

        # 执行操作
        if src == dst:
            stats['skipped'] += 1
            continue

        try:
            shutil.move(str(src), str(dst))
            if do_move and not current_in_subdir:
                stats['moved'] += 1
            if do_rename and not already_renamed:
                stats['renamed'] += 1
        except Exception as e:
            print(f"\n错误: {src} -> {dst}: {e}")
            stats['error'] += 1

    print(f"\n完成!")
    print(f"  检测到: {stats['detected']}")
    print(f"  未检测: {stats['empty']}")
    print(f"  已移动: {stats['moved']}")
    print(f"  已重命名: {stats['renamed']}")
    print(f"  跳过: {stats['skipped']}")
    if stats['error']:
        print(f"  错误: {stats['error']}")


if __name__ == "__main__":
    main()
