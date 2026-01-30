"""
人工审批检测结果

对 Detected/Empty 中的图片进行人工判断：
  - Detected 中的真阳性       → Confirmed/
  - Detected 中的误报         → FalsePositive/
  - Detected 中有目标但识别错误 → BadDetection/
  - Empty 中实际有目标         → Missed/

操作方式 (键盘快捷键):
  Y / 右箭头  = 确认 (识别正确，图中有无人机)
  N / 左箭头  = 否定 (图中没有无人机，误报)
  W / 下箭头  = 识别错误 (图中有无人机但识别结果有误)  [仅 detected/confirmed 模式]
  U / 上箭头  = 撤销上一张
  Q / ESC     = 退出并保存进度

用法:
    # 审查 Detected (确认真阳性 / 排除误报 / 标记识别错误)
    python scripts/manual_review.py --input inference_results/run_20260129_195603 --mode detected

    # 审查 Empty (找出漏检)
    python scripts/manual_review.py --input inference_results/run_20260129_195603 --mode empty

    # 重新审查已确认的图片
    python scripts/manual_review.py --input inference_results/run_20260129_195603 --mode confirmed

    # 指定模型和赛区
    python scripts/manual_review.py --input inference_results/run_20260129_195603 --mode detected --model DroneDetection
    python scripts/manual_review.py --input inference_results/run_20260129_195603 --mode detected --competition "Final Tournament"
"""

import argparse
import json
import shutil
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent.parent

# 键值定义
KEY_Y = ord('y')
KEY_N = ord('n')
KEY_U = ord('u')
KEY_Q = ord('q')
KEY_W = ord('w')
KEY_ESC = 27
KEY_RIGHT = 2555904  # Windows OpenCV 右箭头
KEY_LEFT = 2424832   # Windows OpenCV 左箭头
KEY_DOWN = 2621440   # Windows OpenCV 下箭头
KEY_UP = 2490368     # Windows OpenCV 上箭头


def check_organized(input_dir):
    """检查 run 目录是否已经被 organize_results.py 处理过"""
    # 遍历寻找 Detected 或 Empty 子目录
    for p in input_dir.rglob("Detected"):
        if p.is_dir() and any(p.iterdir()):
            return True
    for p in input_dir.rglob("Empty"):
        if p.is_dir() and any(p.iterdir()):
            return True
    return False


def collect_images(input_dir, mode, model_filter=None, competition_filter=None):
    """收集待审查的图片列表"""
    images = []
    if mode == "detected":
        target_subdir = "Detected"
    elif mode == "empty":
        target_subdir = "Empty"
    else:  # confirmed
        target_subdir = "Confirmed"

    for subdir in sorted(input_dir.rglob(target_subdir)):
        if not subdir.is_dir():
            continue

        # 路径结构: input_dir / competition / model / Detected|Empty
        model_dir = subdir.parent
        model_name = model_dir.name
        competition_dir = model_dir.parent
        competition_name = competition_dir.name

        # 过滤
        if model_filter and model_filter.lower() not in model_name.lower():
            continue
        if competition_filter and competition_filter.lower() not in competition_name.lower():
            continue

        for img_path in subdir.iterdir():
            if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'):
                images.append({
                    'path': img_path,
                    'model': model_name,
                    'competition': competition_name,
                })

    # 按文件名降序排列 (文件名以置信度开头，如 0.91_xxx.jpg)
    images.sort(key=lambda x: x['path'].name, reverse=True)
    return images


def load_progress(progress_file):
    """加载审查进度"""
    if progress_file.exists():
        with open(progress_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'reviewed': {}, 'stats': {'confirmed': 0, 'rejected': 0, 'bad_detection': 0, 'total': 0}}


def save_progress(progress_file, progress):
    """保存审查进度"""
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def draw_info_bar(img, text, bar_height=40):
    """在图片顶部绘制信息栏"""
    h, w = img.shape[:2]
    result = np.zeros((h + bar_height, w, 3), dtype=np.uint8)
    # 深灰背景
    result[:bar_height, :] = (40, 40, 40)
    result[bar_height:, :] = img

    # 文字 (OpenCV 不支持中文，用英文)
    cv2.putText(result, text, (10, bar_height - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return result


def show_help_dialog(window_name, mode):
    """显示快捷键说明弹窗，按任意键关闭"""
    has_w_key = mode in ("detected", "confirmed")
    w, h = 560, 380 if has_w_key else 340
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (50, 50, 50)

    # 标题
    cv2.putText(img, "Manual Review - Keyboard Shortcuts", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2, cv2.LINE_AA)

    # 分隔线
    cv2.line(img, (30, 55), (w - 30, 55), (100, 100, 100), 1)

    if mode == "detected":
        mode_text = "Mode: DETECTED  (confirm / reject / mark bad detection)"
    elif mode == "confirmed":
        mode_text = "Mode: CONFIRMED  (re-review previously confirmed images)"
    else:
        mode_text = "Mode: EMPTY  (find missed targets in empty images)"
    cv2.putText(img, mode_text, (30, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

    shortcuts = [
        ("Y  /  Right Arrow", "YES - correct detection (drone present)"),
        ("N  /  Left Arrow",  "NO  - false positive (no drone)"),
    ]
    if has_w_key:
        shortcuts.append(
            ("W  /  Down Arrow",  "WRONG - drone exists but detection is bad"),
        )
    shortcuts += [
        ("U  /  Up Arrow",    "UNDO last decision"),
        ("Q  /  ESC",         "QUIT and save progress"),
    ]

    y = 130
    for key_text, desc in shortcuts:
        # 按键高亮
        cv2.putText(img, key_text, (50, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 120), 1, cv2.LINE_AA)
        cv2.putText(img, desc, (270, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        y += 40

    # 底部提示
    cv2.line(img, (30, h - 60), (w - 30, h - 60), (100, 100, 100), 1)
    cv2.putText(img, "Progress is auto-saved every 50 images.", (30, h - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1, cv2.LINE_AA)
    cv2.putText(img, "Press any key to start reviewing...", (120, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 200, 255), 1, cv2.LINE_AA)

    cv2.imshow(window_name, img)
    cv2.waitKey(0)


def run_review(images, input_dir, mode, progress_file):
    """主审查循环"""
    progress = load_progress(progress_file)

    # 过滤已审查的图片
    pending = []
    for img_info in images:
        key = str(img_info['path'].relative_to(input_dir))
        if key not in progress['reviewed']:
            pending.append(img_info)

    if not pending:
        print("所有图片已审查完毕!")
        print_stats(progress)
        return

    already_done = len(images) - len(pending)
    print(f"共 {len(images)} 张图片, 已审查 {already_done}, 剩余 {len(pending)}")
    print()
    has_w_key = mode in ("detected", "confirmed")
    print("操作说明:")
    print("  Y / 右箭头  = 确认 (识别正确)")
    print("  N / 左箭头  = 否定 (图中没有无人机)")
    if has_w_key:
        print("  W / 下箭头  = 识别错误 (有无人机但识别有误)")
    print("  U / 上箭头  = 撤销上一张")
    print("  Q / ESC     = 退出并保存")
    print()

    window_name = "Manual Review"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 760)

    # 显示快捷键说明弹窗
    show_help_dialog(window_name, mode)

    idx = 0
    history = []  # 撤销历史

    while idx < len(pending):
        img_info = pending[idx]
        img_path = img_info['path']
        rel_key = str(img_path.relative_to(input_dir))

        # 读取图片
        img = cv2.imdecode(
            np.fromfile(str(img_path), dtype=np.uint8),
            cv2.IMREAD_COLOR
        )
        if img is None:
            print(f"无法读取: {img_path}")
            idx += 1
            continue

        # 信息栏
        reviewed_count = already_done + idx
        total = len(images)
        w_hint = "  W/Down=Wrong" if has_w_key else ""
        info = (f"[{reviewed_count + 1}/{total}] "
                f"{img_info['competition']} | {img_info['model']} | {img_path.name}  "
                f"|| Y/Right=Yes  N/Left=No{w_hint}  U/Up=Undo  Q=Quit")
        display = draw_info_bar(img, info)

        cv2.imshow(window_name, display)
        key = cv2.waitKeyEx(0)

        if key in (KEY_Y, KEY_RIGHT):
            # 确认: 有目标
            progress['reviewed'][rel_key] = 'yes'
            progress['stats']['confirmed'] += 1
            progress['stats']['total'] += 1
            history.append((idx, rel_key))
            idx += 1

        elif key in (KEY_N, KEY_LEFT):
            # 否定: 没有目标
            progress['reviewed'][rel_key] = 'no'
            progress['stats']['rejected'] += 1
            progress['stats']['total'] += 1
            history.append((idx, rel_key))
            idx += 1

        elif key in (KEY_W, KEY_DOWN) and has_w_key:
            # 有目标但识别错误
            progress['reviewed'][rel_key] = 'bad'
            progress['stats']['bad_detection'] = progress['stats'].get('bad_detection', 0) + 1
            progress['stats']['total'] += 1
            history.append((idx, rel_key))
            idx += 1

        elif key in (KEY_U, KEY_UP):
            # 撤销
            if history:
                prev_idx, prev_key = history.pop()
                verdict = progress['reviewed'].pop(prev_key, None)
                if verdict == 'yes':
                    progress['stats']['confirmed'] -= 1
                elif verdict == 'no':
                    progress['stats']['rejected'] -= 1
                elif verdict == 'bad':
                    progress['stats']['bad_detection'] = progress['stats'].get('bad_detection', 0) - 1
                progress['stats']['total'] -= 1
                idx = prev_idx
                print(f"已撤销: {prev_key}")

        elif key in (KEY_Q, KEY_ESC):
            break

        # 每 50 张自动保存
        if progress['stats']['total'] % 50 == 0 and progress['stats']['total'] > 0:
            save_progress(progress_file, progress)

    cv2.destroyAllWindows()
    save_progress(progress_file, progress)
    print()
    print_stats(progress)


def print_stats(progress):
    """打印统计信息"""
    stats = progress['stats']
    print(f"审查统计:")
    print(f"  已审查: {stats['total']}")
    print(f"  确认正确: {stats['confirmed']}")
    print(f"  确认无目标: {stats['rejected']}")
    bad = stats.get('bad_detection', 0)
    if bad:
        print(f"  识别错误: {bad}")


def apply_results(input_dir, mode, progress_file):
    """根据审查结果批量移动图片"""
    progress = load_progress(progress_file)

    if not progress['reviewed']:
        print("没有审查记录，请先运行审查。")
        return

    moved = {'confirmed': 0, 'false_positive': 0, 'bad_detection': 0, 'missed': 0, 'error': 0}

    for rel_key, verdict in progress['reviewed'].items():
        src = input_dir / rel_key
        if not src.exists():
            # 可能已经移动过
            continue

        # 路径结构: competition/model/{Detected|Empty|Confirmed}/image.jpg
        # 目标: competition/model/{Confirmed|FalsePositive|BadDetection|Missed}/image.jpg
        model_dir = src.parent.parent

        if mode in ("detected", "confirmed"):
            if verdict == 'yes':
                dst_dir = model_dir / "Confirmed"
                moved['confirmed'] += 1
            elif verdict == 'bad':
                dst_dir = model_dir / "BadDetection"
                moved['bad_detection'] += 1
            else:
                dst_dir = model_dir / "FalsePositive"
                moved['false_positive'] += 1
        else:  # empty
            if verdict == 'yes':
                # Empty 中实际有目标 → 漏检
                dst_dir = model_dir / "Missed"
                moved['missed'] += 1
            else:
                # Empty 中确认没有目标，不移动
                continue

        # 如果源文件已经在目标目录，跳过
        if src.parent == dst_dir:
            continue

        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name

        try:
            shutil.move(str(src), str(dst))
        except Exception as e:
            print(f"移动失败: {src} -> {dst}: {e}")
            moved['error'] += 1

    print(f"移动完成!")
    if mode in ("detected", "confirmed"):
        print(f"  真阳性 → Confirmed/: {moved['confirmed']}")
        print(f"  识别错误 → BadDetection/: {moved['bad_detection']}")
        print(f"  误报 → FalsePositive/: {moved['false_positive']}")
    else:
        print(f"  漏检 → Missed/: {moved['missed']}")
    if moved['error']:
        print(f"  错误: {moved['error']}")


def main():
    parser = argparse.ArgumentParser(description="人工审批检测结果")
    parser.add_argument('--input', type=str, required=True, help='推理结果目录 (run_XXXXXXXX_XXXXXX)')
    parser.add_argument('--mode', choices=['detected', 'empty', 'confirmed'], default='detected',
                        help='审查模式: detected=审查检出图片, empty=审查未检出图片, confirmed=重新审查已确认图片')
    parser.add_argument('--model', type=str, default=None,
                        help='只审查指定模型 (如 DroneDetection, AntiUAV, model3)')
    parser.add_argument('--competition', type=str, default=None,
                        help='只审查指定赛区 (如 "Final Tournament")')
    parser.add_argument('--apply', action='store_true',
                        help='根据审查结果批量移动图片 (审查完成后执行)')
    parser.add_argument('--stats', action='store_true',
                        help='查看当前审查进度')
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_absolute():
        input_dir = BASE_DIR / args.input

    if not input_dir.exists():
        print(f"错误: 目录不存在 {input_dir}")
        return

    # 检查是否已整理
    if not check_organized(input_dir):
        print("错误: 该目录尚未整理 (未找到 Detected/Empty 子目录)")
        print()
        print("请先运行 organize_results.py 进行整理:")
        print(f"  python scripts/organize_results.py --input {args.input}")
        return

    # 进度文件
    timestamp = datetime.now().strftime("%Y%m%d")
    progress_file = input_dir / f"review_progress_{args.mode}_{timestamp}.json"

    # 查找已有进度文件 (同模式)
    existing = sorted(input_dir.glob(f"review_progress_{args.mode}_*.json"), reverse=True)
    if existing:
        progress_file = existing[0]
        print(f"使用已有进度: {progress_file.name}")

    if args.stats:
        progress = load_progress(progress_file)
        print_stats(progress)
        return

    if args.apply:
        apply_results(input_dir, args.mode, progress_file)
        return

    # 收集图片
    images = collect_images(input_dir, args.mode, args.model, args.competition)
    if not images:
        print("未找到符合条件的图片。")
        return

    # 统计
    models = set(i['model'] for i in images)
    competitions = set(i['competition'] for i in images)
    mode_labels = {
        'detected': '检出图片 (Detected)',
        'empty': '未检出图片 (Empty)',
        'confirmed': '重新审查已确认图片 (Confirmed)',
    }
    print(f"审查模式: {mode_labels[args.mode]}")
    print(f"图片总数: {len(images)}")
    print(f"模型: {', '.join(sorted(models))}")
    print(f"赛区: {', '.join(sorted(competitions))}")
    print()

    run_review(images, input_dir, args.mode, progress_file)

    print()
    print("审查完毕后，运行以下命令批量移动图片:")
    print(f"  python scripts/manual_review.py --input {args.input} --mode {args.mode} --apply")


if __name__ == "__main__":
    main()
