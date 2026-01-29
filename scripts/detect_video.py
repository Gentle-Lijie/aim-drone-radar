#!/usr/bin/env python
"""
视频无人机检测脚本
支持 MP4 文件和摄像头实时检测
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

# 抑制警告
os.environ["YOLO_VERBOSE"] = "False"
warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import torch


def load_model(weights: str):
    """加载 YOLOv5 模型"""
    print(f"加载模型: {weights}")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights, force_reload=False, verbose=False)
    if torch.cuda.is_available():
        model = model.cuda()
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("使用 CPU")
    model.eval()
    return model


def draw_detections(frame, results, conf_threshold: float = 0.5):
    """在帧上绘制检测结果"""
    detections = results.xyxy[0].cpu().numpy()
    count = 0

    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if conf < conf_threshold:
            continue

        count += 1
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # 绘制边界框
        color = (0, 255, 0)  # 绿色
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 绘制标签背景
        label = f"Drone {conf:.2f}"
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)

        # 绘制标签文字
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return frame, count


def process_video(model, source, output: str = None, conf: float = 0.5,
                  show: bool = True, save: bool = False):
    """处理视频流"""
    # 打开视频源
    if source == "0" or source == 0:
        cap = cv2.VideoCapture(0)
        source_name = "摄像头"
    else:
        cap = cv2.VideoCapture(source)
        source_name = Path(source).name

    if not cap.isOpened():
        print(f"[错误] 无法打开视频源: {source}")
        return

    # 获取视频属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n视频源: {source_name}")
    print(f"分辨率: {width}x{height}")
    print(f"帧率: {fps:.1f} FPS")
    if total_frames > 0:
        print(f"总帧数: {total_frames}")
    print(f"置信度阈值: {conf}")
    print("\n按 'q' 退出, 's' 截图, 'p' 暂停\n")

    # 设置视频输出
    writer = None
    if save and output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"输出文件: {output_path}")

    # 处理循环
    frame_count = 0
    start_time = time.time()
    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # 推理
            results = model(frame)

            # 绘制检测结果
            frame, det_count = draw_detections(frame, results, conf)

            # 计算 FPS
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0

            # 绘制状态信息
            status = f"FPS: {current_fps:.1f} | Drones: {det_count}"
            if total_frames > 0:
                progress = frame_count / total_frames * 100
                status += f" | Progress: {progress:.1f}%"

            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # 保存帧
            if writer:
                writer.write(frame)

        # 显示
        if show:
            cv2.imshow("Drone Detection - AIM Radar", frame)

        # 键盘控制
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n用户退出")
            break
        elif key == ord('s'):
            screenshot = f"screenshot_{frame_count}.jpg"
            cv2.imwrite(screenshot, frame)
            print(f"截图保存: {screenshot}")
        elif key == ord('p'):
            paused = not paused
            print("暂停" if paused else "继续")

    # 清理
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # 统计
    total_time = time.time() - start_time
    print(f"\n处理完成!")
    print(f"总帧数: {frame_count}")
    print(f"总时间: {total_time:.2f}s")
    print(f"平均FPS: {frame_count / total_time:.1f}")


def main():
    parser = argparse.ArgumentParser(description="视频无人机检测")
    parser.add_argument("--source", type=str, default="0",
                        help="视频源: 摄像头ID (0) 或视频文件路径")
    parser.add_argument("--weights", type=str, default="models/DroneDetection/best.pt",
                        help="模型权重路径")
    parser.add_argument("--conf", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--output", type=str, default=None, help="输出视频路径")
    parser.add_argument("--no-show", action="store_true", help="不显示窗口")
    parser.add_argument("--save", action="store_true", help="保存检测结果视频")

    args = parser.parse_args()

    print("=" * 50)
    print("AIM Radar - 无人机视频检测")
    print("=" * 50)

    # 检查权重文件
    if not Path(args.weights).exists():
        print(f"[错误] 模型文件不存在: {args.weights}")
        sys.exit(1)

    # 加载模型
    model = load_model(args.weights)

    # 设置输出路径
    output = args.output
    if args.save and not output:
        if args.source == "0" or args.source == 0:
            output = "output/camera_detection.mp4"
        else:
            output = f"output/{Path(args.source).stem}_detection.mp4"

    # 处理视频
    process_video(
        model=model,
        source=args.source,
        output=output,
        conf=args.conf,
        show=not args.no_show,
        save=args.save,
    )


if __name__ == "__main__":
    main()
