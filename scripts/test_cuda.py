#!/usr/bin/env python
"""
测试 CUDA 环境和 GPU 加速
"""

import sys

def main():
    print("=" * 50)
    print("CUDA 环境测试")
    print("=" * 50)

    # 测试 PyTorch
    try:
        import torch
        print(f"\n[PyTorch]")
        print(f"  版本: {torch.__version__}")
        print(f"  CUDA 可用: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"  CUDA 版本: {torch.version.cuda}")
            print(f"  cuDNN 版本: {torch.backends.cudnn.version()}")
            print(f"  GPU 数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name}")
                print(f"    显存: {props.total_memory / 1024**3:.1f} GB")
                print(f"    计算能力: {props.major}.{props.minor}")

            # 简单运算测试
            print(f"\n[GPU 运算测试]")
            x = torch.randn(1000, 1000, device='cuda')
            y = torch.randn(1000, 1000, device='cuda')

            # 预热
            for _ in range(10):
                z = torch.matmul(x, y)
            torch.cuda.synchronize()

            # 计时
            import time
            start = time.time()
            for _ in range(100):
                z = torch.matmul(x, y)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            print(f"  矩阵乘法 (1000x1000) x100: {elapsed*1000:.2f} ms")
            print(f"  GPU 运算正常!")
        else:
            print("  [警告] CUDA 不可用!")

    except ImportError:
        print("[错误] PyTorch 未安装")
        return 1

    # 测试 torchvision
    try:
        import torchvision
        print(f"\n[torchvision]")
        print(f"  版本: {torchvision.__version__}")

        # 测试 NMS 操作
        if torch.cuda.is_available():
            boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=torch.float32, device='cuda')
            scores = torch.tensor([0.9, 0.8], device='cuda')
            result = torchvision.ops.nms(boxes, scores, 0.5)
            print(f"  NMS 测试: 通过")
    except ImportError:
        print("[错误] torchvision 未安装")
    except Exception as e:
        print(f"  [错误] torchvision NMS 测试失败: {e}")
        return 1

    # 测试 ultralytics
    try:
        import ultralytics
        print(f"\n[ultralytics]")
        print(f"  版本: {ultralytics.__version__}")
    except ImportError:
        print("\n[ultralytics]")
        print("  未安装 (pip install ultralytics)")

    print("\n" + "=" * 50)
    print("测试完成!")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())
