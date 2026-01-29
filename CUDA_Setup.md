# 在本仓库中启用 CUDA GPU 加速（可复刻步骤）

本文档总结并复现了本仓库中用于激活 CUDA / GPU 加速的实践与命令，覆盖 Windows 上的预编译 wheel 安装、源码编译（仓库内脚本）和验证步骤。

## 适用场景
- 开发/实验机器：Windows（本仓库当前环境）或 Linux（部分说明为可选）。
- 目标 GPU：NVIDIA（如 RTX 5060）；确保已安装对应版本 CUDA Toolkit 与 NVIDIA 驱动。

## 先决条件
- 已安装 NVIDIA 驱动并能运行 nvidia-smi。
- 已安装相匹配的 CUDA Toolkit（例如 CUDA 12.1 或 13.1，取决于要安装的 PyTorch wheel 或是否源码编译）。
- 已安装 Visual Studio Build Tools（Windows源码编译时需要），并能调用 `vcvars64.bat`。
- 建议在虚拟环境中操作（venv / .venv）。

## 一、检查系统和驱动

在 PowerShell / cmd 中运行：

```powershell
# 查看 GPU 与驱动
nvidia-smi

# （可选）查看 CUDA 版本
nvcc --version
```

如果 `nvidia-smi` 未找到或驱动版本过旧，请先从 NVIDIA 官网安装驱动。

## 二、创建并激活虚拟环境（示例）

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # PowerShell
# 或 .\.venv\Scripts\activate (cmd)
python -m pip install --upgrade pip
```

本仓库常用的虚拟环境路径示例为 `D:\DEV\VLM\.venv`。

## 三、推荐：使用 PyTorch 官方预编译 wheel（优先）

优点：更简单、速度更快、避免源码编译复杂性。关键是选择与本机 CUDA Toolkit 兼容的 wheel。仓库中文档与依赖里给出了多处建议：

- 附件说明建议使用 PyTorch 2.5.1 + cu121（CUDA 12.1）的预编译包
- requirements.txt 注释也给出示例索引（cu124/cuda12.4）

示例安装（以 CUDA 12.1 为例）：

```powershell
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

替代：若需 CUDA 12.4/其他版本，请把 `cu121` 换成对应的 wheel tag（例如 `cu124`）。

注意：确保 `torch` / `torchvision` 版本与仓库 `pyproject.toml` / `requirements.txt` 中的依赖约束兼容。

## 四、源码编译（当没有合适 wheel 或需自定义时）

仓库在 `scripts/` 中提供了 Windows 下用于源码构建的批处理示例：

- `scripts\build_pytorch_cuda131.bat`：用于在 Windows 上用 CUDA 13.1 源码编译并安装 PyTorch（通过 `setup.py develop`）。
- `scripts\build_torchvision_cuda131.bat`：用于源码编译并安装 torchvision。

这两个脚本要点如下：
- 调用 Visual Studio vcvars（`vcvars64.bat`）准备编译环境
- 设置 `PATH` 指向虚拟环境脚本
- 设置 `CUDA_HOME`/`CUDA_PATH` 指向所安装的 CUDA Toolkit（脚本示例为 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1`）
- 设置 `TORCH_CUDA_ARCH_LIST`（影响编译时生成的 GPU 架构代码）
- 使用 Ninja（`USE_NINJA=1`，`CMAKE_GENERATOR=Ninja`）以加速构建

示例（参考脚本的关键步骤，需在管理员权限或合适 shell 下运行）：

```powershell
# 编辑并确认路径：
set "VENV=D:\DEV\VLM\.venv"
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set PATH=%VENV%\Scripts;%PATH%
set CUDA_HOME="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
set CUDA_PATH=%CUDA_HOME%
# 注意：TORCH_CUDA_ARCH_LIST 请根据你的 GPU 调整，例如 RTX 5060 -> 8.9
set TORCH_CUDA_ARCH_LIST=8.9
set USE_NINJA=1
set CMAKE_GENERATOR=Ninja

# 进入 pytorch 源码目录并安装（示例）
cd /d D:\pytorch-src
%VENV%\Scripts\python.exe setup.py develop
```

对 `torchvision` 的源码安装类似：

```powershell
cd /d D:\torchvision-src
%VENV%\Scripts\python.exe -m pip install -e . -v --no-build-isolation
```

重要提示：脚本 `build_pytorch_cuda131.bat` 中示例将 `TORCH_CUDA_ARCH_LIST` 设为 `12.0`（与 CUDA 13.1 对应），但通常最好把它设为目标 GPU 的 compute capability（例如 RTX 5060 为 8.9）。错误的 arch 列表可能导致编译无法覆盖目标 GPU 或浪费编译时间。

## 五、仓库依赖与特殊说明

- `pyproject.toml` / `requirements.txt` 中列出了一些可选/平台相关依赖：
  - `intel-extension-for-pytorch`：仅在 Linux 且用于 Intel CPU 优化，不适用于 Windows。 
  - `salesforce-lavis`（BLIP）在 Windows + Python 新版本可能不兼容，建议在 Linux 或 Python 3.12 及以下环境安装。
  - `bitsandbytes`：可用于量化与显存优化，需与 CUDA 版本兼容。

- 仓库注释中出现多个 CUDA-target 示例（cu121、cu124、源码用 13.1），复刻时请先确认本机 CUDA Toolkit 版本并选择匹配的安装路径/wheel。

## 六、启用混合精度/加速等（可选）

- 使用 AMP（自动混合精度）：PyTorch 原生支持 `torch.cuda.amp.autocast()` 与 `GradScaler`，并在 `accelerate` 中受支持。
- 使用 `bitsandbytes` 做量化/8-bit 优化以节省显存（需正确安装且支持 Windows/CUDA 版本）。

## 七、验证 GPU 与 CUDA 可用性（Python）

打开 Python 或在脚本中运行：

```python
import torch
print('torch version:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('cuda device count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('current device:', torch.cuda.current_device())
    print('device name:', torch.cuda.get_device_name(torch.cuda.current_device()))
```

若 `torch.cuda.is_available()` 返回 True，即表示 PyTorch 能够访问 CUDA。

## 八、常见问题与排查

- 问：安装后 `torch.cuda.is_available()` 为 False
  - 检查 `nvidia-smi` 是否能正确运行
  - 检查驱动是否与 CUDA Toolkit 匹配
  - 确保安装的 `torch` wheel 对应所安装的 CUDA（例如 cu121 vs cu124）
  - 在 Windows 上，若源码编译，请确保 Visual Studio Build Tools 与 vcvars 正确调用

- 问：源码编译速度慢或失败
  - 使用 `USE_NINJA=1` 并安装 Ninja
  - 调整 `TORCH_CUDA_ARCH_LIST` 以只生成必要架构
  - 增加虚拟内存或使用更高内存的机器

## 九、示例：从零到可以在本仓库运行的最短路径（Windows，已安装 CUDA 12.1）

```powershell
# 1. 创建并激活虚拟环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 2. 安装 PyTorch + torchvision 预编译包（CUDA12.1）
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. 安装项目依赖（不强制源码编译）
python -m pip install -r requirements.txt

# 4. 验证
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

# 5. 运行仓库示例
python scripts/run_baseline.py
```

## 十、附录：仓库中可参考的脚本/文件

- `scripts/build_pytorch_cuda131.bat`：Windows 源码编译 PyTorch 示例
- `scripts/build_torchvision_cuda131.bat`：Windows 源码编译 torchvision 示例
- `pyproject.toml` / `requirements.txt`：依赖与注释（包含 CUDA wheel 建议和平台条件）
- `README.md`：项目总体说明与 `configs/config.yaml` 中的 `device: cuda` 配置示例

---

完成：以上步骤汇总了本仓库中启用 CUDA / GPU 加速的实践与可复刻命令。若你希望我把本文件提交到 git（自动 commit），或根据你的本机 CUDA 版本把示例命令适配为具体版本（如 cu124 / cu121 / cu131），告诉我你的 CUDA 版本与是否需要我提交。