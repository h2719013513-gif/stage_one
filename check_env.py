# -*- coding: utf-8 -*-
"""实验 1：深度学习开发环境检测程序

运行方式：
    python check_env.py

输出内容：
    1. Python 版本
    2. PyTorch 版本
    3. 是否支持 CUDA（即能否使用 NVIDIA 显卡做加速）
    4. 一个 3x3 的随机张量
    5. CPU 与 GPU 的区别（现场验收的问答要点）
"""

from __future__ import annotations

import platform
import sys

LINE = "=" * 62


def banner(title: str) -> None:
    print()
    print(LINE)
    print(title)
    print(LINE)


def check_python() -> None:
    banner("[1] Python 版本")
    print(f"Python 版本   : {platform.python_version()}")
    print(f"解释器路径    : {sys.executable}")
    print(f"操作系统      : {platform.system()} {platform.release()} ({platform.machine()})")
    ok = sys.version_info >= (3, 9)
    print(f"是否满足 >=3.9: {'是' if ok else '否'}")


def check_torch() -> "object | None":
    """检测 PyTorch 与 CUDA，返回 torch 模块（未安装时返回 None）。"""
    banner("[2] PyTorch 版本    [3] 是否支持 CUDA")
    try:
        import torch  # noqa: PLC0415  （放在函数内，便于未安装时给出友好提示）
    except ImportError as error:
        print(f"未检测到 PyTorch：{error}")
        print()
        print("请先安装 PyTorch，然后重新运行本程序：")
        print("    CPU 版本 : pip install torch --index-url https://download.pytorch.org/whl/cpu")
        print("    GPU 版本 : pip install torch --index-url https://download.pytorch.org/whl/cu124")
        print("    国内镜像 : pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple")
        return None

    print(f"PyTorch 版本  : {torch.__version__}")
    print(f"编译所用 CUDA : {torch.version.cuda or '未启用（CPU 版本）'}")

    cuda_available = torch.cuda.is_available()
    print(f"是否支持 CUDA : {'是' if cuda_available else '否'}")

    if cuda_available:
        print(f"显卡数量      : {torch.cuda.device_count()}")
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            memory_gb = properties.total_memory / 1024**3
            print(f"  显卡 {index}       : {properties.name}（显存 {memory_gb:.1f} GB）")
        if torch.backends.cudnn.is_available():
            print(f"cuDNN 版本    : {torch.backends.cudnn.version()}")
        print(f"当前计算设备  : {torch.cuda.get_device_name(0)}")
    else:
        print("当前计算设备  : CPU（未检测到可用的 NVIDIA 显卡，或安装的是 CPU 版本 PyTorch）")

    return torch


def show_tensor(torch: object) -> None:
    banner("[4] 3x3 随机张量")
    torch.set_printoptions(precision=4, linewidth=120)
    tensor = torch.rand(3, 3)
    print(f"张量类型 : {type(tensor).__name__}    形状 : {tuple(tensor.shape)}    数据类型 : {tensor.dtype}")
    print(f"所在设备 : {tensor.device}")
    print()
    print(tensor)
    print()
    print(f"张量元素个数 : {tensor.numel()}")
    print(f"最大值       : {tensor.max().item():.4f}")
    print(f"最小值       : {tensor.min().item():.4f}")
    print(f"平均值       : {tensor.mean().item():.4f}")

    if torch.cuda.is_available():
        print()
        print("把张量搬到显卡上做一次加法，验证 GPU 真的可以参与计算：")
        gpu_tensor = tensor.to("cuda")
        print(f"   张量所在设备 : {gpu_tensor.device}")
        print(f"   (a + a)[0]  : {gpu_tensor.add(gpu_tensor)[0].tolist()}")


def explain_cpu_gpu() -> None:
    banner("[5] CPU 与 GPU 的区别（验收问答要点）")
    rows = [
        ("设计目标", "通用串行计算，逻辑判断强", "大规模并行数值计算"),
        ("核心数量", "通常 4~16 个高性能核心", "上千个简单计算核心"),
        ("擅长任务", "复杂分支、操作系统调度", "矩阵乘、卷积等可并行的运算"),
        ("深度学习", "适合小模型、调试、推理演示", "适合大模型训练，速度可快数十倍"),
        ("显存/内存", "使用内存条，容量大、带宽低", "使用显存，容量小、带宽高"),
        ("本项目", "torch.rand 默认在 CPU 上创建", "tensor.to('cuda') 后由 GPU 计算"),
    ]
    print(f"{'对比项':<10}{'CPU':<30}{'GPU'}")
    print("-" * 84)
    for item in rows:
        print(f"{item[0]:<10}{item[1]:<30}{item[2]}")
    print()
    print("一句话总结：CPU 像几个博士生，什么都会做但人数少；")
    print("            GPU 像几千个小学生，只会做简单的加减乘除，但人海战术适合深度学习的矩阵运算。")


def main() -> int:
    print(LINE)
    print("深度学习开发环境检测程序（实验 1）")
    print(LINE)

    check_python()
    torch = check_torch()
    if torch is not None:
        show_tensor(torch)
    explain_cpu_gpu()

    print()
    if torch is None:
        print("检测结论：环境不完整，PyTorch 尚未安装。")
        return 1
    print("检测结论：环境检测完成，可以开始深度学习实验。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
