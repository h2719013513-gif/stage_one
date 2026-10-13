# -*- coding: utf-8 -*-
"""实验 5：用 PyTorch 完整实现一个线性回归模型

流程：生成带噪声的数据 -> 定义模型 -> 前向传播 -> 算 Loss -> loss.backward() -> 优化器更新参数。
全程使用 PyTorch 的自动求导和优化器，不使用任何现成的回归拟合函数。

运行方式：
    python linear_regression.py
    python linear_regression.py --lr 0.1 --epochs 200
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
import torch
import torch.nn as nn

matplotlib.use("Agg")  # 无界面环境也能出图
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output"
FIG_DIR = OUT_DIR / "figures"

# 真实规律：y = 2.5x + 1.2，模型要从带噪声的数据里把它学出来
TRUE_W = 2.5
TRUE_B = 1.2
N_SAMPLES = 200
NOISE_STD = 0.8
DATA_SEED = 42

LINE = "=" * 74


def setup_plot_style() -> None:
    """设置中文字体，避免图中中文显示成方块。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150


# --------------------------------------------------------------------------
# 1. 生成带噪声的线性回归数据
# --------------------------------------------------------------------------
def generate_data(
    n_samples: int = N_SAMPLES,
    noise_std: float = NOISE_STD,
    seed: int = DATA_SEED,
) -> tuple[torch.Tensor, torch.Tensor]:
    """y = TRUE_W * x + TRUE_B + 噪声，x 在 [-3, 3] 上均匀分布。"""
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand(n_samples, 1, generator=generator) * 6 - 3
    noise = torch.randn(n_samples, 1, generator=generator) * noise_std
    y = TRUE_W * x + TRUE_B + noise
    return x, y


# --------------------------------------------------------------------------
# 2. 模型
# --------------------------------------------------------------------------
def build_model() -> nn.Linear:
    """一个最简单的线性模型：y = x @ w + b（输入 1 维、输出 1 维）。"""
    return nn.Linear(1, 1)


# --------------------------------------------------------------------------
# 3. 训练循环（核心）
# --------------------------------------------------------------------------
def train(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    lr: float = 0.1,
    epochs: int = 200,
    log_every: int = 0,
    verbose: bool = True,
) -> dict:
    """标准训练循环：前向 -> 算损失 -> 反向传播 -> 更新参数。"""
    criterion = nn.MSELoss()                                   # 损失函数：均方误差
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)     # 优化器：随机梯度下降

    history: dict = {"epoch": [], "loss": [], "diverged_at": None}

    for epoch in range(1, epochs + 1):
        # (1) 前向传播：用当前参数算预测值
        y_pred = model(x)
        # (2) 计算损失
        loss = criterion(y_pred, y)
        # (3) 清空上一轮的梯度（PyTorch 的梯度是累加的，不清零会越加越大）
        optimizer.zero_grad()
        # (4) 反向传播：自动求出每个参数的梯度
        loss.backward()
        # (5) 优化器根据梯度更新参数
        optimizer.step()

        value = float(loss.item())
        history["epoch"].append(epoch)
        history["loss"].append(value)

        # 学习率太大时损失会爆炸成 inf/nan，及时停下来
        if not math.isfinite(value) or value > 1e8:
            history["diverged_at"] = epoch
            if verbose:
                print(f"  第 {epoch} 轮损失变成 {value}，训练发散，提前停止。")
            break

        if verbose and (epoch == 1 or (log_every and epoch % log_every == 0)):
            w_now = model.weight.item()
            b_now = model.bias.item()
            print(f"  epoch {epoch:>4}   loss = {value:>12.6f}   w = {w_now:>8.4f}   b = {b_now:>8.4f}")

    history["lr"] = lr
    history["epochs_run"] = len(history["epoch"])
    return history


# --------------------------------------------------------------------------
# 4. 保存结果
# --------------------------------------------------------------------------
def save_training_log(history: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "loss"])
        writer.writerows(zip(history["epoch"], history["loss"]))
    return path


def plot_loss_curve(history: dict, path: Path) -> Path:
    """Loss 随训练轮次变化的曲线（纵轴取对数，方便看清下降过程）。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history["loss"], color="#4C72B0", linewidth=2)
    ax.set_xlabel("训练轮次 epoch")
    ax.set_ylabel("损失 Loss（对数坐标）")
    ax.set_yscale("log")
    ax.set_title(f"实验 5  训练损失曲线（学习率 = {history['lr']}，共 {history['epochs_run']} 轮）")
    ax.grid(alpha=0.3, which="both")

    first, last = history["loss"][0], history["loss"][-1]
    ax.annotate(
        f"起始 Loss = {first:.4f}",
        xy=(history["epoch"][0], first),
        xytext=(0.15, 0.75),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "#888"},
    )
    ax.annotate(
        f"最终 Loss = {last:.6f}",
        xy=(history["epoch"][-1], last),
        xytext=(0.55, 0.25),
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "#888"},
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_prediction(model: nn.Module, x: torch.Tensor, y: torch.Tensor, path: Path) -> Path:
    """真实数据散点 + 真实直线 + 模型学到的直线。"""
    with torch.no_grad():
        x_line = torch.linspace(x.min().item(), x.max().item(), 100).reshape(-1, 1)
        y_model_line = model(x_line)
        y_true_line = TRUE_W * x_line + TRUE_B

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(x.numpy(), y.numpy(), s=14, color="#4C72B0", alpha=0.55, label="带噪声的训练数据")
    ax.plot(
        x_line.numpy(),
        y_true_line.numpy(),
        "--",
        color="#55A868",
        linewidth=2,
        label=f"真实规律 y = {TRUE_W}x + {TRUE_B}",
    )
    ax.plot(
        x_line.numpy(),
        y_model_line.numpy(),
        "-",
        color="#C44E52",
        linewidth=2.5,
        label=f"模型预测 y = {model.weight.item():.3f}x + {model.bias.item():.3f}",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("实验 5  预测结果与真实数据对比")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# 5. 主流程
# --------------------------------------------------------------------------
def run(lr: float = 0.1, epochs: int = 200, log_every: int = 20) -> dict:
    print(LINE)
    print("实验 5：用 PyTorch 实现线性回归（自动求导 + 优化器更新参数）")
    print(LINE)

    print("\n[1/5] 生成带噪声的数据")
    x, y = generate_data()
    print(f"  样本数：{x.shape[0]}，x 范围：[{x.min().item():.2f}, {x.max().item():.2f}]")
    print(f"  真实规律：y = {TRUE_W}x + {TRUE_B}，噪声标准差 = {NOISE_STD}")

    print("\n[2/5] 构建模型")
    torch.manual_seed(0)                 # 固定初始参数，结果可复现
    model = build_model()
    print(f"  模型：{model}")
    print(f"  初始参数：w = {model.weight.item():.4f}，b = {model.bias.item():.4f}")

    print(f"\n[3/5] 开始训练（学习率 = {lr}，轮次 = {epochs}）")
    history = train(model, x, y, lr=lr, epochs=epochs, log_every=log_every)

    print("\n[4/5] 训练结果")
    print(f"  真实参数：w = {TRUE_W}，b = {TRUE_B}")
    print(f"  学到的参数：w = {model.weight.item():.4f}，b = {model.bias.item():.4f}")
    print(f"  误差：|Δw| = {abs(model.weight.item() - TRUE_W):.4f}，|Δb| = {abs(model.bias.item() - TRUE_B):.4f}")
    print(f"  起始 Loss = {history['loss'][0]:.6f}  最终 Loss = {history['loss'][-1]:.6f}")

    print("\n[5/5] 保存结果")
    figures = [
        plot_loss_curve(history, FIG_DIR / "实验5_Loss曲线.png"),
        plot_prediction(model, x, y, FIG_DIR / "实验5_预测结果对比.png"),
    ]
    saved_log = save_training_log(history, OUT_DIR / "实验5_训练日志.csv")
    for path in figures:
        print(f"  已保存图片：{path}")
    print(f"  训练日志：{saved_log}")

    print(LINE)
    print("实验 5 运行完成。")
    print(LINE)
    return {"model": model, "history": history, "x": x, "y": y}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch 线性回归训练")
    parser.add_argument("--lr", type=float, default=0.1, help="学习率")
    parser.add_argument("--epochs", type=int, default=200, help="训练轮次")
    parser.add_argument("--log-every", type=int, default=20, help="每隔多少轮打印一次 Loss")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_plot_style()
    run(lr=args.lr, epochs=args.epochs, log_every=args.log_every)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
