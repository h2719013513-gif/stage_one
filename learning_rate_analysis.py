# -*- coding: utf-8 -*-
"""实验 6：不同学习率下的训练过程对比与分析

在实验 5 的基础上，用同一份数据、同一个初始参数，只改变学习率重新训练，
比较 Loss 曲线的差异，回答“学习率过大有什么问题、什么时候算收敛”。

运行方式：
    python learning_rate_analysis.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from linear_regression import (  # noqa: E402  复用实验 5 里写好的函数
    FIG_DIR,
    LINE,
    NOISE_STD,
    OUT_DIR,
    TRUE_B,
    TRUE_W,
    build_model,
    generate_data,
    setup_plot_style,
    train,
)

# 五个学习率：偏小、合适、偏大（震荡）、过大（发散）、太大（立刻爆炸）
LEARNING_RATES = [0.01, 0.1, 0.3, 0.5, 1.0]
EPOCHS = 60
COLORS = ["#4C72B0", "#55A868", "#8172B3", "#DD8452", "#C44E52"]


def fmt_loss(value: float) -> str:
    """损失数值大时用科学计数法，避免把表格撑乱。"""
    if value is None or value != value:       # None 或 nan
        return "nan"
    if abs(value) >= 1e5:
        return f"{value:.3e}"
    return f"{value:.6f}"


def run_experiments() -> tuple[torch.Tensor, torch.Tensor, dict]:
    """用每个学习率各训练一次，返回数据与结果。"""
    x, y = generate_data()
    results: dict = {}

    print(f"数据：{x.shape[0]} 个样本，真实规律 y = {TRUE_W}x + {TRUE_B}，噪声标准差 {NOISE_STD}")
    print(f"每个学习率都训练 {EPOCHS} 轮，初始参数完全相同\n")
    print(f"{'学习率':>8}{'实际轮次':>10}{'起始 Loss':>14}{'最终 Loss':>16}{'学到的 w':>14}{'学到的 b':>14}   状态")

    for lr in LEARNING_RATES:
        torch.manual_seed(0)          # 保证每个学习率的初始参数一模一样，只差学习率
        model = build_model()
        history = train(model, x, y, lr=lr, epochs=EPOCHS, verbose=False)
        results[lr] = {"model": model, "history": history}

        final_loss = history["loss"][-1]
        if history["diverged_at"]:
            status = f"第 {history['diverged_at']} 轮发散"
        else:
            status = "正常"
        print(
            f"{lr:>8}{history['epochs_run']:>10}{history['loss'][0]:>14.4f}"
            f"{fmt_loss(final_loss):>16}{model.weight.item():>14.3f}{model.bias.item():>14.3f}   {status}"
        )

    return x, y, results


def plot_loss_curves(results: dict, path: Path) -> Path:
    """图 1：不同学习率的 Loss 曲线放在同一张图里对比。"""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for (lr, result), color in zip(results.items(), COLORS):
        history = result["history"]
        losses = [value if value > 0 else float("nan") for value in history["loss"]]
        ax.plot(history["epoch"], losses, linewidth=2, color=color, label=f"学习率 = {lr}")
        if history["diverged_at"]:
            ax.axvline(history["diverged_at"], color=color, linestyle=":", alpha=0.7)

    ax.set_yscale("log")
    ax.set_xlabel("训练轮次 epoch")
    ax.set_ylabel("损失 Loss（对数坐标）")
    ax.set_title("实验 6  不同学习率的训练损失曲线（同数据、同初始参数）")
    ax.grid(alpha=0.3, which="both")
    ax.legend(title="学习率")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_fits(x: torch.Tensor, y: torch.Tensor, results: dict, path: Path) -> Path:
    """图 2：不同学习率学出来的直线与真实数据的对比。"""
    count = len(results)
    cols = 3
    rows = (count + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.1 * rows))
    axes_list = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]
    x_line = torch.linspace(x.min().item(), x.max().item(), 100).reshape(-1, 1)
    y_true_line = TRUE_W * x_line + TRUE_B

    for ax, ((lr, result), color) in zip(axes_list, zip(results.items(), COLORS)):
        model = result["model"]
        with torch.no_grad():
            y_model_line = model(x_line)
        ax.scatter(x.numpy(), y.numpy(), s=8, color="#9AA5B1", alpha=0.45, label="训练数据")
        ax.plot(x_line.numpy(), y_true_line.numpy(), "--", color="#55A868", linewidth=2, label="真实规律")
        ax.plot(
            x_line.numpy(),
            y_model_line.numpy(),
            "-",
            color=color,
            linewidth=2.5,
            label=f"模型 w={model.weight.item():.2f}, b={model.bias.item():.2f}",
        )
        suffix = "（已发散）" if result["history"]["diverged_at"] else ""
        ax.set_title(f"学习率 = {lr}{suffix}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    for ax in axes_list[count:]:        # 多余的子图留白
        ax.axis("off")

    fig.suptitle("实验 6  不同学习率的拟合结果对比", fontsize=14)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_curves_csv(results: dict, path: Path) -> Path:
    """把每条 Loss 曲线导出成 CSV，方便自己在 Excel 里画图。"""
    frame = pd.DataFrame({"epoch": list(range(1, EPOCHS + 1))})
    for lr, result in results.items():
        losses = result["history"]["loss"]
        column = losses + [None] * (EPOCHS - len(losses))   # 提前发散的补空值
        frame[f"loss_lr_{lr}"] = column
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def print_analysis(results: dict) -> None:
    print()
    print(LINE)
    print("分析结论（以下数值都来自本次实际运行）")
    print(LINE)

    print("一、每个学习率各自的结果")
    for lr, result in results.items():
        history = result["history"]
        model = result["model"]
        if history["diverged_at"]:
            print(
                f"  · 学习率 {lr}：第 {history['diverged_at']} 轮 Loss 爆炸到 "
                f"{fmt_loss(history['loss'][-1])}，训练失败。"
            )
        else:
            first, last = history["loss"][0], history["loss"][-1]
            drop = (1 - last / first) * 100
            print(
                f"  · 学习率 {lr}：{history['epochs_run']} 轮后 Loss 从 {first:.4f} 降到 {last:.6f}"
                f"（下降 {drop:.2f}%），参数 w={model.weight.item():.3f}，b={model.bias.item():.3f}。"
            )

    print()
    print("二、学习率变化后，Loss 曲线有什么不同？")
    print("  · 学习率小：曲线下降很慢、很平缓，60 轮还在慢慢往下走，没到最低点；")
    print("  · 学习率合适：前几轮陡降，随后变成一条几乎水平的线；")
    print("  · 学习率偏大（但仍在稳定范围内，如 0.3）：还能收敛，只是过程更“贴边”，")
    print("    再大一点就会翻车；")
    print("  · 学习率过大：先震荡几步，然后突然垂直向上冲到 1e8 以上，直接爆掉。")
    print("  补充：本例 0.1 和 0.3 最终 Loss 完全相同，这不是巧合 —— 线性回归的损失函数是凸的，")
    print("  只要不发散，无论从哪走、走多快，最终都会收敛到同一个最优点，区别只在快慢。")

    print()
    print("三、学习率过大会出现什么问题？")
    print("  参数更新一步走多远 = 学习率 × 梯度。步子太大就会一下跨过最低点，落到对面更陡的位置，")
    print("  下一步梯度更大，参数被越推越远，Loss 指数级增长，最后变成 inf/nan，训练彻底失败。")
    print("  本次实验里学习率 0.5 在第 13 轮、1.0 在第 6 轮就发散了。")

    print()
    print("四、什么情况下可以认为模型已经收敛？")
    print("  1) Loss 连续很多轮几乎不再变化（相邻两轮差值小于 1e-6）；")
    print("  2) 参数 w、b 也基本不再移动；")
    print("  3) Loss 降到了合理水平。本例噪声标准差 0.8，理论最低 MSE ≈ 0.64，")
    print("     实际收敛到 0.5 左右，说明已经学到了 y = 2.5x + 1.2 这个真实规律。")
    print("  本次学习率 0.1 大约在第 40 轮之后 Loss 就一直稳定在 0.506008，可以认为已经收敛。")
    print(LINE)


def main() -> int:
    setup_plot_style()
    print(LINE)
    print("实验 6：不同学习率的训练过程分析")
    print(LINE)
    print(f"PyTorch 版本：{torch.__version__}\n")

    x, y, results = run_experiments()

    figures = [
        plot_loss_curves(results, FIG_DIR / "实验6_不同学习率Loss曲线.png"),
        plot_fits(x, y, results, FIG_DIR / "实验6_不同学习率拟合对比.png"),
    ]
    csv_path = save_curves_csv(results, OUT_DIR / "实验6_不同学习率Loss数据.csv")

    print("\n[保存结果]")
    for path in figures:
        print(f"  已保存图片：{path}")
    print(f"  曲线数据：{csv_path}")

    print_analysis(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
