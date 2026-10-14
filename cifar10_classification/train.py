# -*- coding: utf-8 -*-
"""训练与验证：训练循环、保存最佳模型、画 Loss / 准确率曲线。

运行方式：
    python train.py                      # 用 config.py 里的默认超参数
    python train.py --epochs 40 --lr 5e-4

训练流程（每个 epoch）：
    训练集跑一遍 -> 反向传播更新参数 -> 在验证集上评估 -> 验证准确率更高就保存模型
注意：验证集只用来选模型和看效果，绝不参与参数更新；测试集在 evaluate.py 里才用。
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config
from dataset import build_dataloaders, setup_plot_style
from model import build_model, count_parameters

LINE = "=" * 76


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, criterion, optimizer, device) -> tuple[float, float]:
    """跑一个 epoch 的训练，返回（平均损失，准确率）。"""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        outputs = model(images)              # 前向传播
        loss = criterion(outputs, labels)    # 计算损失
        optimizer.zero_grad()                # 清空梯度
        loss.backward()                      # 反向传播
        optimizer.step()                     # 更新参数

        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    """在验证集/测试集上评估，返回（平均损失，准确率）。"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def save_history(history: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "seconds"])
        rows = zip(
            history["epoch"], history["lr"], history["train_loss"], history["train_acc"],
            history["val_loss"], history["val_acc"], history["seconds"],
        )
        writer.writerows(rows)
    return path


def plot_curves(history: dict, path: Path) -> Path:
    """左图画 Loss，右图画准确率，训练集与验证集放在一起对比。"""
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(history["epoch"], history["train_loss"], "-o", markersize=3, label="训练集 Loss")
    axes[0].plot(history["epoch"], history["val_loss"], "-s", markersize=3, label="验证集 Loss")
    axes[0].set_xlabel("训练轮次 epoch")
    axes[0].set_ylabel("损失 Loss")
    axes[0].set_title("训练 / 验证损失曲线")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(history["epoch"], [v * 100 for v in history["train_acc"]], "-o", markersize=3, label="训练集准确率")
    axes[1].plot(history["epoch"], [v * 100 for v in history["val_acc"]], "-s", markersize=3, label="验证集准确率")
    axes[1].set_xlabel("训练轮次 epoch")
    axes[1].set_ylabel("准确率 (%)")
    axes[1].set_title("训练 / 验证准确率曲线")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.suptitle("CIFAR-10 简单 CNN 训练过程", fontsize=14)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def train(
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    lr: float = config.LR,
    weight_decay: float = config.WEIGHT_DECAY,
) -> dict:
    config.ensure_dirs()
    torch.manual_seed(config.SEED)
    device = get_device()

    print(LINE)
    print("CIFAR-10 模型训练")
    print(LINE)
    print(f"设备：{device}" + (f"（{torch.cuda.get_device_name(0)}）" if device.type == "cuda" else ""))

    print("\n[1/4] 准备数据")
    loaders, info = build_dataloaders(batch_size=batch_size)
    print(f"  训练集 {info['train_size']} 张 / 验证集 {info['val_size']} 张 / 测试集 {info['test_size']} 张")

    print("\n[2/4] 构建模型")
    model = build_model().to(device)
    print(f"  SimpleCNN，可训练参数 {count_parameters(model):,} 个")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {key: [] for key in ("epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "seconds")}
    best_val_acc = 0.0
    start_time = time.time()

    print(f"\n[3/4] 开始训练（{epochs} 轮）")
    print(f"{'epoch':>6}{'lr':>10}{'train_loss':>12}{'train_acc':>11}{'val_loss':>11}{'val_acc':>10}{'用时(s)':>10}")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, loaders["val"], criterion, device)
        epoch_seconds = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history["epoch"].append(epoch)
        history["lr"].append(current_lr)
        history["train_loss"].append(round(train_loss, 6))
        history["train_acc"].append(round(train_acc, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["val_acc"].append(round(val_acc, 6))
        history["seconds"].append(round(epoch_seconds, 2))

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "config": {
                        "batch_size": batch_size, "lr": lr, "weight_decay": weight_decay,
                        "train_size": info["train_size"], "val_size": info["val_size"],
                    },
                },
                config.BEST_MODEL,
            )
            marker = "  <- 保存最佳模型"

        print(
            f"{epoch:>6}{current_lr:>10.5f}{train_loss:>12.4f}{train_acc*100:>10.2f}%"
            f"{val_loss:>11.4f}{val_acc*100:>9.2f}%{epoch_seconds:>10.1f}{marker}"
        )

    torch.save({"model_state": model.state_dict(), "epoch": epochs}, config.LAST_MODEL)
    total_seconds = time.time() - start_time

    print("\n[4/4] 保存结果")
    history_path = save_history(history, config.LOG_DIR / "training_history.csv")
    curve_path = plot_curves(history, config.FIG_DIR / "训练曲线_Loss与准确率.png")
    print(f"  最佳模型：{config.BEST_MODEL}（验证准确率 {best_val_acc*100:.2f}%）")
    print(f"  最后一轮模型：{config.LAST_MODEL}")
    print(f"  训练日志：{history_path}")
    print(f"  训练曲线：{curve_path}")
    print(f"  总用时：{total_seconds/60:.1f} 分钟")

    print(LINE)
    print(f"训练完成。最佳验证准确率 = {best_val_acc*100:.2f}%")
    print("下一步：python evaluate.py 在官方测试集上评估")
    print(LINE)
    return {"model": model, "history": history, "best_val_acc": best_val_acc}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CIFAR-10 简单 CNN 训练")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LR)
    parser.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
