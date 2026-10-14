# -*- coding: utf-8 -*-
"""单张图片预测。

三种用法：
    python predict.py --index 0                 # 用测试集里第 0 张（带真实标签，可对照）
    python predict.py --random                  # 随机挑一张测试集图片
    python predict.py --image D:\cat.png        # 预测你自己的图片（会自动缩放成 32x32）

输出：Top-3 类别及概率，并保存一张带预测结果的图片。
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib
import torch
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config
from dataset import CIFARDataset, build_transform, denormalize, ensure_raw_data, load_official_test, setup_plot_style
from evaluate import load_trained_model
from train import get_device

LINE = "=" * 76


def load_test_dataset() -> tuple[CIFARDataset, object]:
    """返回测试集 Dataset 以及它的图像/标签数组。"""
    ensure_raw_data(verbose=False)
    images, labels = load_official_test()
    dataset = CIFARDataset(images, labels, None, build_transform(False))
    return dataset, images


def predict_tensor(model, tensor: torch.Tensor, device, topk: int = 3) -> list[tuple[int, float]]:
    """对单个已归一化的张量做预测，返回 [(类别下标, 概率), ...]。"""
    model.eval()
    with torch.no_grad():
        logits = model(tensor.unsqueeze(0).to(device))
        probabilities = torch.softmax(logits, dim=1)[0]
    values, indices = probabilities.topk(topk)
    return [(int(index), float(value)) for value, index in zip(values, indices)]


def save_single_figure(tensor: torch.Tensor, title: str, path: Path, subtitle: str = "") -> Path:
    """把这一张图和预测结果画出来存盘。"""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(4.5, 5))
    ax.imshow(denormalize(tensor).permute(1, 2, 0).numpy())
    ax.axis("off")
    ax.set_title(title, fontsize=12)
    if subtitle:
        ax.text(0.5, -0.06, subtitle, ha="center", va="top", transform=ax.transAxes, fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CIFAR-10 单张图片预测")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--index", type=int, help="测试集中的第几张（从 0 开始）")
    group.add_argument("--random", action="store_true", help="从测试集随机抽一张")
    group.add_argument("--image", type=str, help="自己的图片路径")
    parser.add_argument("--topk", type=int, default=3, help="显示前几个候选类别")
    args = parser.parse_args(argv)

    config.ensure_dirs()
    device = get_device()
    model, _ = load_trained_model(device=device)

    print(LINE)
    print("CIFAR-10 单张图片预测")
    print(LINE)

    true_label = None
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"找不到图片：{image_path}")
            return 1
        image = Image.open(image_path).convert("RGB").resize((32, 32))
        tensor = build_transform(False)(image)
        source = f"自定义图片：{image_path}"
    else:
        dataset, _ = load_test_dataset()
        index = random.randrange(len(dataset)) if args.random else (args.index or 0)
        if not 0 <= index < len(dataset):
            print(f"下标超出范围：{index}（测试集共 {len(dataset)} 张）")
            return 1
        tensor, true_label = dataset[index]
        source = f"测试集第 {index} 张"

    predictions = predict_tensor(model, tensor, device, topk=args.topk)

    print(f"图片来源：{source}")
    if true_label is not None:
        print(f"真实标签：{config.CLASS_NAMES[true_label]}（{config.CLASS_NAMES_EN[true_label]}）")
    print(f"\nTop-{args.topk} 预测：")
    for rank, (class_index, probability) in enumerate(predictions, start=1):
        print(f"  {rank}. {config.CLASS_NAMES[class_index]:<4}{config.CLASS_NAMES_EN[class_index]:<12}{probability*100:>6.2f}%")

    predicted = predictions[0][0]
    if true_label is not None:
        verdict = "预测正确" if predicted == true_label else "预测错误"
        print(f"\n结论：{verdict}（预测 {config.CLASS_NAMES[predicted]}）")

    title = f"预测：{config.CLASS_NAMES[predicted]}（{predictions[0][1]*100:.1f}%）"
    if true_label is not None:
        title = f"真实：{config.CLASS_NAMES[true_label]}    " + title
    subtitle = "  ".join(f"{config.CLASS_NAMES[i]}:{p*100:.1f}%" for i, p in predictions)
    name = f"单张预测_{'自定义' if args.image else ('随机' if args.random else args.index)}.png"
    figure_path = save_single_figure(tensor, title, config.FIG_DIR / name, subtitle)
    print(f"\n已保存预测图：{figure_path}")
    print(LINE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
