# -*- coding: utf-8 -*-
"""可视化多张图片的预测结果（默认 16 张，标签用绿/红区分对错）。

运行方式：
    python visualize.py                       # 随机 16 张测试集图片
    python visualize.py --num 20 --mode wrong  # 只看预测错的 20 张

输出：
    runs/figures/预测结果可视化.png
    runs/logs/预测结果明细.csv
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config
from dataset import denormalize, setup_plot_style
from evaluate import load_trained_model
from predict import load_test_dataset
from train import get_device

LINE = "=" * 76


def predict_batch(model, tensors: list[torch.Tensor], device) -> np.ndarray:
    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(batch), dim=1)
    return probabilities.cpu().numpy()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="可视化 CIFAR-10 预测结果")
    parser.add_argument("--num", type=int, default=16, help="可视化多少张（至少 10）")
    parser.add_argument("--mode", choices=["random", "correct", "wrong"], default="random",
                        help="随机 / 只挑预测对的 / 只挑预测错的")
    parser.add_argument("--cols", type=int, default=4, help="每行放几张")
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args(argv)

    if args.num < 10:
        print("提示：实验要求至少可视化 10 张，已自动调整为 10 张。")
        args.num = 10

    config.ensure_dirs()
    device = get_device()
    model, _ = load_trained_model(device=device)
    dataset, _ = load_test_dataset()

    print(LINE)
    print("CIFAR-10 预测结果可视化")
    print(LINE)
    print(f"从官方测试集（{len(dataset)} 张）中挑选 {args.num} 张，模式：{args.mode}")

    rng = random.Random(args.seed)
    # 为了让"挑对的/挑错的"可行，先把整个测试集跑一遍拿到预测结果
    all_indices = list(range(len(dataset)))
    all_tensors = []
    all_labels = []
    for index in all_indices:
        tensor, label = dataset[index]
        all_tensors.append(tensor)
        all_labels.append(label)

    probabilities = []
    batch_size = 512
    for start in range(0, len(all_tensors), batch_size):
        chunk = all_tensors[start:start + batch_size]
        probabilities.append(predict_batch(model, chunk, device))
    probabilities = np.concatenate(probabilities, axis=0)
    predictions = probabilities.argmax(axis=1)
    labels = np.array(all_labels)
    correctness = predictions == labels

    print(f"整个测试集准确率：{correctness.mean()*100:.2f}%（{correctness.sum()}/{len(labels)}）")

    if args.mode == "correct":
        pool = [i for i in all_indices if correctness[i]]
    elif args.mode == "wrong":
        pool = [i for i in all_indices if not correctness[i]]
    else:
        pool = all_indices
    if not pool:
        print(f"没有符合条件的样本（模式：{args.mode}）")
        return 1

    chosen = rng.sample(pool, min(args.num, len(pool)))
    rows = int(np.ceil(len(chosen) / args.cols))

    setup_plot_style()
    fig, axes = plt.subplots(rows, args.cols, figsize=(3.2 * args.cols, 3.6 * rows))
    axes_list = list(np.atleast_1d(axes).ravel())

    records = []
    for ax, index in zip(axes_list, chosen):
        tensor = all_tensors[index]
        truth = int(labels[index])
        prediction = int(predictions[index])
        confidence = float(probabilities[index][prediction])
        correct = prediction == truth

        ax.imshow(denormalize(tensor).permute(1, 2, 0).numpy())
        ax.set_title(
            f"真实：{config.CLASS_NAMES[truth]}\n预测：{config.CLASS_NAMES[prediction]}（{confidence*100:.1f}%）",
            fontsize=10,
            color="#2E7D32" if correct else "#C62828",
        )
        ax.axis("off")
        records.append({
            "测试集下标": index,
            "真实类别": config.CLASS_NAMES[truth],
            "预测类别": config.CLASS_NAMES[prediction],
            "置信度": round(confidence, 4),
            "是否正确": "是" if correct else "否",
        })

    for ax in axes_list[len(chosen):]:
        ax.axis("off")

    correct_count = sum(1 for record in records if record["是否正确"] == "是")
    fig.suptitle(
        f"CIFAR-10 预测结果可视化（{len(chosen)} 张，其中正确 {correct_count} 张，"
        f"正确率 {correct_count/len(chosen)*100:.1f}%）",
        fontsize=13,
    )
    fig.tight_layout()
    figure_path = config.FIG_DIR / f"预测结果可视化_{args.mode}.png"
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)

    csv_path = config.LOG_DIR / f"预测结果明细_{args.mode}.csv"
    header = ",".join(records[0].keys())
    lines = [header] + [",".join(str(value) for value in record.values()) for record in records]
    csv_path.write_text("\n".join(lines), encoding="utf-8-sig")

    print(f"已保存图片：{figure_path}")
    print(f"明细数据：{csv_path}")
    print(LINE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
