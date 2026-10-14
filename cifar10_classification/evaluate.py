# -*- coding: utf-8 -*-
"""模型评估：在官方测试集上计算准确率、混淆矩阵、每类精确率/召回率/F1。

运行方式：
    python evaluate.py              # 在官方测试集（10000 张）上评估
    python evaluate.py --split val  # 也可以看验证集

输出的文件：
    runs/figures/测试集混淆矩阵.png        混淆矩阵热力图
    runs/logs/confusion_matrix.csv         混淆矩阵原始数值
    runs/logs/评估报告.txt                  各项指标与文字分析
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config
from dataset import build_dataloaders, setup_plot_style
from model import build_model
from train import get_device

LINE = "=" * 76


def load_trained_model(path: Path = config.BEST_MODEL, device: torch.device | None = None):
    """加载训练好的最佳模型。"""
    device = device or get_device()
    if not path.exists():
        raise FileNotFoundError(f"找不到模型文件：{path}\n请先运行 python train.py")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def collect_predictions(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """跑完整个数据集，收集真实标签与预测标签。"""
    y_true, y_pred = [], []
    for images, labels in loader:
        outputs = model(images.to(device, non_blocking=True))
        y_true.extend(labels.tolist())
        y_pred.extend(outputs.argmax(dim=1).cpu().tolist())
    return np.array(y_true), np.array(y_pred)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """手写混淆矩阵：cm[i][j] = 真实是 i、被预测成 j 的数量。"""
    matrix = np.zeros((config.NUM_CLASSES, config.NUM_CLASSES), dtype=int)
    for truth, prediction in zip(y_true, y_pred):
        matrix[truth][prediction] += 1
    return matrix


def per_class_metrics(matrix: np.ndarray) -> list[dict]:
    """由混淆矩阵算每一类的精确率、召回率、F1。"""
    metrics = []
    for index in range(config.NUM_CLASSES):
        tp = matrix[index, index]
        fp = matrix[:, index].sum() - tp
        fn = matrix[index, :].sum() - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics.append({
            "class": index,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(matrix[index, :].sum()),
        })
    return metrics


def plot_confusion_matrix(matrix: np.ndarray, path: Path, title: str) -> Path:
    """画混淆矩阵热力图（按行归一化，也就是每一类的召回率分布）。"""
    setup_plot_style()
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(9.5, 8))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(image, ax=ax, fraction=0.046, label="该行占比（召回率）")

    labels = [f"{cn}\n{en}" for cn, en in zip(config.CLASS_NAMES, config.CLASS_NAMES_EN)]
    ax.set_xticks(range(config.NUM_CLASSES))
    ax.set_yticks(range(config.NUM_CLASSES))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("模型预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title(title)

    for row in range(config.NUM_CLASSES):
        for col in range(config.NUM_CLASSES):
            value = normalized[row, col]
            ax.text(col, row, f"{value:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if value > 0.5 else "#333333")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def analyze(matrix: np.ndarray) -> dict:
    """找出最容易混淆的类别对和最差/最好的类别。"""
    pairs = []
    for row in range(config.NUM_CLASSES):
        for col in range(config.NUM_CLASSES):
            if row != col and matrix[row, col] > 0:
                pairs.append((int(matrix[row, col]), row, col))
    pairs.sort(reverse=True)

    per_class_acc = [
        (matrix[i, i] / matrix[i, :].sum() if matrix[i, :].sum() else 0.0, i)
        for i in range(config.NUM_CLASSES)
    ]
    per_class_acc.sort()
    return {"pairs": pairs[:5], "worst": per_class_acc[:3], "best": per_class_acc[-3:][::-1]}


def save_report(lines: list[str], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CIFAR-10 模型评估")
    parser.add_argument("--split", choices=["test", "val"], default="test", help="评估哪个集合")
    parser.add_argument("--model", default=str(config.BEST_MODEL), help="模型权重路径")
    args = parser.parse_args(argv)

    config.ensure_dirs()
    device = get_device()

    print(LINE)
    print("CIFAR-10 模型评估")
    print(LINE)

    model, checkpoint = load_trained_model(Path(args.model), device)
    print(f"已加载模型：{args.model}")
    print(f"  该模型来自第 {checkpoint.get('epoch', '?')} 轮，验证准确率 {checkpoint.get('val_acc', float('nan'))*100:.2f}%")

    loaders, info = build_dataloaders()
    split = args.split
    y_true, y_pred = collect_predictions(model, loaders[split], device)

    accuracy = (y_true == y_pred).mean()
    matrix = confusion_matrix(y_true, y_pred)
    metrics = per_class_metrics(matrix)
    analysis = analyze(matrix)

    split_name = "官方测试集" if split == "test" else "验证集"
    print(f"\n评估集合：{split_name}（{len(y_true)} 张）")
    print(f"总体准确率：{accuracy*100:.2f}%")

    print(f"\n{'类别':<8}{'精确率':>10}{'召回率':>10}{'F1':>10}{'样本数':>8}")
    for item in metrics:
        name = config.CLASS_NAMES[item["class"]]
        print(f"{name:<8}{item['precision']*100:>9.2f}%{item['recall']*100:>9.2f}%{item['f1']*100:>9.2f}%{item['support']:>8}")
    macro_f1 = float(np.mean([item["f1"] for item in metrics]))
    print(f"宏平均 F1：{macro_f1*100:.2f}%")

    print("\n最容易混淆的类别对（真实 -> 预测）：")
    for count, row, col in analysis["pairs"]:
        print(f"  {config.CLASS_NAMES[row]} -> {config.CLASS_NAMES[col]}：{count} 张")

    worst = "、".join(f"{config.CLASS_NAMES[i]}（{acc*100:.1f}%）" for acc, i in analysis["worst"])
    best = "、".join(f"{config.CLASS_NAMES[i]}（{acc*100:.1f}%）" for acc, i in analysis["best"])
    print(f"\n识别最差的类别：{worst}")
    print(f"识别最好的类别：{best}")

    figure_path = config.FIG_DIR / f"{'测试集' if split == 'test' else '验证集'}混淆矩阵.png"
    plot_confusion_matrix(matrix, figure_path, f"CIFAR-10 混淆矩阵（{split_name}，总体准确率 {accuracy*100:.2f}%）")

    matrix_path = config.LOG_DIR / "confusion_matrix.csv"
    header = "真实\\预测," + ",".join(config.CLASS_NAMES_EN)
    body = [",".join([config.CLASS_NAMES_EN[row]] + [str(v) for v in matrix[row]]) for row in range(config.NUM_CLASSES)]
    save_report([header] + body, matrix_path)

    report_lines = [
        f"评估集合：{split_name}（{len(y_true)} 张）",
        f"总体准确率：{accuracy*100:.2f}%",
        f"宏平均 F1：{macro_f1*100:.2f}%",
        "",
        f"{'类别':<10}{'精确率':>10}{'召回率':>10}{'F1':>10}",
    ]
    for item in metrics:
        report_lines.append(
            f"{config.CLASS_NAMES[item['class']]:<10}{item['precision']*100:>9.2f}%"
            f"{item['recall']*100:>9.2f}%{item['f1']*100:>9.2f}%"
        )
    report_lines += ["", "最容易混淆的类别对："]
    report_lines += [f"  {config.CLASS_NAMES[r]} -> {config.CLASS_NAMES[c]}：{n} 张" for n, r, c in analysis["pairs"]]
    report_lines += ["", f"识别最差的类别：{worst}", f"识别最好的类别：{best}"]
    report_path = save_report(report_lines, config.LOG_DIR / "评估报告.txt")

    print(f"\n混淆矩阵图：{figure_path}")
    print(f"混淆矩阵数据：{matrix_path}")
    print(f"评估报告：{report_path}")
    print(LINE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
