# -*- coding: utf-8 -*-
"""数据准备：解析 CIFAR-10 原始数据、统计类别分布、划分数据集、预处理、DataLoader。

为什么不用 torchvision.datasets.CIFAR10 的自动下载？
    官方下载地址 www.cs.toronto.edu 在国内经常连不上，torchvision 的自动下载会直接卡死。
    所以这里自己解析 CIFAR-10 官方的 pickle 格式：只要把 cifar-10-python.tar.gz 放到
    数据目录，整个流程就能离线跑通，也顺便把"数据准备"这一步讲得更清楚。

数据划分（训练/验证/测试三者互不重叠）：
    官方训练集 50000 张 -> 按类别分层抽样，划出训练集与验证集
    官方测试集 10000 张 -> 完整保留，作为最终测试集
划分结果保存在 splits.json 中，训练、评估、可视化都用同一份划分，保证可复现。
"""

from __future__ import annotations

import hashlib
import json
import pickle
import random
import tarfile
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import config

matplotlib.use("Agg")  # 无界面环境也能出图
import matplotlib.pyplot as plt  # noqa: E402

ARCHIVE = config.DATA_DIR / "cifar-10-python.tar.gz"
RAW_DIR = config.DATA_DIR / "cifar-10-batches-py"
# 官方压缩包的 MD5，用来确认下载完整、没有被篡改
ARCHIVE_MD5 = "c58f30108f718f92721af3b95e74349a"
TRAIN_BATCH_FILES = [f"data_batch_{index}" for index in range(1, 6)]
TEST_BATCH_FILE = "test_batch"

LINE = "=" * 76


# --------------------------------------------------------------------------
# 1. 解压与解析原始数据
# --------------------------------------------------------------------------
def md5_of(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_raw_data(verbose: bool = True) -> Path:
    """确保解压好的 cifar-10-batches-py 目录存在，没有就解压 tar.gz。"""
    if (RAW_DIR / TEST_BATCH_FILE).exists():
        return RAW_DIR
    if not ARCHIVE.exists():
        raise FileNotFoundError(
            f"\n找不到数据集压缩包：{ARCHIVE}\n"
            "请先下载 cifar-10-python.tar.gz 放到上面这个目录（约 163MB）。\n"
            "国内可用镜像（实测 3 MB/s）：\n"
            "  https://dataset.bj.bcebos.com/cifar/cifar-10-python.tar.gz\n"
            "官方地址：https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        )

    digest = md5_of(ARCHIVE)
    if verbose:
        status = "与官方一致" if digest == ARCHIVE_MD5 else "与官方不一致（建议重新下载）"
        print(f"  压缩包 MD5：{digest}（{status}）")

    print(f"  正在解压 {ARCHIVE.name} ...")
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ARCHIVE, "r:gz") as tar:
        try:
            tar.extractall(config.DATA_DIR, filter="data")
        except TypeError:          # Python < 3.11.4 没有 filter 参数
            tar.extractall(config.DATA_DIR)
    print(f"  解压完成：{RAW_DIR}")
    return RAW_DIR


def load_batch(path: Path) -> tuple[np.ndarray, list[int]]:
    """读取一个 batch 文件，返回（图像数组 [N,32,32,3] uint8，标签列表）。"""
    with path.open("rb") as handle:
        # CIFAR-10 是 Python 2 打包的，键和字符串都是 bytes，所以要指定 encoding="bytes"
        entry = pickle.load(handle, encoding="bytes")
    data = entry[b"data"]                                   # (N, 3072) uint8
    labels = entry[b"labels"]
    # 原始数据是 3x32x32 按通道展开的，还原成 [N,32,32,3] 方便交给 Pillow 处理
    images = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return images, labels


def load_official_train() -> tuple[np.ndarray, list[int]]:
    """官方训练集：5 个 batch，共 50000 张。"""
    images_list, labels_list = [], []
    for name in TRAIN_BATCH_FILES:
        images, labels = load_batch(RAW_DIR / name)
        images_list.append(images)
        labels_list.extend(labels)
    return np.concatenate(images_list, axis=0), labels_list


def load_official_test() -> tuple[np.ndarray, list[int]]:
    """官方测试集：10000 张。"""
    return load_batch(RAW_DIR / TEST_BATCH_FILE)


# --------------------------------------------------------------------------
# 2. 类别分布统计
# --------------------------------------------------------------------------
def class_counts(labels: list[int]) -> np.ndarray:
    return np.bincount(np.asarray(labels), minlength=config.NUM_CLASSES)


def setup_plot_style() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150


def plot_class_distribution(counts: np.ndarray, title: str, path: Path) -> Path:
    """画类别分布柱状图。"""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    positions = range(config.NUM_CLASSES)
    bars = ax.bar(positions, counts, color=plt.get_cmap("tab10")(range(config.NUM_CLASSES)))
    ax.bar_label(bars, fontsize=9, padding=2)
    ax.set_xticks(list(positions))
    ax.set_xticklabels([f"{cn}\n{en}" for cn, en in zip(config.CLASS_NAMES, config.CLASS_NAMES_EN)], fontsize=9)
    ax.set_ylabel("图片数量")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# 3. 划分训练集 / 验证集（测试集用官方测试集，始终独立）
# --------------------------------------------------------------------------
def make_splits(
    train_labels: list[int],
    train_per_class: int,
    val_per_class: int,
    seed: int = config.SEED,
) -> dict[str, list[int]]:
    """按类别分层抽样：每类取 train_per_class 张做训练，再取 val_per_class 张做验证。"""
    rng = random.Random(seed)
    by_class: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(train_labels):
        by_class[label].append(index)

    train_indices: list[int] = []
    val_indices: list[int] = []
    for label in range(config.NUM_CLASSES):
        indices = by_class[label][:]
        rng.shuffle(indices)
        if train_per_class + val_per_class > len(indices):
            raise ValueError(f"类别 {label} 只有 {len(indices)} 张，不够划分")
        train_indices.extend(indices[:train_per_class])
        val_indices.extend(indices[train_per_class:train_per_class + val_per_class])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return {"train": train_indices, "val": val_indices}


def save_splits(splits: dict[str, list[int]], path: Path = config.SPLIT_FILE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: [int(v) for v in values] for key, values in splits.items()}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def load_splits(path: Path = config.SPLIT_FILE) -> dict[str, list[int]] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# 4. 预处理与 Dataset / DataLoader
# --------------------------------------------------------------------------
def build_transform(is_train: bool) -> transforms.Compose:
    """训练集做数据增强，验证/测试集只做归一化（不能引入随机性）。"""
    if is_train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),      # 先补边再随机裁剪
            transforms.RandomHorizontalFlip(),         # 随机水平翻转
            transforms.ToTensor(),                     # -> [0,1] 的 float32 张量
            transforms.Normalize(config.CIFAR_MEAN, config.CIFAR_STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(config.CIFAR_MEAN, config.CIFAR_STD),
    ])


class CIFARDataset(Dataset):
    """把内存里的 numpy 图像数组包装成 PyTorch 数据集。"""

    def __init__(
        self,
        images: np.ndarray,
        labels: list[int],
        indices: list[int] | None = None,
        transform=None,
    ):
        self.images = images
        self.labels = list(labels)
        self.indices = list(range(len(self.labels))) if indices is None else list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        index = self.indices[position]
        image = Image.fromarray(self.images[index])         # HWC uint8 -> PIL
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.labels[index])


def build_dataloaders(
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
) -> tuple[dict[str, DataLoader], dict]:
    """构建 train / val / test 三个 DataLoader，返回 (loaders, info)。"""
    ensure_raw_data()
    train_images, train_labels = load_official_train()
    test_images, test_labels = load_official_test()

    splits = load_splits()
    if splits is None:
        splits = make_splits(train_labels, config.TRAIN_PER_CLASS, config.VAL_PER_CLASS)
        save_splits(splits)
        print(f"  已生成数据划分并保存：{config.SPLIT_FILE}")

    overlap = set(splits["train"]) & set(splits["val"])
    assert not overlap, f"训练集与验证集出现重叠：{len(overlap)} 张"

    train_set = CIFARDataset(train_images, train_labels, splits["train"], build_transform(True))
    val_set = CIFARDataset(train_images, train_labels, splits["val"], build_transform(False))
    test_set = CIFARDataset(test_images, test_labels, None, build_transform(False))

    loaders = {
        "train": DataLoader(train_set, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=torch.cuda.is_available()),
        "val": DataLoader(val_set, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=torch.cuda.is_available()),
        "test": DataLoader(test_set, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=torch.cuda.is_available()),
    }
    info = {
        "train_size": len(train_set),
        "val_size": len(val_set),
        "test_size": len(test_set),
        "train_images": train_images,
        "train_labels": train_labels,
        "test_images": test_images,
        "test_labels": test_labels,
        "splits": splits,
    }
    return loaders, info


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """把归一化后的图片还原到 [0,1]，用于可视化。"""
    mean = torch.tensor(config.CIFAR_MEAN).view(3, 1, 1)
    std = torch.tensor(config.CIFAR_STD).view(3, 1, 1)
    return (tensor.cpu() * std + mean).clamp(0, 1)


# --------------------------------------------------------------------------
# 5. 数据准备主流程
# --------------------------------------------------------------------------
def main() -> int:
    print(LINE)
    print("CIFAR-10 数据准备")
    print(LINE)
    config.ensure_dirs()
    print(config.describe())
    print(f"\n[1/4] 检查并解压原始数据 -> {config.DATA_DIR}")
    ensure_raw_data()

    print("\n[2/4] 读取官方数据")
    train_images, train_labels = load_official_train()
    test_images, test_labels = load_official_test()
    print(f"  官方训练集：{train_images.shape[0]} 张，图像形状 {train_images.shape[1:]}")
    print(f"  官方测试集：{test_images.shape[0]} 张")

    print("\n[3/4] 统计各类别分布")
    train_counts = class_counts(train_labels)
    test_counts = class_counts(test_labels)
    print(f"  {'类别':<8}{'英文名':<12}{'训练集':>8}{'测试集':>8}")
    for index, (cn, en) in enumerate(zip(config.CLASS_NAMES, config.CLASS_NAMES_EN)):
        print(f"  {cn:<8}{en:<12}{train_counts[index]:>8}{test_counts[index]:>8}")
    fig1 = plot_class_distribution(train_counts, "CIFAR-10 官方训练集类别分布（50000 张）",
                                   config.FIG_DIR / "数据分布_官方训练集.png")
    fig2 = plot_class_distribution(test_counts, "CIFAR-10 官方测试集类别分布（10000 张）",
                                   config.FIG_DIR / "数据分布_官方测试集.png")
    print(f"  已保存分布图：{fig1.name}、{fig2.name}")
    print("  说明：官方数据是严格均衡的，每个类别都是 5000 + 1000 张。")

    print("\n[4/4] 划分训练集 / 验证集")
    splits = make_splits(train_labels, config.TRAIN_PER_CLASS, config.VAL_PER_CLASS)
    path = save_splits(splits)
    train_sub_counts = class_counts([train_labels[i] for i in splits["train"]])
    val_sub_counts = class_counts([train_labels[i] for i in splits["val"]])
    print(f"  训练集：{len(splits['train'])} 张（每类 {train_sub_counts.min()}~{train_sub_counts.max()} 张）")
    print(f"  验证集：{len(splits['val'])} 张（每类 {val_sub_counts.min()}~{val_sub_counts.max()} 张）")
    print(f"  测试集：{len(test_labels)} 张（官方测试集，完整使用）")
    print(f"  划分文件：{path}")
    print("  三者互不重叠：验证集从官方训练集里划出，测试集用官方测试集；")
    print("  归一化用的均值方差是官方公布的训练集统计值，没有使用验证集/测试集信息。")

    print(LINE)
    print("数据准备完成。")
    print(LINE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
