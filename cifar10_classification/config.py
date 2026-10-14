# -*- coding: utf-8 -*-
"""项目配置：路径、类别名、超参数。

数据与训练产物默认放在 E 盘（本机 C 盘空间紧张），可用环境变量覆盖：
    CIFAR10_DATA_DIR   数据集目录
    CIFAR10_RUN_DIR    训练产物目录（权重、图片、日志）
如果默认盘符不存在（换到别的电脑），自动回退到项目目录下的 data / runs。
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent


def _resolve_dir(env_name: str, default: str, fallback: Path) -> Path:
    """优先环境变量，其次默认路径，最后回退到项目目录。"""
    value = os.environ.get(env_name)
    if value:
        return Path(value)
    candidate = Path(default)
    if candidate.drive and Path(candidate.drive + os.sep).exists():
        return candidate
    return fallback


# ---------------------------------------------------------------- 路径
DATA_DIR = _resolve_dir("CIFAR10_DATA_DIR", r"E:\cifar10\data", PROJECT_DIR / "data")
RUN_DIR = _resolve_dir("CIFAR10_RUN_DIR", r"E:\cifar10\runs", PROJECT_DIR / "runs")
CKPT_DIR = RUN_DIR / "checkpoints"
FIG_DIR = RUN_DIR / "figures"
LOG_DIR = RUN_DIR / "logs"
SPLIT_FILE = RUN_DIR / "splits.json"
BEST_MODEL = CKPT_DIR / "best_model.pt"
LAST_MODEL = CKPT_DIR / "last_model.pt"

# ---------------------------------------------------------------- 类别
# 顺序与 CIFAR-10 官方 meta 文件一致
CLASS_NAMES = ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"]
CLASS_NAMES_EN = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
NUM_CLASSES = len(CLASS_NAMES)

# 归一化参数：CIFAR-10 官方训练集统计出来的固定常数。
# 注意：这是"只用训练集算出来的"，没有用到验证集/测试集的任何信息。
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

# ---------------------------------------------------------------- 超参数
SEED = 42
BATCH_SIZE = 128
EPOCHS = 25
LR = 1e-3
WEIGHT_DECAY = 5e-4

# 数据划分：官方训练集 50000 张按类别分层抽样
#   训练集 = 每类 2000 张（共 20000）
#   验证集 = 每类 1000 张（共 10000）
# 测试集固定使用官方测试集全部 10000 张
TRAIN_PER_CLASS = 2000
VAL_PER_CLASS = 1000

NUM_WORKERS = 0          # Windows 下用 0 最稳，数据集已全部读进内存，不影响速度


def ensure_dirs() -> None:
    for directory in (DATA_DIR, RUN_DIR, CKPT_DIR, FIG_DIR, LOG_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def describe() -> str:
    return (
        f"数据目录  : {DATA_DIR}\n"
        f"产物目录  : {RUN_DIR}\n"
        f"超参数    : epochs={EPOCHS}, batch={BATCH_SIZE}, lr={LR}, weight_decay={WEIGHT_DECAY}\n"
        f"数据划分  : 训练集每类 {TRAIN_PER_CLASS} 张，验证集每类 {VAL_PER_CLASS} 张，测试集用官方全部 10000 张"
    )
