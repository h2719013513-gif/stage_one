# -*- coding: utf-8 -*-
"""生成实验 2 使用的原始数据集 data/raw/students_scores_raw.csv。

数据来源为程序模拟生成（不使用任何第三方库），固定随机种子，结果可复现。
为了让后续的数据清洗环节有真实的处理对象，原始数据中故意混入了：
    * 缺失值（空白单元格、"缺考" 文本）
    * 异常值（-1 分、200 分、999 分、大于 1 的出勤率）
    * 重复记录（同一条记录被登记了两次）

运行方式：
    python scripts/generate_dataset.py
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "raw" / "students_scores_raw.csv"

FIELDS = ["学号", "姓名", "班级", "性别", "数学", "英语", "Python", "出勤率"]
CLASSES = ["计科2101", "计科2102", "计科2103", "计科2104"]
SURNAMES = "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜戚谢邹喻柏水"
GIVEN_NAMES = [
    "伟", "芳", "娜", "秀英", "敏", "静", "丽", "强", "磊", "洋",
    "艳", "勇", "军", "杰", "娟", "涛", "明", "超", "秀兰", "霞",
    "平", "刚", "桂英", "文", "辉", "建国", "玉兰", "雨轩", "思远",
    "梓萱", "俊杰", "浩然", "可欣", "一鸣", "佳怡", "宇轩", "若曦",
]


def random_score(rng: random.Random, mean: float, sigma: float) -> float:
    """生成一个大致服从正态分布、并限制在 [35, 100] 内的成绩。"""
    value = rng.gauss(mean, sigma)
    return round(max(35.0, min(100.0, value)), 1)


def build_rows(count: int = 121, seed: int = 20260918) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []

    for index in range(1, count + 1):
        name = rng.choice(SURNAMES) + rng.choice(GIVEN_NAMES)
        rows.append(
            {
                "学号": f"2021{index:04d}",
                "姓名": name,
                "班级": CLASSES[(index - 1) % len(CLASSES)],
                "性别": rng.choice(["男", "女"]),
                "数学": f"{random_score(rng, 72, 12):.1f}",
                "英语": f"{random_score(rng, 75, 11):.1f}",
                "Python": f"{random_score(rng, 78, 10):.1f}",
                "出勤率": f"{rng.uniform(0.82, 1.0):.2f}",
            }
        )

    # ---- 人为注入脏数据 ----
    rows[6]["数学"] = ""              # 缺失：空白单元格
    rows[14]["英语"] = "缺考"          # 缺失：非数值文本
    rows[22]["Python"] = "-1"          # 异常：负数
    rows[30]["数学"] = "200"           # 异常：超出满分
    rows[52]["出勤率"] = "1.35"         # 异常：出勤率大于 100%
    rows[59]["Python"] = ""            # 缺失：空白单元格
    rows[72]["英语"] = "999"           # 异常：明显不可能的分数
    rows[78]["英语"] = ""              # 缺失：空白单元格
    rows[87]["数学"] = ""              # 缺失：整行成绩都缺失
    rows[87]["英语"] = ""
    rows[87]["Python"] = ""
    rows[99]["出勤率"] = "无"           # 缺失：无法解析的文本

    # 重复登记一条记录
    rows.append(dict(rows[43]))

    return rows


def main() -> None:
    rows = build_rows()
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig 让 Excel 直接双击打开时不会出现中文乱码
    with RAW_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"已生成原始数据：{RAW_CSV}")
    print(f"记录数：{len(rows)}（其中 1 条为重复记录，清洗后剩余 {len({r['学号'] for r in rows})} 条）")


if __name__ == "__main__":
    main()
