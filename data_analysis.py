# -*- coding: utf-8 -*-
"""实验 2：Python 数据处理与可视化

完整流程：读取 CSV -> 处理缺失值与异常值 -> 计算统计指标 -> 绘制统计图表 -> 保存新的 CSV。
全部处理逻辑都封装在函数中，可重复运行，不需要手工修改原始 CSV。

运行方式：
    python data_analysis.py
    python data_analysis.py --input data/raw/students_scores_raw.csv --output-csv data/processed/students_scores_clean.csv

依赖：pandas、matplotlib（见 requirements.txt）
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")  # 无界面环境下也能出图
import matplotlib.pyplot as plt  # noqa: E402  （必须在 use("Agg") 之后导入）

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data" / "raw" / "students_scores_raw.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "students_scores_clean.csv"
DEFAULT_STATS = ROOT / "output" / "statistics_summary.csv"
DEFAULT_FIG_DIR = ROOT / "output" / "figures"

SCORE_COLS = ["数学", "英语", "Python"]
NUMERIC_COLS = SCORE_COLS + ["出勤率"]
# 允许的取值范围：成绩 0~100 分，出勤率 0~1（即 0%~100%）
VALUE_RANGE = {"数学": (0, 100), "英语": (0, 100), "Python": (0, 100), "出勤率": (0, 1)}
# 这些文本都按“缺失值”处理
# 这些文本都按“缺失值”处理（"<na>" 是 pandas 缺失值转成字符串后的形式）
MISSING_TOKENS = {"", "na", "<na>", "n/a", "null", "none", "nan", "missing", "缺考", "无", "-", "--", "?"}

LINE = "=" * 78
GRADE_BINS = [-0.01, 60, 70, 80, 90, 100.01]
GRADE_LABELS = ["不及格", "及格", "中等", "良好", "优秀"]
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


# --------------------------------------------------------------------------
# 1. 读取数据
# --------------------------------------------------------------------------
def load_data(csv_path: str | Path) -> pd.DataFrame:
    """读取原始 CSV 文件，返回 DataFrame。"""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到数据文件：{csv_path}")
    # 原始文件用 utf-8-sig 保存，读取时同样用它才能正确解析中文表头
    frame = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str)
    frame.columns = [str(col).strip() for col in frame.columns]
    return frame


# --------------------------------------------------------------------------
# 2. 处理缺失值与异常数据
# --------------------------------------------------------------------------
def clean_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """清洗数据：去重、识别缺失值、剔除异常值，并用同班级中位数填补。

    返回 (清洗后的数据, 处理报告)。
    """
    data = frame.copy()
    report: dict = {
        "原始记录数": int(len(data)),
        "重复记录删除": 0,
        "缺失值": {col: 0 for col in NUMERIC_COLS},
        "异常值": {col: 0 for col in NUMERIC_COLS},
        "填补值": {col: 0 for col in NUMERIC_COLS},
    }

    # 2.1 删除重复记录（同一学号只保留第一条）
    before = len(data)
    data = data.drop_duplicates(subset=["学号"], keep="first").reset_index(drop=True)
    report["重复记录删除"] = before - len(data)

    # 2.2 把 "缺考"、空白之类的文本统一识别为缺失值；把无法解析的文本记为异常值
    for col in NUMERIC_COLS:
        text = data[col].astype(str).str.strip()
        # 空单元格在 pandas 读入后是缺失值（NaN/NA），要和 "缺考" 这类文本一起视为缺失
        is_missing_token = data[col].isna() | text.str.lower().isin(MISSING_TOKENS)
        numeric = pd.to_numeric(text.where(~is_missing_token), errors="coerce")

        unparsable = ~is_missing_token & numeric.isna()
        low, high = VALUE_RANGE[col]
        out_of_range = numeric.notna() & ((numeric < low) | (numeric > high))

        report["缺失值"][col] = int(is_missing_token.sum())
        report["异常值"][col] = int(unparsable.sum() + out_of_range.sum())
        numeric = numeric.mask(out_of_range)  # 异常值同样置为缺失，等待填补
        data[col] = numeric

    # 2.3 用同班级同科目中位数填补缺失，班级整体缺失时退回全体中位数
    for col in NUMERIC_COLS:
        if not data[col].isna().any():
            continue
        global_median = data[col].median()
        class_median = data.groupby("班级")[col].transform("median")
        report["填补值"][col] = int(data[col].isna().sum())
        data[col] = data[col].fillna(class_median).fillna(global_median)
        data[col] = data[col].round(1 if col in SCORE_COLS else 2)

    # 2.4 派生指标：平均分与等级
    data["平均分"] = data[SCORE_COLS].mean(axis=1).round(2)
    data["等级"] = pd.cut(data["平均分"], bins=GRADE_BINS, labels=GRADE_LABELS, right=False)
    data = data.sort_values("学号").reset_index(drop=True)
    return data, report


# --------------------------------------------------------------------------
# 3. 统计指标
# --------------------------------------------------------------------------
def compute_statistics(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """计算每列的记录数、平均值、最大值、最小值、标准差和中位数。"""
    rows = []
    for col in columns:
        series = data[col]
        rows.append(
            {
                "指标": col,
                "记录数": int(series.count()),
                "平均值": round(float(series.mean()), 3),
                "最大值": round(float(series.max()), 3),
                "最小值": round(float(series.min()), 3),
                # ddof=1 表示样本标准差
                "标准差": round(float(series.std(ddof=1)), 3),
                "中位数": round(float(series.median()), 3),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 4. 可视化
# --------------------------------------------------------------------------
def setup_plot_style() -> None:
    """设置中文字体，避免图片里的中文变成方块。"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150


def plot_score_histogram(data: pd.DataFrame, fig_dir: Path) -> Path:
    """图 1：三门课程成绩分布直方图。"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    bins = list(range(0, 101, 10))
    for ax, col, color in zip(axes, SCORE_COLS, PALETTE):
        ax.hist(data[col], bins=bins, color=color, edgecolor="white", alpha=0.9)
        ax.axvline(
            data[col].mean(),
            color="#C44E52",
            linestyle="--",
            linewidth=1.5,
            label=f"平均值 {data[col].mean():.1f}",
        )
        ax.set_title(f"{col}成绩分布")
        ax.set_xlabel("分数")
        ax.set_ylabel("人数")
        ax.set_xticks(bins)
        ax.grid(axis="y", alpha=0.3)
        ax.legend()
    fig.suptitle("图 1  三门课程成绩分布直方图", fontsize=14)
    fig.tight_layout()
    path = fig_dir / "fig1_成绩分布直方图.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_class_average_bar(data: pd.DataFrame, fig_dir: Path) -> Path:
    """图 2：各班各科平均分分组柱状图。"""
    grouped = data.groupby("班级")[SCORE_COLS].mean().round(2)
    fig, ax = plt.subplots(figsize=(10, 5))
    positions = list(range(len(grouped)))
    width = 0.26
    for offset, (col, color) in enumerate(zip(SCORE_COLS, PALETTE)):
        shifted = [pos + (offset - 1) * width for pos in positions]
        bars = ax.bar(shifted, grouped[col], width=width, label=col, color=color)
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    ax.set_xticks(positions)
    ax.set_xticklabels(grouped.index)
    ax.set_ylabel("平均分")
    ax.set_ylim(0, 100)
    ax.set_title("图 2  各班级各科平均分对比")
    ax.grid(axis="y", alpha=0.3)
    # 图例放到坐标区下方，避免遮住第一组柱子的数值标签
    ax.legend(title="科目", loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3)
    fig.tight_layout()
    path = fig_dir / "fig2_班级平均分柱状图.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_score_boxplot(data: pd.DataFrame, fig_dir: Path) -> Path:
    """图 3：三门课程成绩箱线图（用于观察离散程度与离群点）。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    box = ax.boxplot([data[col] for col in SCORE_COLS], tick_labels=SCORE_COLS, patch_artist=True)
    for patch, color in zip(box["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    ax.set_ylabel("分数")
    ax.set_ylim(0, 105)
    ax.set_title("图 3  三门课程成绩箱线图")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = fig_dir / "fig3_成绩箱线图.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_grade_pie(data: pd.DataFrame, fig_dir: Path) -> Path:
    """图 4：成绩等级分布饼图。"""
    counts = data["等级"].value_counts().reindex(GRADE_LABELS).fillna(0)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    _, _, autotexts = ax.pie(
        counts,
        labels=[f"{label}\n{int(count)} 人" for label, count in counts.items()],
        autopct="%1.1f%%",
        colors=PALETTE + ["#937860"],
        startangle=90,
        counterclock=False,
        wedgeprops={"edgecolor": "white"},
    )
    for text in autotexts:
        text.set_color("white")
        text.set_fontsize(9)
    ax.set_title("图 4  成绩等级分布")
    fig.tight_layout()
    path = fig_dir / "fig4_等级分布饼图.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_all(data: pd.DataFrame, fig_dir: Path) -> list[Path]:
    """绘制全部图表并返回图片路径列表。"""
    fig_dir.mkdir(parents=True, exist_ok=True)
    setup_plot_style()
    return [
        plot_score_histogram(data, fig_dir),
        plot_class_average_bar(data, fig_dir),
        plot_score_boxplot(data, fig_dir),
        plot_grade_pie(data, fig_dir),
    ]


# --------------------------------------------------------------------------
# 5. 保存结果
# --------------------------------------------------------------------------
def save_csv(frame: pd.DataFrame, csv_path: str | Path) -> Path:
    """把 DataFrame 保存为 CSV（utf-8-sig，方便 Excel 直接打开）。"""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return csv_path


# --------------------------------------------------------------------------
# 6. 主流程
# --------------------------------------------------------------------------
def print_report(report: dict) -> None:
    print(f"  原始记录数      : {report['原始记录数']}")
    print(f"  删除重复记录    : {report['重复记录删除']} 条")
    print("  识别到的缺失值  : " + "，".join(f"{k} {v} 个" for k, v in report["缺失值"].items()))
    print("  识别到的异常值  : " + "，".join(f"{k} {v} 个" for k, v in report["异常值"].items()))
    print("  中位数填补      : " + "，".join(f"{k} {v} 个" for k, v in report["填补值"].items()))


def run(
    input_csv: str | Path,
    output_csv: str | Path,
    stats_csv: str | Path,
    fig_dir: str | Path,
) -> None:
    print(LINE)
    print("Python 数据处理与可视化（实验 2）")
    print(LINE)

    print(f"\n[1/5] 读取 CSV 文件：{input_csv}")
    raw = load_data(input_csv)
    print(f"  读取成功，共 {len(raw)} 条记录，字段：{', '.join(raw.columns)}")

    print("\n[2/5] 处理缺失值与异常数据")
    clean, report = clean_data(raw)
    print_report(report)
    print(f"  清洗后记录数    : {len(clean)}")

    print("\n[3/5] 统计指标")
    stats = compute_statistics(clean, NUMERIC_COLS + ["平均分"])
    print(stats.to_string(index=False))

    print("\n[4/5] 绘制统计图表")
    figures = plot_all(clean, Path(fig_dir))
    for path in figures:
        print(f"  已保存图片：{path}")

    print("\n[5/5] 保存处理结果")
    saved_data = save_csv(clean, output_csv)
    saved_stats = save_csv(stats, stats_csv)
    print(f"  处理后数据：{saved_data}")
    print(f"  统计指标表：{saved_stats}")

    print("\n处理后的数据预览（前 5 条）：")
    print(clean.head(5).to_string(index=False))
    print(LINE)
    print("实验 2 运行完成。")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="学生成绩数据清洗、统计与可视化")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="原始 CSV 路径")
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT), help="处理后 CSV 的输出路径")
    parser.add_argument("--stats-csv", default=str(DEFAULT_STATS), help="统计指标 CSV 的输出路径")
    parser.add_argument("--fig-dir", default=str(DEFAULT_FIG_DIR), help="图表输出目录")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run(args.input, args.output_csv, args.stats_csv, args.fig_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
