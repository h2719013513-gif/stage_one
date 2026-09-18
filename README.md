# 深度学习实验一阶段：环境搭建 · 数据处理 · Git 管理

本仓库是《深度学习》课程第一阶段（实验 1～3）的提交成果，包含环境检测程序、
数据处理与可视化程序、数据集、图表结果以及环境配置说明。

## 一、目录结构

```
stage_one/
├── check_env.py                     实验 1：深度学习开发环境检测程序
├── data_analysis.py                 实验 2：数据读取 / 清洗 / 统计 / 可视化
├── data/
│   ├── raw/students_scores_raw.csv        原始数据（含缺失值、异常值、重复记录）
│   └── processed/students_scores_clean.csv 清洗后的数据
├── output/
│   ├── figures/                            统计图表（PNG）
│   └── statistics_summary.csv              平均值 / 最大值 / 最小值 / 标准差等
├── scripts/generate_dataset.py      原始数据生成脚本（固定随机种子，可复现）
├── requirements.txt                 依赖清单
└── README.md                        本文件
```

## 二、环境配置说明

### 1. 创建 conda 环境

实验使用 **conda 环境 `py`**（Python 3.11）。本机环境位置：`E:\conda_envs\py`。

```bash
# 方式一：安装到 conda 默认环境目录
conda create -n py python=3.11 pip -y

# 方式二：安装到指定目录（本机采用的方式，E 盘空间充足）
conda create -p E:\conda_envs\py python=3.11 pip -y

conda activate py
```

> 注意：本机 conda 的 libmamba 求解器取索引异常，创建环境时需追加 `--solver=classic`。
> 国内网络下载慢时可加清华镜像：
> `-c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ --override-channels`

### 2. 安装依赖

```bash
conda activate py

# 数据处理与绘图
pip install numpy pandas matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple

# PyTorch（CUDA 12.6 版，适配本机 NVIDIA GeForce RTX 3050 Laptop GPU）
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

如果只是 CPU 环境（没有 NVIDIA 显卡），用 CPU 版即可：

```bash
pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple
```

也可以直接安装 `requirements.txt`：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. 验证环境

```bash
conda activate py
python check_env.py
```

## 三、实验 1：环境检测程序 `check_env.py`

运行：

```bash
python check_env.py
```

程序输出内容：

1. Python 版本、解释器路径、操作系统；
2. PyTorch 版本、编译所用的 CUDA 版本；
3. 是否支持 CUDA（显卡型号、显存、cuDNN 版本）；
4. 一个 3×3 的随机张量，并打印形状、数据类型、所在设备、最大值/最小值/平均值；
5. CPU 与 GPU 的区别对照表（验收问答要点）。

## 四、实验 2：数据处理与可视化 `data_analysis.py`

### 1. 数据集说明

数据集为程序模拟生成的学生成绩数据（`data/raw/students_scores_raw.csv`），共 **122 条记录**
（其中 1 条为重复登记），字段包括：

| 字段 | 说明 |
| --- | --- |
| 学号 | 主键，如 20210001 |
| 姓名 / 班级 / 性别 | 学生基本信息，共 4 个班级 |
| 数学 / 英语 / Python | 三门课程成绩，0～100 分 |
| 出勤率 | 0～1 之间的小数 |

原始数据中故意混入了以下"脏数据"，用于演示数据清洗：

- **缺失值**：空白单元格、`缺考`、`无` 等文本；
- **异常值**：`-1` 分、`200` 分、`999` 分、大于 1 的出勤率；
- **重复记录**：同一条学生记录被登记两次。

### 2. 运行

```bash
python data_analysis.py
```

也可以自定义输入输出路径：

```bash
python data_analysis.py \
    --input  data/raw/students_scores_raw.csv \
    --output-csv data/processed/students_scores_clean.csv \
    --stats-csv  output/statistics_summary.csv \
    --fig-dir    output/figures
```

### 3. 处理流程（全部封装为函数，不依赖手工修改 CSV）

| 步骤 | 函数 | 说明 |
| --- | --- | --- |
| 读取 CSV | `load_data()` | 按 `utf-8-sig` 读取，兼容 Excel 中文 |
| 数据清洗 | `clean_data()` | 去重 → 识别缺失值 → 剔除异常值 → 用**同班级同科目中位数**填补 → 生成平均分与等级 |
| 统计指标 | `compute_statistics()` | 记录数、平均值、最大值、最小值、标准差（样本）、中位数 |
| 绘制图表 | `plot_all()` | 直方图、分组柱状图、箱线图、饼图 |
| 保存结果 | `save_csv()` | 输出清洗后数据与统计指标表 |

### 4. 输出结果

| 文件 | 内容 |
| --- | --- |
| `data/processed/students_scores_clean.csv` | 清洗后的数据（含平均分、等级） |
| `output/statistics_summary.csv` | 各指标的平均值、最大值、最小值、标准差等 |
| `output/figures/fig1_成绩分布直方图.png` | 三门课程成绩分布直方图 |
| `output/figures/fig2_班级平均分柱状图.png` | 各班级各科平均分对比 |
| `output/figures/fig3_成绩箱线图.png` | 成绩离散程度与离群点 |
| `output/figures/fig4_等级分布饼图.png` | 成绩等级分布 |

## 五、实验 3：Git 项目管理

### 1. 初始化与提交

```bash
git init
git add .
git commit -m "提交实验一环境检测程序 check_env.py"
git commit -m "提交实验二数据处理程序 data_analysis.py 与数据集"
git commit -m "提交实验二统计图表与处理结果"
git commit -m "补充 README 环境配置说明"
```

### 2. 关联远程仓库并推送

在 GitHub / Gitee 上新建空仓库后：

```bash
git remote add origin <仓库地址>
git branch -M main
git push -u origin main
```

### 3. 查看提交记录

```bash
git log --oneline --graph
```

## 六、CPU 与 GPU 的区别（验收问答）

| 对比项 | CPU | GPU |
| --- | --- | --- |
| 设计目标 | 通用串行计算，逻辑判断能力强 | 大规模并行数值计算 |
| 核心数量 | 通常 4～16 个高性能核心 | 上千个简单计算核心 |
| 擅长任务 | 复杂分支、操作系统调度 | 矩阵乘法、卷积等可并行运算 |
| 深度学习 | 小模型、调试、推理演示 | 大模型训练，速度可快数十倍 |
| 存储 | 内存条，容量大、带宽相对低 | 显存，容量小、带宽高 |

一句话总结：CPU 像几个什么都会做的博士生，GPU 像几千个只会简单运算的小学生，
而深度学习的大量矩阵运算恰好适合"人海战术"。
