# 深度学习实验一阶段：环境搭建 · 数据处理 · Git 管理

本仓库是《深度学习》课程第一阶段（实验 1～3）的提交成果，包含环境检测程序、
数据处理与可视化程序、数据集、图表结果以及环境配置说明。

**仓库地址：** https://github.com/h2719013513-gif/stage_one

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

### 0. 本机实测环境（2026-09-18 验证通过）

> 本机 C 盘空间紧张，因此 **conda 环境、Git、下载缓存全部放在 E 盘**，仅项目代码放在工作目录。

| 项目 | 实测结果 |
| --- | --- |
| Anaconda / conda | 26.5.3，安装于 `E:\anaconda3` |
| conda 环境 `py` | `E:\conda_envs\py`，路径已加入 `envs_dirs`，可直接 `conda activate py` |
| Python | 3.11.16 |
| PyTorch | 2.14.0+cu126（编译 CUDA 12.6） |
| CUDA 是否可用 | **是**，NVIDIA GeForce RTX 3050 Laptop GPU（4.0 GB 显存），cuDNN 91002 |
| numpy / pandas / matplotlib | 2.4.6 / 3.0.6 / 3.11.2 |
| Git | MinGit 2.55.0.windows.5，安装于 `E:\tools\MinGit`，已加入用户 PATH |
| 显卡驱动 | 591.74 |

### 1. 创建 conda 环境

实验使用 **conda 环境 `py`**（Python 3.11）。本机环境位置：`E:\conda_envs\py`。

```bash
# 方式一：安装到 conda 默认环境目录
conda create -n py python=3.11 pip -y

# 方式二：安装到指定目录（本机采用的方式，E 盘空间充足）
conda create -p E:\conda_envs\py python=3.11 pip -y

# 让 conda 能按名字找到该目录下的环境
conda config --add envs_dirs E:\conda_envs

conda activate py
```

> 注意：本机 conda 的 libmamba 求解器取索引异常（会误报 python/pip 不存在），
> 创建环境时需追加 `--solver=classic`。
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

如果官方源速度太慢（本机实测仅 1 MB/s 左右），可以从国内镜像下载 wheel 后再本地安装：

```bash
# 上海交大镜像实测约 6~11 MB/s，约 2.4 GB
curl.exe -L -o torch-2.14.0+cu126-cp311-cp311-win_amd64.whl ^
  "https://mirror.sjtu.edu.cn/pytorch-wheels/cu126/torch-2.14.0%2Bcu126-cp311-cp311-win_amd64.whl"

pip install --no-cache-dir torch-2.14.0+cu126-cp311-cp311-win_amd64.whl
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

本机实测输出（节选）：

```
Python 版本   : 3.11.16
PyTorch 版本  : 2.14.0+cu126
编译所用 CUDA : 12.6
是否支持 CUDA : 是
  显卡 0       : NVIDIA GeForce RTX 3050 Laptop GPU（显存 4.0 GB）
cuDNN 版本    : 91002
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

### 1. Git 安装

本机原本没有安装 Git，且当前账号不在管理员组（无法写入 `C:\Program Files`），
因此部署官方 **MinGit 2.55.0.windows.5** 便携版到 E 盘并加入用户 PATH：

```powershell
# 下载（国内镜像）
curl.exe -L -o MinGit.zip "https://registry.npmmirror.com/-/binary/git-for-windows/v2.55.0.windows.5/MinGit-2.55.0.5-64-bit.zip"

# 解压到 E 盘
Expand-Archive MinGit.zip -DestinationPath E:\tools\MinGit

# 加入用户 PATH（新开终端后即可直接使用 git 命令）
[Environment]::SetEnvironmentVariable('Path',
    [Environment]::GetEnvironmentVariable('Path','User') + ';E:\tools\MinGit\cmd', 'User')

git --version   # git version 2.55.0.windows.5
```

### 2. 提交记录（共 9 次有意义的提交，满足"至少三次"的要求）

| 序号 | 提交内容 |
| --- | --- |
| 1 | 初始化仓库：添加 `.gitignore` 与依赖清单 `requirements.txt` |
| 2 | 实验 1：新增环境检测程序 `check_env.py` |
| 3 | 实验 2：新增数据生成脚本与原始数据集（122 条，含脏数据） |
| 4 | 实验 2：新增数据处理与可视化程序 `data_analysis.py` |
| 5 | 实验 2：提交处理后的数据、统计指标表与四张图表 |
| 6 | 文档：补充 README 运行方法、环境配置与 CPU/GPU 区别 |
| 7 | 文档：补充本机实测环境信息 |
| 8 | 实验 2：优化等级分布饼图（过滤人数为 0 的等级，避免标签重叠） |
| 9 | 文档：补充远程仓库地址与提交者信息 |

查看提交记录：

```bash
git log --oneline --graph
git log --stat
```

> 提交者信息：`贺斌杰 <h2719013513-gif@users.noreply.github.com>`（GitHub 提供的 noreply 邮箱）。
> 如需更换邮箱：`git config user.email "你的邮箱"`，已有提交可用
> `git rebase --root --exec "git commit --amend --reset-author --no-edit"` 重写作者信息。

### 3. 关联远程仓库并推送

在 GitHub / Gitee 上新建空仓库后：

```bash
git remote add origin <仓库地址>
git branch -M main
git push -u origin main
```

首次推送可能需要登录：GitHub 使用 **Personal Access Token**（Settings → Developer settings → Tokens），
Gitee 可使用账号密码或私人令牌。

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
