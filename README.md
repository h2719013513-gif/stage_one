# 深度学习实验：环境搭建 · 数据处理 · 张量与神经网络 · Git 管理

本仓库是《深度学习》课程的实验提交成果，包含三个阶段：

- **第一阶段（实验 1～3）**：环境检测程序、数据处理与可视化、Git 项目管理；
- **第二阶段（实验 4～6）**：PyTorch 张量与自动求导、从零实现线性回归训练、不同学习率的训练过程分析；
- **第三阶段（计算机视觉项目）**：基于 PyTorch 的 CIFAR-10 图像分类系统，见
  [cifar10_classification/](cifar10_classification/README.md)（简单 CNN、训练/验证/测试、混淆矩阵、单张预测、结果可视化）。

**仓库地址：** https://github.com/h2719013513-gif/stage_one

## 一、目录结构

```
stage_one/
├── cifar10_classification/          计算机视觉项目：CIFAR-10 图像分类（含独立 README）
├── check_env.py                     实验 1：深度学习开发环境检测程序
├── data_analysis.py                 实验 2：数据读取 / 清洗 / 统计 / 可视化
├── tensor_practice.py               实验 4：张量操作与自动求导练习
├── linear_regression.py             实验 5：线性回归完整训练（含 Loss 曲线与预测对比图）
├── learning_rate_analysis.py        实验 6：不同学习率的训练过程对比与分析
├── 实验6_学习率分析.md               实验 6 的分析报告
├── 实验步骤与常见问题.md             大白话版操作手册（怎么跑、出错怎么办）
├── 实验4-6_步骤与常见问题.md         实验 4～6 的操作手册与问题排查
├── run_exp1_env_check.bat           一键运行脚本（双击即可，下同）
├── run_exp2_data_analysis.bat
├── run_exp4_tensor_practice.bat
├── run_exp5_linear_regression.bat
├── run_exp6_learning_rate.bat
├── data/
│   ├── raw/students_scores_raw.csv        原始数据（含缺失值、异常值、重复记录）
│   └── processed/students_scores_clean.csv 清洗后的数据
├── output/
│   ├── figures/                            所有图表（PNG）
│   ├── 实验5_训练日志.csv                   实验 5 的每轮 Loss
│   ├── 实验6_不同学习率Loss数据.csv          实验 6 的曲线原始数据
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

## 六、实验 4：张量与自动求导 `tensor_practice.py`

运行方式：

```bash
python tensor_practice.py
```

代码用 5 个函数对应实验要求的 5 件事：

| 实验要求 | 对应函数 | 做了什么 |
| --- | --- | --- |
| 创建不同维度的 Tensor | `demo_create_tensors()` | 0 维标量、1 维向量、2 维矩阵、3 维张量、4 维（一批图片），并演示 `arange/linspace/zeros/ones/eye/randn` |
| 加法、乘法、矩阵乘法 | `demo_basic_ops()` | `a + b`、`a * b`（逐元素）、`a @ b`（矩阵乘法）、广播、统计运算 |
| Tensor 维度变换 | `demo_reshape()` | `reshape/view/unsqueeze/squeeze/transpose/permute/flatten/cat/stack`，每一步都打印形状变化 |
| y = wx + b 求梯度 | `demo_autograd()` | 用 `requires_grad=True` 建 w、b，`loss.backward()` 自动求导，并与手算结果对照 |
| 修改参数观察输出 | `demo_change_params()` | 先手动换不同 (w, b) 看输出怎么变；再沿着梯度反方向更新参数，观察 Loss 逐轮下降 |

自动求导的关键输出（与手算完全一致）：

```
反向传播后：w.grad = 10.0，b.grad = 4.0
手算验证：
  d(loss)/dw = x1 + x2 + x3 + x4 = 10.0  -> 与 w.grad 一致
  d(loss)/db = 1 + 1 + 1 + 1 = 4  -> 与 b.grad 一致
```

修改参数后 Loss 的变化（每轮更新后打印）：

```
  轮次         w         b       Loss（更新后）
   0    0.5000    0.0000         25.3750
   1    1.8750    0.4750          0.7209
   5    2.1383    0.5928          0.0277
```

## 七、实验 5：从零实现线性回归 `linear_regression.py`

运行方式：

```bash
python linear_regression.py                 # 默认学习率 0.1，训练 200 轮
python linear_regression.py --lr 0.05 --epochs 300
```

训练流程完全手写，没有用任何现成回归函数：

1. `generate_data()`：生成 200 个带噪声的样本，真实规律 `y = 2.5x + 1.2`，噪声标准差 0.8；
2. `build_model()`：`nn.Linear(1, 1)`，就是 y = wx + b；
3. `train()`：标准五步循环 —— 前向传播 → 算 Loss → `optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()`；
4. 损失函数用 `nn.MSELoss()`，优化器用 `torch.optim.SGD`；
5. 训练完输出 Loss 曲线和「预测 vs 真实」对比图。

实测结果（学习率 0.1、200 轮）：

| 项目 | 数值 |
| --- | --- |
| 初始 Loss | 19.316862 |
| 最终 Loss | 0.506008 |
| 真实参数 | w = 2.5，b = 1.2 |
| 学到的参数 | w = 2.5011，b = 1.3060 |

> Loss 从 19.32 降到 0.506，已经接近噪声带来的理论下限（0.8² ≈ 0.64），
> 说明模型确实学到了数据背后的真实规律，而不是死记数据点。

输出文件：

| 文件 | 内容 |
| --- | --- |
| `output/figures/实验5_Loss曲线.png` | Loss 随训练轮次变化的曲线（纵轴对数坐标） |
| `output/figures/实验5_预测结果对比.png` | 训练数据散点 + 真实规律直线 + 模型预测直线 |
| `output/实验5_训练日志.csv` | 每一轮的 epoch 与 loss，可在 Excel 里自己画图 |

## 八、实验 6：不同学习率的训练过程分析 `learning_rate_analysis.py`

运行方式：

```bash
python learning_rate_analysis.py
```

在实验 5 的代码基础上复用训练函数，**同一份数据、同一个初始参数，只改学习率**，各训练 60 轮：

实测结果：

| 学习率 | 实际轮次 | 最终 Loss | 学到的 w | 学到的 b | 结果 |
| --- | --- | --- | --- | --- | --- |
| 0.01 | 60 | 0.590334 | 2.426 | 1.048 | 正常，但还没完全收敛 |
| 0.1 | 60 | 0.506008 | 2.501 | 1.306 | 正常，收敛 |
| 0.3 | 60 | 0.506008 | 2.501 | 1.306 | 正常，收敛 |
| 0.5 | 13 | 1.600e+08 | 14337.832 | -588.447 | 第 13 轮发散 |
| 1.0 | 6 | 1.426e+08 | -34015.055 | 1399.942 | 第 6 轮发散 |

结论（详细分析见 [实验6_学习率分析.md](实验6_学习率分析.md)）：

1. **学习率太小**（0.01）：Loss 平缓下降，60 轮还没到最低点，收敛慢、费时间；
2. **学习率合适**（0.1）：前几轮陡降，第 40 轮后 Loss 稳定在 0.506008，参数基本等于真实值；
3. **学习率偏大**（0.5、1.0）：步子跨过最低点，参数被越推越远，Loss 指数级增长成 `inf/nan`，训练失败；
4. **判断收敛**：Loss 连续多轮几乎不变（变化 < 1e-6）+ 参数基本不动 + Loss 降到合理水平，三条同时满足即可。

输出文件：

| 文件 | 内容 |
| --- | --- |
| `output/figures/实验6_不同学习率Loss曲线.png` | 五个学习率的 Loss 曲线画在同一张图上（对数坐标） |
| `output/figures/实验6_不同学习率拟合对比.png` | 五个学习率学到的直线与真实数据的对比 |
| `output/实验6_不同学习率Loss数据.csv` | 每条曲线的原始数据 |

## 九、CPU 与 GPU 的区别（验收问答）

| 对比项 | CPU | GPU |
| --- | --- | --- |
| 设计目标 | 通用串行计算，逻辑判断能力强 | 大规模并行数值计算 |
| 核心数量 | 通常 4～16 个高性能核心 | 上千个简单计算核心 |
| 擅长任务 | 复杂分支、操作系统调度 | 矩阵乘法、卷积等可并行运算 |
| 深度学习 | 小模型、调试、推理演示 | 大模型训练，速度可快数十倍 |
| 存储 | 内存条，容量大、带宽相对低 | 显存，容量小、带宽高 |

一句话总结：CPU 像几个什么都会做的博士生，GPU 像几千个只会简单运算的小学生，
而深度学习的大量矩阵运算恰好适合"人海战术"。
