# CIFAR-10 图像分类（简单 CNN / PyTorch）

用 PyTorch 训练一个简单的卷积神经网络，对 CIFAR-10 的 10 类图片做分类，
完成从数据准备、模型训练、评估到单张预测的完整流程。

数据集官网：https://www.cs.toronto.edu/~kriz/cifar.html

## 一、目录结构

```
cifar10_classification/
├── config.py           路径与超参数（数据默认放 E 盘，可用环境变量覆盖）
├── dataset.py          数据准备：解析 CIFAR-10、统计类别分布、划分训练/验证/测试、预处理
├── model.py            SimpleCNN 模型定义
├── train.py            训练 + 验证循环，保存最佳模型，画 Loss/准确率曲线
├── evaluate.py         测试集评估：准确率、混淆矩阵、每类精确率/召回率/F1
├── predict.py          单张图片预测（测试集图片 / 自己的图片）
├── visualize.py        可视化多张预测结果（绿=正确，红=错误）
├── run_cifar10_all.bat 一键跑完整流程
└── outputs/            本次运行的图表与日志（提交用）
    ├── figures/        数据分布、训练曲线、混淆矩阵、预测可视化
    └── logs/           训练日志、评估报告、混淆矩阵原始数据
```

数据集（约 163MB）和模型权重放在 **E 盘**，不进 Git 仓库：

| 内容 | 路径 |
| --- | --- |
| 数据集压缩包 | `E:\cifar10\data\cifar-10-python.tar.gz` |
| 解压后的数据 | `E:\cifar10\data\cifar-10-batches-py` |
| 模型权重、图表、日志 | `E:\cifar10\runs` |

## 二、环境

使用本机已有的 conda 环境 `py`（Python 3.11.16）：

```bat
conda activate py
```

依赖：`torch 2.14.0+cu126`、`torchvision 0.29.1`、`numpy`、`matplotlib`、`pillow`。
安装 torchvision（与 torch 版本配套）：

```bat
pip install torchvision --index-url https://download.pytorch.org/whl/cu126
```

## 三、数据准备

### 1. 获取数据

官方地址 `www.cs.toronto.edu` 在国内经常连不上，实测可用的镜像：

```bat
curl.exe -L -o E:\cifar10\data\cifar-10-python.tar.gz https://dataset.bj.bcebos.com/cifar/cifar-10-python.tar.gz
```

下载后脚本会自动校验 MD5（官方值 `c58f30108f718f92721af3b95e74349a`）并解压。

本项目**没有使用 torchvision 的自动下载**，而是自己解析官方 pickle 格式，
这样只要压缩包在本地，整个流程就能离线跑通。

### 2. 划分数据集

| 集合 | 来源 | 数量 |
| --- | --- | --- |
| 训练集 | 官方训练集 50000 张中按类别分层抽样 | 20000 张（每类 2000） |
| 验证集 | 官方训练集剩下的部分 | 10000 张（每类 1000） |
| 测试集 | 官方测试集，完整使用 | 10000 张 |

划分规则：

- 三者在样本层面**完全不重叠**（训练集与验证集不共享任何一张图片，测试集来自官方测试集）；
- 划分结果保存在 `splits.json`，训练/评估/可视化共用同一份划分，保证可复现；
- 预处理只对训练集做数据增强（随机裁剪 + 随机水平翻转），验证集和测试集只做归一化；
- 归一化用的均值方差是 CIFAR-10 官方公布的**训练集**统计值，
  **没有使用验证集或测试集的信息**。

### 3. 类别分布

官方数据严格均衡：训练集每类 5000 张、测试集每类 1000 张。
分布图见 `outputs/figures/数据分布_官方训练集.png` 与 `数据分布_官方测试集.png`。

## 四、模型

```
输入 3×32×32
  → 卷积块1：Conv(3→32) + BN + ReLU，Conv(32→32) + BN + ReLU，最大池化  → 32×16×16
  → 卷积块2：Conv(32→64) + BN + ReLU，Conv(64→64) + BN + ReLU，最大池化  → 64×8×8
  → 展平 → 全连接 256 + ReLU + Dropout(0.3) → 全连接 10
```

可训练参数 **1,117,162** 个，属于简单 CNN，不需要复杂网络结构。

## 五、训练

超参数：`AdamW`，学习率 1e-3，权重衰减 5e-4，余弦退火调度，batch size 128，训练 25 轮。

实测（NVIDIA RTX 3050 Laptop GPU）：

| 项目 | 结果 |
| --- | --- |
| 每轮耗时 | 约 9.5 秒 |
| 总用时 | 3.9 分钟 |
| 最佳验证准确率 | **78.36%**（第 24 轮） |

训练曲线见 `outputs/figures/训练曲线_Loss与准确率.png`。

> 曲线里验证集准确率略高于训练集，这不是异常：训练集做了随机裁剪和翻转，
> 相当于一直在做"更难的题"；验证集是干净图片且不增强，所以指标反而更好。
> 如果训练集准确率远高于验证集，那才是过拟合。

## 六、评估结果（官方测试集 10000 张）

总体准确率 **77.91%**，宏平均 F1 **77.82%**（两者接近，说明各类别表现比较均衡）。

| 类别 | 精确率 | 召回率 | F1 |
| --- | --- | --- | --- |
| 飞机 airplane | 77.64% | 80.90% | 79.24% |
| 汽车 automobile | 88.22% | 90.60% | 89.39% |
| 鸟 bird | 67.13% | 63.30% | 65.16% |
| 猫 cat | 62.10% | 58.00% | 59.98% |
| 鹿 deer | 74.41% | 75.60% | 75.00% |
| 狗 dog | 67.28% | 69.70% | 68.47% |
| 青蛙 frog | 80.48% | 83.70% | 82.06% |
| 马 horse | 82.99% | 81.50% | 82.24% |
| 船 ship | 89.49% | 87.70% | 88.59% |
| 卡车 truck | 88.10% | 88.10% | 88.10% |

结果分析：

1. **识别最好的是人造物体**：汽车 90.6%、船 87.7%、卡车 88.1%。它们轮廓规整、颜色单一，特征好学。
2. **识别最差的是动物**：猫 58.0%、鸟 63.3%、狗 69.7%。它们毛发纹理复杂、姿态多变、背景杂乱。
3. **最典型的错误是猫和狗互相认错**（猫→狗 173 张，狗→猫 152 张），
   其次是鸟→鹿（91 张）、鸟→飞机（71 张）。这四类在 32×32 分辨率下确实很难分。
4. **没有严重过拟合**：验证集 Loss 一路下降到 0.626，验证准确率还在缓慢提升。

进一步提升的方向：用满官方 50000 张训练数据、加更多增强（Cutout / Mixup）、
训练更多轮、换更深的网络或加上残差连接。

混淆矩阵见 `outputs/figures/测试集混淆矩阵.png`，逐类指标见 `outputs/logs/评估报告.txt`。

## 七、单张预测与结果可视化

```bat
python predict.py --index 0        :: 预测测试集第 0 张（会显示真实标签对照）
python predict.py --random         :: 随机抽一张测试集图片
python predict.py --image D:\cat.png   :: 预测自己的图片
python visualize.py --num 16              :: 可视化 16 张随机测试图片
python visualize.py --num 12 --mode wrong :: 只看预测错的样本
```

实测输出示例：

```
图片来源：测试集第 0 张
真实标签：猫（cat）
Top-3 预测：
  1. 猫   cat          86.42%
  2. 狗   dog           7.22%
  3. 青蛙  frog         5.56%
结论：预测正确（预测 猫）
```

可视化图：`outputs/figures/预测结果可视化_random.png`（16 张，标题绿色为预测正确、红色为预测错误）、
`预测结果可视化_wrong.png`（12 张全部预测错误的样本）、`单张预测_0.png`。

## 八、怎么跑（三种方式）

### 方式 1：一键脚本（最省事）

双击 `run_cifar10_all.bat`，它会按顺序执行：数据准备 → 训练 → 评估 → 单张预测 → 可视化。

### 方式 2：命令行分步执行

```bat
conda activate py
cd /d C:\Users\贺斌杰\Documents\ChatGPT\stage_one\cifar10_classification
python dataset.py       :: 1. 数据准备：解压、统计类别分布、划分数据集
python train.py         :: 2. 训练并保存最佳模型
python evaluate.py      :: 3. 测试集评估：准确率 + 混淆矩阵
python predict.py --index 0        :: 4. 单张预测
python visualize.py --num 16       :: 5. 可视化 16 张预测结果
```

不激活 conda 也可以，直接用全路径 `E:\conda_envs\py\python.exe`。

### 方式 3：PyCharm

打开对应 `.py` 文件，按 `Ctrl + Shift + F10` 运行。解释器选择 `Python 3.11 (py)`。

## 九、常见问题

1. **下载数据集报错或特别慢**：官方源在国内基本连不上，用百度云镜像
   `https://dataset.bj.bcebos.com/cifar/cifar-10-python.tar.gz`（实测 3 MB/s）。
2. **`FileNotFoundError: 找不到数据集压缩包`**：把 `cifar-10-python.tar.gz` 放到
   `E:\cifar10\data\` 下再运行 `dataset.py`，脚本会自动解压。
3. **想换数据/产物目录**：设置环境变量 `CIFAR10_DATA_DIR`、`CIFAR10_RUN_DIR` 即可；
   如果 E 盘不存在，代码会自动回退到项目目录下的 `data` / `runs`。
4. **显存不够**：默认 batch size 128 在 4GB 显存上没问题；报 OOM 就调小
   `--batch-size 64`。
5. **没有 GPU**：代码会自动用 CPU，只是慢一些（本机 GPU 每轮 9.5 秒，CPU 大约 1～2 分钟）。
6. **想重新划分数据集**：删掉 `E:\cifar10\runs\splits.json` 再运行 `dataset.py`。
7. **训练中断**：重新运行 `train.py` 会从头训练（模型文件会被覆盖）。
