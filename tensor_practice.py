# -*- coding: utf-8 -*-
"""实验 4：PyTorch 张量与自动求导

运行方式：
    python tensor_practice.py

内容：
    1. 创建不同维度的张量
    2. 张量加法、乘法（逐元素）与矩阵乘法
    3. 张量维度变换
    4. 构造 y = w*x + b 并用 PyTorch 自动求导计算参数梯度
    5. 修改参数，观察函数输出变化
"""

from __future__ import annotations

import torch

LINE = "=" * 72


def section(number: str, text: str) -> None:
    print()
    print(LINE)
    print(f"{number} {text}")
    print(LINE)


def info(name: str, tensor: torch.Tensor) -> None:
    """打印张量的形状、维度数、元素个数等关键信息。"""
    print(
        f"{name:<16} 形状={str(tuple(tensor.shape)):<12} "
        f"维度数={tensor.ndim:<3} 元素个数={tensor.numel():<6} 数据类型={tensor.dtype}"
    )


# --------------------------------------------------------------------------
# 1. 创建不同维度的张量
# --------------------------------------------------------------------------
def demo_create_tensors() -> None:
    section("1.", "创建不同维度的张量")

    scalar = torch.tensor(3.14)                                    # 0 维：标量
    vector = torch.tensor([1.0, 2.0, 3.0])                         # 1 维：向量
    matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])                # 2 维：矩阵
    cube = torch.rand(2, 3, 4)                                     # 3 维：例如 2 张 3x4 的图
    batch = torch.zeros(8, 3, 32, 32)                              # 4 维：一批 32x32 的彩色图片

    print("-- 不同维度的张量 --")
    info("0 维（标量）", scalar)
    info("1 维（向量）", vector)
    info("2 维（矩阵）", matrix)
    info("3 维（张量）", cube)
    info("4 维（一批图片）", batch)

    print()
    print("-- 常用的创建函数 --")
    info("torch.arange(12)", torch.arange(12))
    info("torch.linspace", torch.linspace(0, 1, 5))
    info("torch.zeros", torch.zeros(2, 3))
    info("torch.ones", torch.ones(2, 3))
    info("torch.eye(3)", torch.eye(3))
    info("torch.randn", torch.randn(2, 3))
    print(f"{'指定数据类型':<16} torch.ones(2,2,dtype=torch.int64) -> {torch.ones(2, 2, dtype=torch.int64).dtype}")


# --------------------------------------------------------------------------
# 2. 加法、乘法、矩阵乘法
# --------------------------------------------------------------------------
def demo_basic_ops() -> None:
    section("2.", "张量加法、乘法与矩阵乘法")

    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

    print(f"a =\n{a}")
    print(f"\nb =\n{b}")

    print("\n-- 加法：对应位置相加 --")
    print(f"a + b =\n{a + b}")
    print(f"torch.add(a, b) 结果相同：{torch.equal(a + b, torch.add(a, b))}")

    print("\n-- 乘法：逐元素相乘（对应位置相乘）--")
    print(f"a * b =\n{a * b}")
    print(f"a * b 和矩阵乘法不一样，注意区分")

    print("\n-- 矩阵乘法：行乘以列再求和 --")
    print(f"a @ b =\n{a @ b}")
    print(f"torch.mm(a, b) 结果相同：{torch.equal(a @ b, torch.mm(a, b))}")
    print(f"a @ b 的第一个元素 = 1*5 + 2*7 = {1*5 + 2*7}")

    print("\n-- 广播机制：形状不同的张量也能运算 --")
    row = torch.tensor([10.0, 20.0])
    print(f"a + [10, 20] =\n{a + row}   （每一行都加上 [10, 20]）")
    print(f"a * 2 =\n{a * 2}   （标量与每个元素相乘）")

    print("\n-- 统计运算 --")
    print(f"a.sum()={a.sum().item()}  a.mean()={a.mean().item()}  a.max()={a.max().item()}  a.min()={a.min().item()}")


# --------------------------------------------------------------------------
# 3. 维度变换
# --------------------------------------------------------------------------
def demo_reshape() -> None:
    section("3.", "张量维度变换")

    x = torch.arange(12, dtype=torch.float32)
    print(f"原始 x：形状 {tuple(x.shape)}  内容 {x.tolist()}")

    print("\n-- reshape / view：改变形状，元素个数不变 --")
    x_3x4 = x.reshape(3, 4)
    print(f"x.reshape(3, 4)：{tuple(x_3x4.shape)}")
    print(x_3x4)
    print(f"x.view(2, 6)：{tuple(x.view(2, 6).shape)}")
    print("说明：view 要求内存连续，reshape 更宽松，日常优先用 reshape")

    print("\n-- unsqueeze / squeeze：增加或去掉长度为 1 的维度 --")
    col = x.unsqueeze(1)
    print(f"x.unsqueeze(1)：{tuple(x.shape)} -> {tuple(col.shape)}   （12 个元素变成 12 行 1 列）")
    print(f"col.squeeze(1)：{tuple(col.shape)} -> {tuple(col.squeeze(1).shape)}")

    print("\n-- transpose / permute：交换维度顺序 --")
    mat = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    print(f"mat：{tuple(mat.shape)}\n{mat}")
    print(f"mat.transpose(0, 1)：{tuple(mat.transpose(0, 1).shape)}")
    print(mat.transpose(0, 1))

    t3 = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    print(f"\n三维张量：{tuple(t3.shape)}   permute(2,0,1) 之后：{tuple(t3.permute(2, 0, 1).shape)}")

    print("\n-- flatten / cat / stack --")
    print(f"t3.flatten()：{tuple(t3.shape)} -> {tuple(t3.flatten().shape)}")
    print(f"torch.cat([mat, mat], dim=0)：{tuple(torch.cat([mat, mat], dim=0).shape)}   （拼接，维度不变）")
    print(f"torch.stack([mat, mat], dim=0)：{tuple(torch.stack([mat, mat], dim=0).shape)}   （堆叠，多一个维度）")


# --------------------------------------------------------------------------
# 4. 自动求导：y = w*x + b
# --------------------------------------------------------------------------
def demo_autograd() -> tuple[torch.Tensor, torch.Tensor]:
    section("4.", "构造 y = w*x + b，用自动求导计算参数梯度")

    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    w = torch.tensor(2.0, requires_grad=True)   # requires_grad=True 表示需要求导
    b = torch.tensor(1.0, requires_grad=True)

    print(f"输入 x = {x.tolist()}")
    print(f"参数 w = {w.item()}（requires_grad={w.requires_grad}）")
    print(f"参数 b = {b.item()}（requires_grad={b.requires_grad}）")

    y = w * x + b
    print(f"\ny = w*x + b = {y.tolist()}")
    print(f"y 的梯度函数：{y.grad_fn}   （说明 y 记录在计算图里，可以反向求导）")

    loss = y.sum()          # 为了求导，把 y 变成一个标量
    loss.backward()         # 反向传播，自动算出 w 和 b 的梯度

    print(f"\nloss = sum(y) = {loss.item()}")
    print(f"反向传播后：w.grad = {w.grad.item()}，b.grad = {b.grad.item()}")
    print("手算验证：")
    print(f"  d(loss)/dw = x1 + x2 + x3 + x4 = {x.sum().item()}  -> 与 w.grad 一致")
    print(f"  d(loss)/db = 1 + 1 + 1 + 1 = {x.numel()}  -> 与 b.grad 一致")

    return w, b


# --------------------------------------------------------------------------
# 5. 修改参数，观察输出变化
# --------------------------------------------------------------------------
def demo_change_params() -> None:
    section("5.", "修改参数，观察函数输出变化")

    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    target = torch.tensor([3.0, 5.0, 7.0, 9.0])   # 真实规律：y = 2x + 1

    print("-- 第一组：手动换不同的 (w, b)，看输出怎么变 --")
    print(f"{'w':>6}{'b':>6}   y = w*x + b                        y 的平均值")
    for w_value, b_value in [(1.0, 0.0), (2.0, 1.0), (3.0, 2.0), (4.0, 3.0)]:
        y_value = w_value * x + b_value
        print(f"{w_value:>6.1f}{b_value:>6.1f}   {str(y_value.tolist()):<35}{y_value.mean().item():>8.2f}")
    print("规律：w 越大曲线越陡（x 前面的系数），b 越大整体上移（截距）。")

    print("\n-- 第二组：用梯度反向修改参数，看误差怎么变 --")
    w = torch.tensor(0.5, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    lr = 0.05

    print(f"{'轮次':>4}{'w':>10}{'b':>10}{'Loss（更新后）':>16}")

    def current_loss() -> float:
        with torch.no_grad():
            return float(((w * x + b - target) ** 2).mean().item())

    print(f"{0:>4}{w.item():>10.4f}{b.item():>10.4f}{current_loss():>16.4f}")

    for step in range(1, 6):
        y_pred = w * x + b
        loss = ((y_pred - target) ** 2).mean()

        loss.backward()                       # 求梯度
        with torch.no_grad():                 # 修改参数时不需要再记录计算图
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()

        print(f"{step:>4}{w.item():>10.4f}{b.item():>10.4f}{current_loss():>16.4f}")

    with torch.no_grad():
        y_final = w * x + b
    print(f"\n训练 5 步后：w = {w.item():.4f}，b = {b.item():.4f}")
    print(f"预测值 {[round(v, 3) for v in y_final.tolist()]}")
    print(f"真实值 {target.tolist()}")
    print("结论：沿着梯度的反方向修改参数，预测值会越来越接近真实值，Loss 越来越小。")


def main() -> int:
    torch.manual_seed(2026)
    print(LINE)
    print("PyTorch 张量与自动求导（实验 4）")
    print(LINE)
    print(f"PyTorch 版本：{torch.__version__}    CUDA 可用：{torch.cuda.is_available()}")

    demo_create_tensors()
    demo_basic_ops()
    demo_reshape()
    demo_autograd()
    demo_change_params()

    print()
    print(LINE)
    print("实验 4 运行完成。")
    print("要点回顾：")
    print("  1. 张量的维度数(ndim)就是形状(shape)的长度，numel 是元素总个数；")
    print("  2. a*b 是逐元素相乘，a@b 才是矩阵乘法，二者完全不同；")
    print("  3. reshape/view/unsqueeze/squeeze/transpose/permute 只改变形状，不改变数据；")
    print("  4. requires_grad=True 的张量会被记录在计算图里，调用 backward() 就能自动求梯度；")
    print("  5. 参数更新必须放在 torch.no_grad() 里，否则会被记进计算图。")
    print(LINE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
