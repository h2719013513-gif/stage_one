@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set MPLCONFIGDIR=E:\conda_cache\matplotlib
cd /d "%~dp0"
echo ============================================================
echo  CIFAR-10 图像分类：数据准备 - 训练 - 评估 - 预测 - 可视化
echo  解释器：E:\conda_envs\py\python.exe
echo ============================================================
echo.
echo [1/5] 数据准备（解压、统计类别分布、划分数据集）
E:\conda_envs\py\python.exe dataset.py
if errorlevel 1 goto error
echo.
echo [2/5] 训练模型（约 4 分钟，结果保存到 E:\cifar10\runs）
E:\conda_envs\py\python.exe train.py
if errorlevel 1 goto error
echo.
echo [3/5] 在官方测试集上评估（准确率 + 混淆矩阵）
E:\conda_envs\py\python.exe evaluate.py
if errorlevel 1 goto error
echo.
echo [4/5] 单张图片预测
E:\conda_envs\py\python.exe predict.py --index 0
echo.
echo [5/5] 可视化 16 张预测结果
E:\conda_envs\py\python.exe visualize.py --num 16
echo.
echo ------------------------------------------------------------
echo  全部完成！图表在 E:\cifar10\runs\figures，日志在 E:\cifar10\runs\logs
echo  按任意键关闭本窗口。
echo ------------------------------------------------------------
pause >nul
exit /b 0

:error
echo.
echo 出错了：请看上面的错误信息，常见问题见 README.md 第九节。
pause >nul
exit /b 1