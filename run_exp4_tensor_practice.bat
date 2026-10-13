@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set MPLCONFIGDIR=E:\conda_cache\matplotlib
cd /d "%~dp0"
echo ============================================================
echo  实验 4：PyTorch 张量与自动求导
echo  解释器：E:\conda_envs\py\python.exe
echo ============================================================
echo.
E:\conda_envs\py\python.exe tensor_practice.py
echo.
echo ------------------------------------------------------------
echo  结果直接打印在窗口里。
echo  运行结束，按任意键关闭本窗口。
echo ------------------------------------------------------------
pause >nul