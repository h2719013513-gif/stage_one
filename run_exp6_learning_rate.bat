@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set MPLCONFIGDIR=E:\conda_cache\matplotlib
cd /d "%~dp0"
echo ============================================================
echo  实验 6：不同学习率对比分析
echo  解释器：E:\conda_envs\py\python.exe
echo ============================================================
echo.
E:\conda_envs\py\python.exe learning_rate_analysis.py
echo.
echo ------------------------------------------------------------
echo  两组对比图已保存到 output\figures 目录。
echo  运行结束，按任意键关闭本窗口。
echo ------------------------------------------------------------
pause >nul