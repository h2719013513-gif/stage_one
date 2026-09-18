@echo off
chcp 936 >nul
set MPLCONFIGDIR=E:\conda_cache\matplotlib
cd /d "%~dp0"
echo ============================================================
echo  实验 1：深度学习开发环境检测
echo  解释器：E:\conda_envs\py\python.exe
echo ============================================================
echo.
E:\conda_envs\py\python.exe check_env.py
echo.
echo ------------------------------------------------------------
echo  运行结束，按任意键关闭本窗口。
echo ------------------------------------------------------------
pause >nul