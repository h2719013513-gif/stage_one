@echo off
chcp 65001 >nul
set MPLCONFIGDIR=E:\conda_cache\matplotlib
cd /d "%~dp0"
echo ============================================================
echo  实验 2：Python 数据处理与可视化
echo  解释器：E:\conda_envs\py\python.exe
echo ============================================================
echo.
E:\conda_envs\py\python.exe data_analysis.py
echo.
echo ------------------------------------------------------------
echo  处理结果：data\processed\students_scores_clean.csv
echo             output\statistics_summary.csv
echo             output\figures\  (4 张图)
echo  运行结束，按任意键关闭本窗口。
echo ------------------------------------------------------------
pause >nul