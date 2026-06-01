@echo off
chcp 65001 >nul
echo ================================================
echo  创建虚拟环境并安装本项目依赖
echo ================================================
python --version
if errorlevel 1 (
  echo 未检测到 python，请先安装 Python，并勾选 Add python.exe to PATH。
  pause
  exit /b 1
)
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python src\check_environment.py
echo ================================================
echo  安装流程结束。如果上方都显示 OK，就可以运行 run_all_windows.bat。
echo ================================================
pause
