@echo off
chcp 65001 >nul
echo ================================================
echo  LeNet-5 Shape32-5 一键复现实验
echo ================================================
if exist .venv\Scripts\activate (
  call .venv\Scripts\activate
) else (
  echo 未发现 .venv 虚拟环境。建议先双击 install_windows.bat。
)
python src\check_environment.py
python src\generate_dataset.py
python src\train_lenet.py --epochs 30 --batch_size 64 --lr 0.001
python src\evaluate_baseline_svm.py
python src\visualize_results.py
python src\visualize_enhanced.py
echo ================================================
echo  全部运行完成。请检查 outputs 和 figures 文件夹。
echo ================================================
pause
