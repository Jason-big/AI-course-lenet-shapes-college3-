from __future__ import annotations
import sys
import importlib

packages = [
    ('numpy', 'NumPy'),
    ('PIL', 'Pillow'),
    ('matplotlib', 'Matplotlib'),
    ('sklearn', 'scikit-learn'),
    ('torch', 'PyTorch'),
    ('tqdm', 'tqdm'),
]

print('=' * 70)
print('Python 环境检查')
print('=' * 70)
print('Python 版本:', sys.version.replace('\n', ' '))
print('Python 路径:', sys.executable)
print('-' * 70)

ok = True
for module_name, display_name in packages:
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, '__version__', '已安装，但未读取到版本号')
        print(f'[OK] {display_name:<14} {version}')
    except Exception as e:
        ok = False
        print(f'[缺失] {display_name:<14} 请运行：pip install -r requirements.txt')

print('-' * 70)
try:
    import torch
    print('PyTorch CUDA 是否可用:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('当前 GPU:', torch.cuda.get_device_name(0))
    else:
        print('提示：CUDA 不可用也没关系，本项目图片很小，用 CPU 也可以训练。')
except Exception:
    pass

print('=' * 70)
if ok:
    print('环境检查通过：可以继续运行 generate_dataset / train_lenet / visualize_results。')
else:
    print('环境不完整：请先安装缺失依赖，再重新运行本脚本。')
print('=' * 70)
