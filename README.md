# 基于 LeNet-5 卷积神经网络的几何图形分类实验

## 1. 项目简介

本项目为《人工智能》课程论文配套代码，主要复现 LeNet-5 风格卷积神经网络，并将其应用于 Shape32-5 几何图形五分类任务。实验类别包括圆形、方形、三角形、十字形和五角星。项目完成了数据整理、数据预处理、模型搭建、模型训练、对照实验、测试评估和结果可视化等流程。

本实验使用 LeNet-5 CNN 作为主要模型，并设置 RBF 核 SVM 作为传统机器学习基线模型进行对照。实验结果表明，LeNet-5 在测试集上取得了 95.00% 的准确率，优于 RBF-SVM 基线模型。

## 2. 项目结构

```text
ai_course_lenet_shapes_project/
├── src/
│   ├── generate_dataset.py
│   ├── train_lenet.py
│   ├── evaluate_baseline_svm.py
│   ├── visualize_results.py
│   └── model.py
├── data/
│   └── shapes32/
│       ├── train/
│       ├── val/
│       └── test/
├── outputs/
│   ├── best.pt
│   ├── history.json
│   ├── metrics_lenet.json
│   └── metrics_svm.json
├── figures/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── class_metrics.png
├── requirements.txt
└── README.md
```

## 3. 实验环境

本项目在 Windows 系统下运行，主要依赖如下：

```text
Python 3.10
PyTorch
scikit-learn
Pillow
Matplotlib
NumPy
```

安装依赖：

```bash
pip install -r requirements.txt
```

## 4. 数据集说明

本文使用 Shape32-5 几何图形数据集，图像尺寸为 32×32，输入通道数为 1。数据集包含 5 个类别：

```text
circle
square
triangle
cross
star
```

每类样本划分如下：

| 类别       |  训练集 | 验证集 | 测试集 |   合计 |
| -------- | ---: | --: | --: | ---: |
| circle   |  350 |  80 |  80 |  510 |
| square   |  350 |  80 |  80 |  510 |
| triangle |  350 |  80 |  80 |  510 |
| cross    |  350 |  80 |  80 |  510 |
| star     |  350 |  80 |  80 |  510 |
| total    | 1750 | 400 | 400 | 2550 |

如果仓库中没有上传完整 data 文件夹，可以运行以下命令重新生成或整理数据：

```bash
python src/generate_dataset.py
```

## 5. 运行步骤

### 5.1 生成或整理数据集

```bash
python src/generate_dataset.py
```

运行后，数据会保存在：

```text
data/shapes32/
```

### 5.2 训练 LeNet-5 模型

```bash
python src/train_lenet.py --epochs 30 --batch_size 64 --lr 0.001
```

训练完成后，模型权重和训练日志会保存到：

```text
outputs/
```

### 5.3 运行 SVM 对照实验

```bash
python src/evaluate_baseline_svm.py
```

该步骤会使用原始像素特征训练 RBF-SVM 基线模型，并输出对照实验结果。

### 5.4 生成可视化结果

```bash
python src/visualize_results.py
```

运行后会生成训练曲线、混淆矩阵、类别指标图和错误案例图等，结果保存在：

```text
figures/
```

## 6. 实验结果

LeNet-5 模型在测试集上的结果如下：

| 模型          | Accuracy | Precision | Recall | F1-score |
| ----------- | -------: | --------: | -----: | -------: |
| LeNet-5 CNN |   95.00% |    95.11% | 95.00% |   95.01% |
| RBF-SVM     |   82.25% |    82.86% | 82.25% |   82.43% |

实验结果表明，LeNet-5 能够通过卷积层和池化层学习图像的边缘、角点、交叉点和闭合轮廓等局部结构特征，因此相比直接使用原始像素向量的 RBF-SVM 具有更好的分类效果。

## 7. 参考资料

## 参考资料

本项目在完成 LeNet-5 几何图形分类实验过程中，主要参考了以下论文、开源资料和工具文档：

[1] LeCun Y, Bottou L, Bengio Y, Haffner P. Gradient-based learning applied to document recognition[J]. Proceedings of the IEEE, 1998, 86(11): 2278-2324.
[2] Fukushima K. Neocognitron: A self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position[J]. Biological Cybernetics, 1980, 36(4): 193-202.
[3] Rumelhart D E, Hinton G E, Williams R J. Learning representations by back-propagating errors[J]. Nature, 1986, 323(6088): 533-536.
[4] Goodfellow I, Bengio Y, Courville A. Deep Learning[M]. Cambridge: MIT Press, 2016.
[5] Kingma D P, Ba J. Adam: A method for stochastic optimization[C]//International Conference on Learning Representations. 2015.
[6] Cortes C, Vapnik V. Support-vector networks[J]. Machine Learning, 1995, 20(3): 273-297.
[7] Burges C J C. A tutorial on support vector machines for pattern recognition[J]. Data Mining and Knowledge Discovery, 1998, 2(2): 121-167.
[8] Zhang A, Lipton Z C, Li M, Smola A J. Dive into Deep Learning[EB/OL]. https://d2l.ai/
[9] PyTorch Documentation. https://pytorch.org/docs/stable/index.html
[10] Paszke A, Gross S, Massa F, et al. PyTorch: An imperative style, high-performance deep learning library[C]//Advances in Neural Information Processing Systems. 2019, 32.
[11] scikit-learn Documentation. https://scikit-learn.org/stable/
[12] Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: Machine learning in Python[J]. Journal of Machine Learning Research, 2011, 12: 2825-2830.
[13] Pillow Documentation. https://pillow.readthedocs.io/
[14] Matplotlib Documentation. https://matplotlib.org/stable/
[15] Hunter J D. Matplotlib: A 2D graphics environment[J]. Computing in Science & Engineering, 2007, 9(3): 90-95.
[16] Kaggle Shape32 dataset. https://www.kaggle.com/datasets/zhenx1nxi/shape32
[17] ChawDoe. LeNet5-MNIST-PyTorch[EB/OL]. https://github.com/ChawDoe/LeNet5-MNIST-PyTorch
[18] GitHub Docs: Uploading a project to GitHub. https://docs.github.com/en/get-started/start-your-journey/uploading-a-project-to-github


## 8. 说明

本项目仅用于《人工智能》课程论文实验复现与学习交流。论文写作过程中使用大语言模型辅助进行结构梳理、文字润色、代码调试和图片生成建议，最终实验运行、结果截图和论文整理由本人完成。
