# LeNet-5 课程论文复现实验：自己实现教程

## 一、你这篇论文到底做了什么

这篇论文复现的是经典卷积神经网络 LeNet-5。原始 LeNet-5 主要用于手写数字识别，本文把它改造成五分类模型，用来自构数据集 Shape32-5 的图像分类任务。数据集不是直接下载公开数据，而是用 Python 自动生成五类 32×32 灰度图：圆形、方形、三角形、十字形、五角星。

整个流程是：

1. 先生成自构数据集；
2. 把图片读入程序并归一化；
3. 搭建 LeNet-5 风格 CNN；
4. 用训练集训练模型，用验证集选择最佳模型；
5. 用测试集计算准确率、精确率、召回率、F1 值和混淆矩阵；
6. 再用 SVM 做对照实验；
7. 最后把训练曲线、混淆矩阵、错误案例等图放进课程论文。

## 二、为什么这个选题适合课程论文

课程要求复现一种人工智能算法，并尽量包含算法原理、模型搭建、实验平台、训练过程、实验结果展示与分析。LeNet-5 很适合，因为它是 CNN 的经典基础模型，结构不复杂，但能完整体现深度学习图像分类的关键步骤。

另外，课程要求里提到“自构数据集”可以加分，所以本项目没有直接用 MNIST，而是生成了 Shape32-5 数据集。这个数据集有自己的类别、标签和扰动方式，在全班里重复率会更低。

## 三、文件应该怎么用

你下载 `ai_course_lenet_shapes_project.zip` 后，先解压。解压后进入项目文件夹：

```bash
cd ai_course_lenet_shapes_project
```

安装依赖：

```bash
pip install -r requirements.txt
```

如果你用的是 Anaconda，可以先建一个环境：

```bash
conda create -n ai_course python=3.10 -y
conda activate ai_course
pip install -r requirements.txt
```

## 四、第一步：生成自构数据集

运行：

```bash
python src/generate_dataset.py
```

运行后会生成：

```text
data/shapes32/train/
data/shapes32/val/
data/shapes32/test/
data/metadata.csv
```

每个 split 下面都有五个类别文件夹：circle、square、triangle、cross、star。标签就是由文件夹名称决定的，比如 `data/shapes32/train/circle/circle_0001.png` 的标签就是 circle。

## 五、第二步：训练 LeNet-5

运行：

```bash
python src/train_lenet.py --epochs 30 --batch_size 64 --lr 0.001
```

你会看到每一轮 epoch 的训练日志，例如：

```text
Epoch 01: loss=1.4336, train_acc=0.4183, val_acc=0.5000
Epoch 30: loss=0.0430, train_acc=0.9846, val_acc=0.9875
```

训练结束后会生成：

```text
outputs/lenet5_shapes32_best.pt
outputs/history.json
outputs/metrics_lenet.json
```

其中：

- `lenet5_shapes32_best.pt` 是保存下来的模型权重；
- `history.json` 是训练过程数据，用来画训练曲线；
- `metrics_lenet.json` 是测试集结果，用来写论文实验结果。

## 六、第三步：运行 SVM 对照实验

运行：

```bash
python src/evaluate_baseline_svm.py
```

这个脚本会把每张 32×32 图像直接展平成 1024 维像素向量，然后训练一个 RBF 核 SVM。论文中用它和 LeNet-5 做对比，目的是说明 CNN 能自动提取局部空间结构，而 SVM 只看展开后的像素向量。

## 七、第四步：生成论文图片

运行：

```bash
python src/visualize_results.py
```

运行后会在 `figures/` 里得到论文用图，包括：

- 数据集样本图；
- 类别分布图；
- LeNet-5 网络结构图；
- 训练损失曲线；
- 训练/验证准确率曲线；
- 混淆矩阵；
- 代码运行日志截图；
- 错误案例或困难样本图。

## 八、模型结构怎么讲

你可以这样理解 LeNet-5：

第一层卷积层先在图像里找简单边缘和局部线条；第一层池化把图像压缩，让模型对轻微平移不那么敏感；第二层卷积继续把低级边缘组合成更复杂的形状结构，比如角点、交叉点和闭合轮廓；最后全连接层把这些特征整合起来，判断它到底是圆形、方形、三角形、十字形还是五角星。

本文模型结构是：

```text
Input 1×32×32
→ Conv 5×5, 6 channels
→ Tanh
→ Average Pooling
→ Conv 5×5, 16 channels
→ Tanh
→ Average Pooling
→ Flatten
→ Fully Connected 120
→ Fully Connected 84
→ Output 5 classes
```

## 九、论文里的结果怎么解释

本次实验结果是：

- LeNet-5 测试准确率：95.00%；
- SVM 测试准确率：82.25%；
- LeNet-5 比 SVM 高 12.75 个百分点。

可以这样分析：CNN 的优势在于它不是把图像当成一串孤立数字，而是通过卷积核学习局部结构。比如十字形有明显的横竖交叉特征，三角形有明显尖角和三条边，五角星有多个尖角，所以 CNN 容易学到这些图形特征。圆形和方形有时候会混淆，是因为它们都是闭合轮廓，在低分辨率和噪声影响下，部分方形角点不明显，看起来会更接近圆形。

## 十、你提交前必须改哪些地方

1. 把 Word 封面里的学院、专业、姓名、学号、指导老师补全；
2. 自己在电脑上完整运行一次代码；
3. 用你自己的终端截图替换论文里的“代码运行截图”；
4. 把整个项目上传到 GitHub；
5. 把论文最后的 GitHub 链接替换为你的真实链接；
6. 检查“课程论文来源说明”，确保它符合你的真实完成情况。

## 十一、GitHub 上传命令

```bash
git init
git add .
git commit -m "Initial reproduction of LeNet-5 on Shape32-5 dataset"
git branch -M main
git remote add origin https://github.com/你的用户名/lenet-shape32-reproduction.git
git push -u origin main
```

上传成功后，把仓库链接放进论文第 9 部分。

## 十二、答辩时可以怎么说

老师如果问“为什么不用 MNIST”，你可以回答：

> 因为课程要求中自构数据集是加分项，而且 MNIST 使用频率比较高，所以我选择用 Python 自己构建 Shape32-5 数据集。它虽然是合成数据，但加入了旋转、平移、噪声和模糊，更能体现模型对图形结构的学习能力。

老师如果问“为什么 LeNet-5 比 SVM 好”，你可以回答：

> SVM 使用的是展平后的原始像素特征，图像的二维空间结构会被弱化；而 LeNet-5 的卷积层可以提取局部边缘、角点和交叉结构，因此更适合图像识别任务。

老师如果问“模型还有什么不足”，你可以回答：

> 数据集仍然是程序生成的简化图像，和真实拍摄图像有差距；模型结构也比较浅，对复杂背景和更细粒度类别不一定适用。后续可以加入真实拍摄样本，或者尝试 ResNet 等更深的网络。
