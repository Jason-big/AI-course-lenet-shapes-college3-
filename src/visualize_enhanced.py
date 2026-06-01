from __future__ import annotations
from pathlib import Path
import json, csv, random, math, os, sys
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'shapes32'
FIG = ROOT / 'figures'
OUT = ROOT / 'outputs'
FIG.mkdir(exist_ok=True)

font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
if Path(font_path).exists():
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Noto Sans CJK JP'
plt.rcParams['axes.unicode_minus'] = False

classes = ['circle','square','triangle','cross','star']
cn = {'circle':'圆形','square':'方形','triangle':'三角形','cross':'十字形','star':'五角星'}
colors = ['#6C8EBF','#82B366','#D6B656','#B85450','#9673A6']


def savefig(path, dpi=220):
    plt.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()


def sample_image(cls, split='test', idx=0):
    files = sorted((DATA/split/cls).glob('*.png'))
    return Image.open(files[idx % len(files)]).convert('L')


def draw_box(ax, xy, w, h, text, fc='#F8FAFC', ec='#34495E', fontsize=10, lw=1.4, radius=0.05):
    x, y = xy
    box = FancyBboxPatch((x,y), w, h, boxstyle=f"round,pad=0.02,rounding_size={radius}",
                         linewidth=lw, edgecolor=ec, facecolor=fc)
    ax.add_patch(box)
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=fontsize)
    return box


def arrow(ax, start, end, color='#374151', lw=1.7, ms=14):
    arr = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=ms, lw=lw, color=color,
                          shrinkA=6, shrinkB=6)
    ax.add_patch(arr)
    return arr

# Fig 0: whole experimental workflow
fig, ax = plt.subplots(figsize=(12, 5.6))
ax.axis('off')
steps = [
    ('需求分析\n课程论文: 复现AI算法', '#E8F0FE'),
    ('自构数据集\nShape32-5 五类图形', '#E7F6EC'),
    ('数据预处理\n灰度读取/归一化/划分', '#FFF4CC'),
    ('模型搭建\nLeNet-5 CNN', '#FCE8E6'),
    ('训练与评估\nAccuracy/F1/混淆矩阵', '#EFE7F6'),
    ('结果讨论\n对照实验/错误案例', '#E5F5FA'),
]
x0, y0, w, h, gap = 0.25, 0.56, 1.65, 1.1, 0.25
for i, (t, c) in enumerate(steps):
    draw_box(ax, (x0+i*(w+gap), y0), w, h, t, fc=c, fontsize=11)
    if i < len(steps)-1:
        arrow(ax, (x0+i*(w+gap)+w, y0+h/2), (x0+(i+1)*(w+gap), y0+h/2))
# bottom feedback arrows
bottom = [(1.2,0.18,'调参：学习率、Epoch、Batch Size'), (4.8,0.18,'复现实验：记录日志并生成论文图表'), (8.4,0.18,'提交材料：论文、代码、数据集、README')]
for x,y,t in bottom:
    draw_box(ax, (x,y), 2.2, 0.7, t, fc='#FFFFFF', ec='#94A3B8', fontsize=10)
arrow(ax, (4.8,0.88), (5.95,1.48), color='#64748B', ms=12)
arrow(ax, (8.4,0.88), (9.85,1.48), color='#64748B', ms=12)
ax.set_xlim(0, 12.2); ax.set_ylim(0,2.25)
ax.text(0.25,2.05,'(a) 本课程论文的完整复现流程', fontsize=13, fontweight='bold')
savefig(FIG/'fig01_overall_workflow.png')

# Fig 1: dataset construction workflow + samples
fig = plt.figure(figsize=(12,7.4))
gs = fig.add_gridspec(2, 5, height_ratios=[1.0,1.1], hspace=0.35, wspace=0.12)
ax = fig.add_subplot(gs[0,:]); ax.axis('off')
workflow = [('随机设定参数\n位置/大小/角度', '#E8F0FE'),('绘制基础图形\n圆/方/三角/十字/星', '#E7F6EC'),('加入扰动\n噪声/模糊/干扰线', '#FFF4CC'),('自动标注\n文件夹名=类别标签', '#FCE8E6'),('划分数据\ntrain/val/test', '#EFE7F6')]
for i,(t,c) in enumerate(workflow):
    draw_box(ax, (0.3+i*2.2,0.38), 1.75, 0.7, t, fc=c, fontsize=10)
    if i<4: arrow(ax, (0.3+i*2.2+1.75,0.73),(0.3+(i+1)*2.2,0.73),ms=12)
ax.set_xlim(0,11.4); ax.set_ylim(0.1,1.5)
ax.text(0.1,1.32,'(a) 自构数据集生成逻辑', fontsize=12, fontweight='bold')
for i, cls in enumerate(classes):
    ax2 = fig.add_subplot(gs[1,i])
    imgs = [sample_image(cls,'train',j*9) for j in range(6)]
    canvas = Image.new('L',(32*3,32*2),255)
    for j,img in enumerate(imgs):
        canvas.paste(img,(32*(j%3),32*(j//3)))
    ax2.imshow(canvas,cmap='gray', vmin=0, vmax=255)
    ax2.set_title(f'{cn[cls]}\n{cls}', fontsize=10)
    ax2.axis('off')
fig.suptitle('Shape32-5 自构数据集：生成流程与类别样本', fontsize=15, fontweight='bold', y=0.98)
savefig(FIG/'fig02_dataset_generation_and_samples.png')

# Fig 2: augmentation comparison
base = sample_image('star','train',2)
variants = []
labels = []
for name, im in [
    ('原始样本', base),
    ('旋转', base.rotate(18, fillcolor=255)),
    ('平移', ImageOps.expand(base.crop((0,0,30,30)), border=(2,2,0,0), fill=255).resize((32,32))),
    ('模糊', base.filter(ImageFilter.GaussianBlur(radius=0.8))),
    ('噪声', None),
    ('干扰线', None),
]:
    if im is None and name=='噪声':
        arr=np.array(base).astype(np.int16)+np.random.normal(0,18,(32,32))
        im=Image.fromarray(np.clip(arr,0,255).astype(np.uint8))
    elif im is None:
        im=base.copy(); d=ImageDraw.Draw(im); d.line([(2,20),(29,23)], fill=85, width=1); d.line([(5,8),(25,5)], fill=120, width=1)
    variants.append(im); labels.append(name)
fig, axes = plt.subplots(2,3, figsize=(8.5,5.8))
for ax, im, lab in zip(axes.ravel(), variants, labels):
    ax.imshow(im,cmap='gray', vmin=0, vmax=255)
    ax.set_title(lab, fontsize=11)
    ax.axis('off')
fig.suptitle('数据增强与扰动示例：使模型学习“形状结构”而不是固定模板', fontsize=14, fontweight='bold')
savefig(FIG/'fig03_augmentation_examples.png')

# Fig 3: original vs project LeNet architecture
fig, ax = plt.subplots(figsize=(13.5,6))
ax.axis('off')
levels = [
    ('Input\n1×32×32', 0.2, '#F8FAFC'),
    ('C1 Conv\n6@28×28\n5×5 kernel', 1.8, '#DDEBFF'),
    ('S2 AvgPool\n6@14×14', 3.4, '#E7F6EC'),
    ('C3 Conv\n16@10×10\n5×5 kernel', 5.0, '#DDEBFF'),
    ('S4 AvgPool\n16@5×5', 6.8, '#E7F6EC'),
    ('Flatten\n400', 8.35, '#FFF4CC'),
    ('FC\n120', 9.55, '#FCE8E6'),
    ('FC\n84', 10.65, '#FCE8E6'),
    ('Output\n5 classes', 11.75, '#EFE7F6')]
for i,(txt,x,c) in enumerate(levels):
    ww=1.05 if i not in [3,8] else 1.3
    draw_box(ax, (x,1.45), ww,1.1,txt,fc=c,fontsize=10,ec='#334155')
    if i < len(levels)-1:
        arrow(ax,(x+ww,2.0),(levels[i+1][1],2.0),ms=12)
# draw stacked feature maps
for x,n,wid,hei,col in [(1.25,6,.32,.62,'#A7C7E7'),(2.9,6,.25,.45,'#B9E4C9'),(4.35,16,.26,.55,'#A7C7E7'),(6.25,16,.21,.37,'#B9E4C9')]:
    for k in range(min(n,8)):
        ax.add_patch(Rectangle((x+0.035*k,0.48+0.025*k),wid,hei,facecolor=col,edgecolor='#64748B',lw=0.8,alpha=0.9))
ax.text(0.2,3.05,'(a) 本文复现的 LeNet-5 风格网络结构',fontsize=13,fontweight='bold')
ax.text(0.2,0.2,'(b) 结构来源说明：参考 LeCun 等人 1998 年 LeNet-5 网络层次思想重新绘制，本文将输出层改为 5 类几何图形分类。',fontsize=10,color='#475569')
ax.set_xlim(0,13.4); ax.set_ylim(0,3.4)
savefig(FIG/'fig04_lenet_architecture_composite.png')

# Fig 4: convolution and pooling principle
fig, axes = plt.subplots(1,2, figsize=(12,5.5))
# convolution demo
ax=axes[0]; ax.set_title('(a) 卷积核在图像上滑动提取局部特征',fontsize=12,fontweight='bold'); ax.axis('off')
# grid 6x6 and kernel 3x3
for i in range(6):
    for j in range(6):
        ax.add_patch(Rectangle((j,5-i),1,1,facecolor='#F8FAFC',edgecolor='#CBD5E1'))
for i in range(3):
    for j in range(3):
        ax.add_patch(Rectangle((1+j,2+i),1,1,facecolor='#DDEBFF',edgecolor='#2563EB',lw=1.6,alpha=0.75))
ax.text(3.1,3.5,'3×3 卷积核\n关注局部区域',fontsize=10,ha='left',va='center')
arrow(ax,(4.2,3.5),(7.0,3.5),ms=16)
for i in range(4):
    for j in range(4):
        ax.add_patch(Rectangle((7+j,4-i),.8,.8,facecolor='#E7F6EC',edgecolor='#94A3B8'))
ax.text(8.6,0.35,'输出特征图',fontsize=10,ha='center')
ax.set_xlim(-0.3,11.2); ax.set_ylim(0,6.3)
# pooling demo
ax=axes[1]; ax.set_title('(b) 平均池化降低尺寸并增强平移鲁棒性',fontsize=12,fontweight='bold'); ax.axis('off')
vals=np.array([[1,2,1,0],[2,4,3,1],[1,3,6,2],[0,1,2,3]])
for i in range(4):
    for j in range(4):
        ax.add_patch(Rectangle((j,4-i),1,1,facecolor='#FFF7ED',edgecolor='#FDBA74'))
        ax.text(j+.5,4-i+.5,str(vals[i,j]),ha='center',va='center',fontsize=11)
for x,y in [(0,3),(2,3),(0,1),(2,1)]:
    ax.add_patch(Rectangle((x,y),2,2,fill=False,edgecolor='#EA580C',lw=2.2))
arrow(ax,(4.7,2.5),(6.3,2.5),ms=16)
pooled=np.array([[2.25,1.25],[1.25,3.25]])
for i in range(2):
    for j in range(2):
        ax.add_patch(Rectangle((6.8+j*1.2,3.1-i*1.2),1.2,1.2,facecolor='#E7F6EC',edgecolor='#22C55E'))
        ax.text(6.8+j*1.2+.6,3.1-i*1.2+.6,f'{pooled[i,j]:.2f}',ha='center',va='center',fontsize=11)
ax.text(7.95,1.0,'每个 2×2 区域取平均值',fontsize=10,ha='center')
ax.set_xlim(-0.3,10); ax.set_ylim(0.5,5.2)
fig.suptitle('CNN 两个关键操作：卷积与池化', fontsize=15, fontweight='bold')
savefig(FIG/'fig05_convolution_pooling_principle.png')

# Fig 5: feature maps (optional torch)
try:
    import torch
    sys.path.insert(0, str(ROOT/'src'))
    from model import LeNet5
    model = LeNet5(num_classes=5)
    model.load_state_dict(torch.load(OUT/'lenet5_shapes32_best.pt', map_location='cpu'))
    model.eval()
    im = sample_image('triangle','test',0)
    arr = np.array(im).astype('float32')/255.0
    x = torch.tensor(((arr-0.5)/0.5)[None,None,:,:])
    with torch.no_grad():
        c1 = model.features[:2](x)[0].numpy()
        s2 = model.features[:3](x)[0].numpy()
        c3 = model.features[:5](x)[0].numpy()
        logits = model(x).softmax(dim=1).numpy()[0]
    fig = plt.figure(figsize=(12,7.5))
    gs=fig.add_gridspec(3,8, height_ratios=[1,1,1.1], wspace=0.12, hspace=0.28)
    ax=fig.add_subplot(gs[0:2,0:2]); ax.imshow(im,cmap='gray'); ax.set_title('输入样本\ntriangle',fontsize=11,fontweight='bold'); ax.axis('off')
    for i in range(6):
        ax=fig.add_subplot(gs[0,i+2]); ax.imshow(c1[i],cmap='viridis'); ax.set_title(f'C1-{i+1}',fontsize=8); ax.axis('off')
    for i in range(8):
        ax=fig.add_subplot(gs[1,i]); ax.imshow(c3[i],cmap='viridis'); ax.set_title(f'C3-{i+1}',fontsize=8); ax.axis('off')
    ax=fig.add_subplot(gs[2,0:4]);
    ax.bar([cn[c] for c in classes], logits, color=colors)
    ax.set_ylim(0,1.05); ax.set_ylabel('预测概率'); ax.set_title('Softmax 输出概率')
    for i,p in enumerate(logits): ax.text(i,p+0.02,f'{p:.2f}',ha='center',fontsize=8)
    ax=fig.add_subplot(gs[2,4:]); ax.axis('off')
    draw_box(ax,(0.05,0.52),0.9,0.3,'浅层特征：边缘/线条/角点',fc='#E8F0FE',fontsize=10)
    draw_box(ax,(0.05,0.12),0.9,0.3,'深层特征：形状组合与类别判别',fc='#E7F6EC',fontsize=10)
    arrow(ax,(0.5,0.52),(0.5,0.42),ms=12)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    fig.suptitle('LeNet-5 中间特征图可视化：从像素到类别概率',fontsize=15,fontweight='bold')
    savefig(FIG/'fig06_feature_map_visualization.png')
except Exception as e:
    print('feature map figure skipped:', e)

# Fig 6: training dashboard combined
with open(OUT/'history.json', encoding='utf-8') as f: hist=json.load(f)
epochs=[h['epoch'] for h in hist]
train_loss=[h['train_loss'] for h in hist]
train_acc=[h['train_accuracy'] for h in hist]
val_acc=[h['val_accuracy'] for h in hist]
val_f1=[h.get('val_f1_macro', np.nan) for h in hist]
fig, axes=plt.subplots(1,2,figsize=(12,4.8))
axes[0].plot(epochs, train_loss, marker='o', markersize=3, lw=2, color='#2563EB')
axes[0].set_title('(a) 训练损失下降曲线',fontweight='bold'); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].grid(alpha=.25)
axes[1].plot(epochs, train_acc, marker='o', markersize=3, lw=2, label='Train Acc', color='#16A34A')
axes[1].plot(epochs, val_acc, marker='s', markersize=3, lw=2, label='Val Acc', color='#DC2626')
axes[1].plot(epochs, val_f1, marker='^', markersize=3, lw=2, label='Val F1', color='#7C3AED')
axes[1].set_title('(b) 准确率与 F1 变化曲线',fontweight='bold'); axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Score'); axes[1].set_ylim(0.35,1.04); axes[1].grid(alpha=.25); axes[1].legend()
fig.suptitle('训练过程综合展示：损失下降与验证性能提升',fontsize=15,fontweight='bold')
savefig(FIG/'fig07_training_dashboard.png')

# Fig 7: metric + confusion matrix dashboard
with open(OUT/'metrics_lenet.json', encoding='utf-8') as f: lenet=json.load(f)
with open(OUT/'metrics_svm.json', encoding='utf-8') as f: svm=json.load(f)
cm=np.array(lenet['confusion_matrix'])
fig=plt.figure(figsize=(12,5.2))
gs=fig.add_gridspec(1,2,width_ratios=[1.1,1])
ax=fig.add_subplot(gs[0,0])
metric_names=['Accuracy','Precision','Recall','F1']
lenet_vals=[lenet['accuracy'],lenet['precision_macro'],lenet['recall_macro'],lenet['f1_macro']]
svm_vals=[svm['accuracy'],svm['precision_macro'],svm['recall_macro'],svm['f1_macro']]
x=np.arange(len(metric_names)); bw=.35
ax.bar(x-bw/2, lenet_vals, bw, label='LeNet-5', color='#2563EB')
ax.bar(x+bw/2, svm_vals, bw, label='RBF-SVM', color='#F97316')
ax.set_xticks(x); ax.set_xticklabels(metric_names); ax.set_ylim(0,1.08); ax.set_ylabel('Score'); ax.set_title('(a) LeNet-5 与 SVM 指标对比',fontweight='bold'); ax.legend(loc='lower right'); ax.grid(axis='y',alpha=.25)
for i,v in enumerate(lenet_vals): ax.text(i-bw/2,v+0.02,f'{v*100:.1f}%',ha='center',fontsize=8)
for i,v in enumerate(svm_vals): ax.text(i+bw/2,v+0.02,f'{v*100:.1f}%',ha='center',fontsize=8)
ax=fig.add_subplot(gs[0,1])
im=ax.imshow(cm,cmap='Blues')
ax.set_title('(b) LeNet-5 测试集混淆矩阵',fontweight='bold')
ax.set_xticks(range(5)); ax.set_yticks(range(5)); ax.set_xticklabels([cn[c] for c in classes],rotation=35,ha='right'); ax.set_yticklabels([cn[c] for c in classes]); ax.set_xlabel('预测类别'); ax.set_ylabel('真实类别')
for i in range(5):
    for j in range(5):
        ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>40 else '#111827',fontsize=9)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.suptitle('定量实验结果综合图：CNN 明显优于传统像素特征 SVM',fontsize=15,fontweight='bold')
savefig(FIG/'fig08_results_dashboard.png')

# Fig 8: per-class precision/recall/f1
# parse classification_report by using values from output? Use sklearn report dict unavailable. Use known metrics from text? compute from cm.
prec = np.diag(cm)/cm.sum(axis=0)
rec = np.diag(cm)/cm.sum(axis=1)
f1 = 2*prec*rec/(prec+rec)
fig, ax=plt.subplots(figsize=(10,5.4))
x=np.arange(5); bw=.25
ax.bar(x-bw, prec, bw, label='Precision', color='#2563EB')
ax.bar(x, rec, bw, label='Recall', color='#16A34A')
ax.bar(x+bw, f1, bw, label='F1-score', color='#7C3AED')
ax.set_xticks(x); ax.set_xticklabels([cn[c] for c in classes]); ax.set_ylim(0.75,1.05); ax.set_ylabel('Score')
ax.set_title('各类别识别效果：十字形最稳定，圆形/方形存在少量混淆',fontsize=14,fontweight='bold')
ax.grid(axis='y',alpha=.25); ax.legend(loc='lower right')
for j,vals in enumerate([prec,rec,f1]):
    offset=[-bw,0,bw][j]
    for i,v in enumerate(vals): ax.text(i+offset,v+0.006,f'{v:.2f}',ha='center',fontsize=8)
savefig(FIG/'fig09_per_class_metrics.png')

# Fig 9: project structure and file outputs
fig,ax=plt.subplots(figsize=(11,6.2))
ax.axis('off')
root_box = draw_box(ax,(0.25,4.8),2.2,.55,'项目根目录\nai_course_lenet_shapes_project',fc='#E8F0FE',fontsize=11,ec='#2563EB')
items=[
    ('src/\n核心代码',0.35,3.4,'#DDEBFF','generate_dataset.py\ntrain_lenet.py\nevaluate_baseline_svm.py\nvisualize_results.py'),
    ('data/\n自构数据集',3.0,3.4,'#E7F6EC','shapes32/train\nshapes32/val\nshapes32/test\nmetadata.csv'),
    ('outputs/\n实验结果',5.65,3.4,'#FFF4CC','best.pt\nhistory.json\nmetrics_lenet.json\nmetrics_svm.json'),
    ('figures/\n论文图片',8.3,3.4,'#FCE8E6','样本图\n网络结构图\n训练曲线\n混淆矩阵\n错误案例'),
]
for title,x,y,c,txt in items:
    draw_box(ax,(x,y),2.1,.65,title,fc=c,fontsize=11)
    draw_box(ax,(x,y-1.65),2.1,1.35,txt,fc='#FFFFFF',ec='#CBD5E1',fontsize=9)
    arrow(ax,(1.35,4.8),(x+1.05,y+.65),ms=10,color='#64748B')
ax.text(0.25,0.65,'说明：提交 GitHub 时建议保留 src、data、figures、outputs、docs、README.md、requirements.txt。',fontsize=10,color='#475569')
ax.set_xlim(0,10.8); ax.set_ylim(0.35,5.65)
ax.set_title('项目文件结构与复现实验输出文件',fontsize=15,fontweight='bold')
savefig(FIG/'fig10_project_structure.png')

print('Enhanced figures generated in', FIG)
