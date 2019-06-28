# YOLOv3-model-pruning

用 YOLOv3 模型在一个开源的人手检测数据集 [oxford hand](http://www.robots.ox.ac.uk/~vgg/data/hands/) 上做人手检测，并在此基础上进行实现模型压缩与加速。对于该数据集，对 YOLOv3 进行 channel pruning 之后，模型的参数量、模型大小降为原来的 1/6 ，FLOPs 降为原来的 1/3，前向推断的速度可以达到原来的 2 倍，同时可以保持 mAP 基本不变。

## 环境

Python3.6, Pytorch 1.0及以上

代码的实现参考了 eriklindernoren 的 [PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) ，因此代码的依赖环境可参考其 repo

*目前部分代码还在修改和完善，最近比较忙，待整理好再发出来，不过按照下面给出的论文进行复现也不会很难*

## 数据集准备

1. 下载[数据集](http://www.robots.ox.ac.uk/~vgg/data/hands/downloads/hand_dataset.tar.gz)，得到压缩文件
2. 将压缩文件解压到 data 目录，得到 hand_dataset 文件夹
3. 在 data 目录下执行 converter.py，生成 images、labels 文件夹和 train.txt、valid.txt 文件。训练集中一共有 4087 张图
   片，测试集中一共有 821 张图片

## 正常训练（Baseline）

```bash
python train.py --model_def config/yolov3-hand.cfg
```

## 剪枝算法介绍

本代码基于论文 [Learning Efficient Convolutional Networks Through Network Slimming (ICCV 2017)](http://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html) 进行改进实现的 channel pruning算法。原始论文中的算法是针对分类模型的，基于 BN 层的 gamma 系数进行剪枝的。本项目用到的剪枝算法不受限于 YOLOv3 模型，稍作改进理论上也是可以移植到其他目标检测模型的。

### 剪枝算法的步骤

1. 进行稀疏化训练

   ```bash
   python train.py --model_def config/yolov3-hand.cfg -sr --s 0.01
   ```

2. 基于 test_prune.py 文件进行剪枝（通过设定合理的剪枝规则），得到剪枝后的模型

3. 对剪枝后的模型进行微调（本项目对原算法进行了改进，即使不用微调也能达到较高的 mAP）

   ```bash
   python train.py --model_def config/prune_yolov3-hand.cfg -pre checkpoints/prune_yolov3_ckpt.pth
   ```

### 稀疏训练过程的可视化

1. 所有 BN 的 gamma 系数的五个五分位点随时间的变化图：

   ![](https://raw.githubusercontent.com/Lam1360/md-image/master/img/20190628202900.png)
   > 可以看到 10 次迭代后，60%的 gamma 系数已趋向于 0，40 次迭代后 80% 的 gamma 系数已趋向于 0

2. YOLOv3中第一个 BN 层的 gamma 系数随迭代次数的变化情况：

   ![](https://raw.githubusercontent.com/Lam1360/md-image/master/img/20190628202755.png)
   > 可以看到部分 gamma 系数逐步趋向于 0（表明其重要性逐渐削弱），而部分 gamma 系数能够保持其权重（表明其对网络的输出有一定的重要性）

3. 所有 BN 的 gamma 系数的分布随迭代次数的变化：

   ![](https://raw.githubusercontent.com/Lam1360/md-image/master/img/20190628203732.png)
   > 可以看到分布的重心逐渐向 0 靠近，表明 gamma 系数逐渐变得稀疏

### 剪枝前后的对比

1. 下图为设定合理阈值进行剪枝前后通道数的变化：

   ![](https://raw.githubusercontent.com/Lam1360/md-image/master/img/20190628205342.png)
   > 可以看到部分卷积层的通道数大幅度减少

2. 剪枝前后指标对比:

   |          | Parameters | Flops | Forward time（RTX 2070 TI） | mAP    |
   | -------- | ---------- | ----- | --------------------------- | ------ |
   | Baseline | 61.5M      | 32.8B | 15.0 ms                     | 0.7692 |
   | Prune    | 10.9M      | 9.6B  | 7.7 ms                      | 0.7722 |
   | Finetune | 同上       | 同上  | 同上                        | 0.7750 |

   > 可以看到，加入稀疏正则项之后，mAP 反而更高了（在实验过程中发现，其实 mAP上下波动 0.02 是正常现象），因此可以认为稀疏训练得到的 mAP 与正常训练几乎一致。将 prune 后得到的模型进行 finetune 并没有明显的提升，因此剪枝三步可以直接简化成两步。剪枝前后模型的参数量、模型大小降为原来的 1/6 ，FLOPs 降为原来的 1/3，前向推断的速度可以达到原来的 2 倍，同时可以保持 mAP 基本不变。

## Contact

有问题可以直接提 Issue，也可以发邮箱 huifeng.lin at qq.com

