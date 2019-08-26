# YOLOv3-model-pruning

用 YOLOv3 模型在一个开源的人手检测数据集 [oxford hand](http://www.robots.ox.ac.uk/~vgg/data/hands/) 上做人手检测，并在此基础上做模型剪枝。对于该数据集，对 YOLOv3 进行 channel pruning 之后，模型的参数量、模型大小减少 80% ，FLOPs 降低 70%，前向推断的速度可以达到原来的 200%，同时可以保持 mAP 基本不变。

## 环境

Python3.6, Pytorch 1.0及以上

YOLOv3 的实现参考了 eriklindernoren 的 [PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) ，因此代码的依赖环境也可以参考其 repo

## 数据集准备

1. 下载[数据集](http://www.robots.ox.ac.uk/~vgg/data/hands/downloads/hand_dataset.tar.gz)，得到压缩文件
2. 将压缩文件解压到 data 目录，得到 hand_dataset 文件夹
3. 在 data 目录下执行 converter.py，生成 images、labels 文件夹和 train.txt、valid.txt 文件。训练集中一共有 4807 张图
   片，测试集中一共有 821 张图片

## 正常训练（Baseline）

```bash
python train.py --model_def config/yolov3-hand.cfg
```

## 剪枝算法介绍

本代码基于论文 [Learning Efficient Convolutional Networks Through Network Slimming (ICCV 2017)](http://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html) 进行改进实现的 channel pruning算法，类似的代码实现还有这个 [yolov3-network-slimming](https://github.com/talebolano/yolov3-network-slimming)。原始论文中的算法是针对分类模型的，基于 BN 层的 gamma 系数进行剪枝的。

### 剪枝算法的大概步骤

以下只是算法的大概步骤，具体实现过程中还要做 s 参数的尝试或者需要进行迭代式剪枝等。

1. 进行稀疏化训练

   ```bash
   python train.py --model_def config/yolov3-hand.cfg -sr --s 0.01
   ```

2. 基于 test_prune.py 文件进行剪枝，得到剪枝后的模型

3. 对剪枝后的模型进行微调
   ```bash
   python train.py --model_def config/prune_yolov3-hand.cfg -pre checkpoints/prune_yolov3_ckpt.pth
   ```

### 剪枝前后的对比

1. 下图为对部分卷积层进行剪枝前后通道数的变化：

   ![](https://raw.githubusercontent.com/Lam1360/md-image/master/img/20190628205342.png)
   > 部分卷积层的通道数大幅度减少

2. 剪枝前后指标对比：

   |                | 参数数量 | 模型体积 |Flops | 前向推断耗时（2070 TI） |  mAP   |
   | :------------: | :------:| :-----: | :---: | :-------------------: | :----: |
   | Baseline (416) |  61.5M  | 246.4MB |32.8B  |         15.0 ms       | 0.7692 |
   |  Prune (416)   |  10.9M  | 43.6MB  | 9.6B  |         7.7 ms        | 0.7722 |
   | Finetune (416) |   同上   | 同上    | 同上  |          同上         | 0.7750 |
   
   > 加入稀疏正则项之后，mAP 反而更高了（在实验过程中发现，其实 mAP上下波动 0.02 是正常现象），因此可以认为稀疏训练得到的 mAP 与正常训练几乎一致。将 prune 后得到的模型进行 finetune 并没有明显的提升，因此剪枝三步可以直接简化成两步。剪枝前后模型的参数量、模型大小降为原来的 1/6 ，FLOPs 降为原来的 1/3，前向推断的速度可以达到原来的 2 倍，同时可以保持 mAP 基本不变。*需要明确的是，上面表格中剪枝的效果是只是针对该数据集的，不一定能保证在其他数据集上也有同样的效果*
   
3. 剪枝后模型的测试：

   Prune 模型的权重已放在百度网盘上 （[提取码: gnzx](https://pan.baidu.com/s/13Ycj7JccBHWYF590bgFRxQ)），可以通过执行以下代码进行测试：
   ```bash
   python test.py --model_def config/prune_yolov3-hand.cfg --weights_path weights/prune_yolov3_ckpt.pth --data_config config/oxfordhand.data --class_path data/oxfordhand.names --conf_thres 0.01
   ```
