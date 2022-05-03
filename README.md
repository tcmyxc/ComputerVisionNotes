# DL-digital-image-processing
这个仓库是本人读研期间研究内容的整理，主要是深度学习在计算机视觉领域的应用。

> 说是数字图像处理，但是现在冈萨雷斯最新的《数字图像处理》都开始讲解神经网络了。
>
> 深度学习，卷起来了[doge]

# 所需环境

```tex
miniconda 或者 venv（创建虚拟环境）
python 3.7+
pytorch 1.11.0
torchvision 0.12.0
```

导出依赖到文件

```bash
python3 -m pip freeze > requirements.txt
```

从文件安装依赖
```bash
pip install -r requirements.txt
```

# 目录

## 图像分类

- AlexNet
  - [论文地址](https://arxiv.org/abs/1404.5997)
  - [沐神的论文讲解](https://www.bilibili.com/video/BV1ih411J7Kz)
  - [源码讲解](https://www.bilibili.com/video/BV1aY4y1k767)
- VggNet
  - [论文地址](https://arxiv.org/abs/1409.1556)
  - [论文讲解](https://www.bilibili.com/video/BV1PB4y117Yh/)
  - [源码讲解 TODO]()
- GoogLeNet
- ResNet
- ResNeXt
- MobileNet_V1、V2、V3
- ShuffleNet_V1、V2
- EfficientNet_V1、V2 
- Vision Transformer
  - [论文地址](https://arxiv.org/abs/2010.11929)
  - [李沐&朱毅老师联合投稿的论文讲解](https://www.bilibili.com/video/BV15P4y137jb)
  - [源码讲解 TODO]()
- Swin Transformer
- ConvNeXt

## 目标检测

- R-CNN
- Faster RCNN/FPN
- SSD
- RetinaNet 
- YOLO系列

## 语义分割

- FCN 
- DeepLabV3
- U-Net 
