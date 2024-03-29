# Computer Vision

记录本人在研究生期间学习的论文及代码实现

# 所需环境

```tex
miniconda 或者 venv（创建虚拟环境）
python 3.7+
pytorch 1.11.0+
torchvision 0.12.0+
```

导出依赖到文件

```bash
python3 -m pip freeze > requirements.txt

# 或者
pipreqs . --encoding=utf8 --force
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
  - [源码讲解](https://www.bilibili.com/video/BV1wU4y1m7zo/)
- ResNet
  - [论文地址](https://arxiv.org/pdf/1512.03385.pdf)
  - [沐神的论文讲解](https://www.bilibili.com/video/BV1Fb4y1h73E)
  - 源码讲解
- ResNeXt
- MobileNet_V1、V2、V3
- ShuffleNet_V1、V2
- EfficientNet_V1、V2 
- Vision Transformer
  - [论文地址](https://arxiv.org/abs/2010.11929)
  - [李沐&朱毅老师联合投稿的论文讲解](https://www.bilibili.com/video/BV15P4y137jb)
  - 源码讲解
- Swin Transformer
- ConvNeXt

## 目标检测

- R-CNN
  - 论文地址
  - [视频讲解](https://www.bilibili.com/video/BV1iY4y1z78q)

- SPP-net
  - 论文地址
  - [视频讲解](https://www.bilibili.com/video/BV1vB4y19712)

- Fast R-CNN
  - 论文地址
  - [视频讲解](https://www.bilibili.com/video/BV1Z3411A7u6)

- Faster RCNN
- SSD
- RetinaNet 
- YOLO系列

## 语义分割

- FCN 
- DeepLabV3
- U-Net 
