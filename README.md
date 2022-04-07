# EfficientNetV2
In this project we are considering re-implementation and evaluation of the method ["EfficientNetV2: Smaller Models and Faster Training".](https://arxiv.org/abs/2104.00298). We desire the use the pretrained model and finetune it on CIFAR10 dataset.

The original [implementation](https://github.com/google/automl/tree/master/efficientnetv2) uses TensorFlow. Since we desire PyTorch-based implementations, we are considering these implementations in the literature:
1) https://github.com/hankyul2/EfficientNetV2-pytorch

2) https://github.com/jahongir7174/EfficientNetV2

3) https://github.com/d-li14/efficientnetv2.pytorch

There is also an implementation used in [TorchVision](http://pytorch.org/vision/main/generated/torchvision.models.efficientnet_v2_s.html) that we are considering.


## Literature Review
Earliest works such as [DenseNet](https://arxiv.org/abs/1608.06993) and [EfficientNet](https://arxiv.org/abs/1905.11946) focus mainly on parameter efficiency, while on the other side, works such as [RegNet](https://arxiv.org/abs/2003.13678) and [ResNest](https://arxiv.org/abs/2004.08955) focus on improving the training speed.
EfficientNetV2 is inspired by EfficientNet and focus on improving both training speed and parameter efficiency.
