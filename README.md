# EfficientNetV2
This is our project for the course "EC 523: Deep Learning", presented in Boston University. In this project, we investigate and evaluate the method in ["EfficientNetV2: Smaller Models and Faster Training".](https://arxiv.org/abs/2104.00298). EfficientNetV2 is a family of convolutional networks, with the focus on improving the training speed and parameter efficiency of literature methods. Due to the limitation of computional resources, we use the pretrained model on ImageNet dataset, and finetune its parameters on the CIFAR datasets.

The original [implementation](https://github.com/google/automl/tree/master/efficientnetv2) uses TensorFlow, but in this project we use a PyTorch-based implementation, using the [timm](https://github.com/rwightman/pytorch-image-models) library. Some of the other PyTorch-based implementations in the literature are as follow:
1) https://github.com/hankyul2/EfficientNetV2-pytorch

2) https://github.com/jahongir7174/EfficientNetV2

3) https://github.com/d-li14/efficientnetv2.pytorch



## Literature Review
Training efficiency is one of the important concerns in the deep learning area, specifically image classification tasks. Earliest works such as [DenseNet](https://arxiv.org/abs/1608.06993) and [EfficientNet](https://arxiv.org/abs/1905.11946) focus mainly on parameter efficiency, with the cost of slow training, while on the other side, works such as [RegNet](https://arxiv.org/abs/2003.13678) and [ResNeSt](https://arxiv.org/abs/2004.08955) focus on improving the training speed, with the cost of more parameters. 

EfficientNetV2 is mainly inspired by EfficientNet, and addresses the mentioned limitations of the literature works. They leverage a combination of training-aware Neural Architecture Search (NAS) and scaling, to jointly improve training speed and parameter efficiency. Moreover, they propose a variation of progressive learning, which adaptively adjusts the regularization and the image size, which leads to speeding up the training without losing the accuracy. EfficientNetV2 trains up to 4x faster and is up to 6.8x smaller in parameter size, than existing works. 


## Setup
The list of packages required to run the codes: 
python 3.7.7, cuda 10.1, PyTorch 1.11.0, timm 0.5.4

The file "main.ipynb" uses the pre-trained model on ImageNet and fine-tune it on CIFAR-10 and CIFAR-100 datasets. Due to the computational limitation, we have set the epochs=50 and batch_size=32, but all the set of parameters, including the regularization terms, are adjustable through the file. We compare the performance of this method with two of the literature methods: EfficientNet and [Vision Transformers](https://arxiv.org/abs/2010.11929).

The file "image_test.py" uses a pre-trained model on ImageNet to classify a sample dog picture, with printing the top-5 probablities of the classifier's outcome. 

## Results
Here are the results of fine-tuning the EfficientNetV2 model on CIFAR-10 dataset. As evaluation metrics, we use the training-loss, top-1 and top-5 accuracies of the model.

![loss-cifar10](https://user-images.githubusercontent.com/47225763/167212696-9e7b3c83-e9f1-4200-a3c6-9d0e488b249d.png)

![top1-cifar10](https://user-images.githubusercontent.com/47225763/167212700-bc96c272-b74a-4713-8d0b-b559170f433e.png)

![top5-cifar10](https://user-images.githubusercontent.com/47225763/167212708-0c7b943a-09b1-43bf-a088-4daa5b411887.png)

