# 该模块主要是为了实现数据集的下载，主要是Mnist数据集

'''
# Part1 引入相关的库函数
'''
import torch
from torch.utils import data
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

'''
# Part2 获取数据集的转换操作，以及数据集的获取
'''

transform_action=transforms.Compose([
    transforms.ToTensor()
])

Mnist_dataset=torchvision.datasets.MNIST(root='./Mnist',train=True,transform=transform_action,download=True)


'''
# 开始测试
'''

if __name__=='__main__':
    imag,label=Mnist_dataset[0]
    plt.figure(figsize=(45,45))
    plt.imshow(imag.permute(2,1,0))
    plt.show()