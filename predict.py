# 该模块主要是为了预测推理的，输入一个图像得到一个浅层或者输入浅层得到一个图像
'''
# Part1 引入相关的模型
'''
import torch
from dataset import Mnist_dataset
import matplotlib.pyplot as plt

'''
# part2 下载模型
'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = torch.load('VQVAE_eopch_20.pt')
net.to(device)
net.eval()
data_cs = Mnist_dataset

'''
# Part3 开始测试
'''

if __name__ == '__main__':
    with torch.no_grad():
        img, label = data_cs[2]

        # 将 img 转移到正确的设备
        img = img.to(device)

        # 开始绘制初始的图像
        print('已经绘制了初始图像')
        plt.imshow(img.detach().cpu().squeeze(0).numpy(), cmap='gray')  # 转换为 NumPy 数组并显示
        plt.show()

        img = img.unsqueeze(0)  # 添加 batch 维度
        x_pre, encode_z, finnel_latent = net(img)  # 推理

        x_pre = x_pre.reshape(1, 28, 28)  # 确保形状是 (1, 28, 28)
        encode_z_detach=encode_z.detach()
        finnel_latent_detach=finnel_latent.detach()

        print('该图像的编码向量形状为{}'.format(encode_z_detach.size()))
        print('该图像查询到的编码向量形状为{}'.format(finnel_latent_detach.size()))
        print('两者之间的差距为{}'.format(torch.abs(torch.sum(encode_z_detach-finnel_latent_detach))))

        # 开始绘制结果图像
        print('已经绘制了结果图像')

        # 因为 mid_latent_predict 形状是 (1, 28, 28)，去除 batch_size 维度
        plt.imshow(x_pre.detach().cpu().squeeze(0).numpy(), cmap='gray')  # 使用灰度图的颜色映射
        plt.show()
