# 训练模型，该模块主要是为了实现对于模型的训练，
'''
# Part1 引入相关的库函数
'''

import torch
from torch import nn
from dataset import Mnist_dataset
from VQVAE import VQVAE
import torch.utils.data as data
import matplotlib.pyplot as plt

'''
初始化一些训练参数
'''
EPOCH = 50
Mnist_dataloader = data.DataLoader(dataset=Mnist_dataset, batch_size=64, shuffle=True)

# 前向传播的模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = VQVAE(img_channel=1, encode_f1_channel=64, latent_size=64, num_emd=512)
net = net.to(device)


# 反向更新参数
lr = 1e-3
optim = torch.optim.Adam(params=net.parameters(), lr=lr)


# 定义VQVAE的损失函数，主要包含重建损失,承诺损失和向量量化的损失。
def vqvae_loss(x, rec_x, encode_z, finnel_latent, beta):
    # 首先是重建损失
    loss1 = nn.MSELoss()
    rec_loss = loss1(rec_x, x)
    # 承诺损失：编码器输出和量化后的向量之间的L2距离，更新编码器的参数
    loss2 = nn.MSELoss()
    commitment_loss = loss2(encode_z, finnel_latent.detach())

    # 向量量化损失：编码器输出和量化后的向量之间的L2距离，更新字典
    loss3 = nn.MSELoss()
    quantization_loss = loss3(encode_z.detach(), finnel_latent)
    return rec_loss + commitment_loss + beta * quantization_loss

# 可视化函数
def visualize_reconstruction(original, reconstructed):
    original = original.cpu().detach().numpy()
    reconstructed = reconstructed.cpu().detach().numpy()
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(original[0][0], cmap='gray')
    axes[0].set_title('Original Image')
    axes[1].imshow(reconstructed[0][0], cmap='gray')
    axes[1].set_title('Reconstructed Image')
    plt.show()


'''
# 开始训练
'''
# net.train() # 设置为训练模式

for epoch in range(EPOCH):
    n_iter = 0
    for batch_img, _ in Mnist_dataloader:
        batch_img = batch_img.to(device)
        # 先进行前向传播
        batch_img_pre, encode_z, finnel_latent = net(batch_img)  #

        # 计算损失
        loss_cal = vqvae_loss(rec_x=batch_img_pre, x=batch_img, encode_z=encode_z, finnel_latent=finnel_latent, beta=1)

        # 清除梯度
        optim.zero_grad()
        # 反向传播
        loss_cal.backward()
        # 更新参数
        optim.step()

        l = loss_cal.item()

        if n_iter % 100 == 0:
            print('此时的epoch为{},iter为{},loss为{}'.format(epoch, n_iter, l))

        n_iter += 1
    # 每隔一定次数保存图片
    if (epoch + 1) % 2 == 0:
        visualize_reconstruction(batch_img, batch_img_pre)

    if epoch == 20:
        # 注意pt文件是保存整个模型及其参数的，pth文件只是保存参数
        torch.save(net, 'VQVAE_eopch_{}.pt'.format(epoch))
        break
