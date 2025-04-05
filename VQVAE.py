# 该模块主要是为了实现VAE模型的，

'''
# Part1 引入相关的库函数
'''
import torch
from torch import nn
from dataset import Mnist_dataset

'''
# Part2 设计AE的类函数
'''

class ResidualBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        tmp = self.relu(x)
        tmp = self.conv1(tmp)
        tmp = self.relu(tmp)
        tmp = self.conv2(tmp)
        return x + tmp

class VQVAE(nn.Module):
    def __init__(self, img_channel, encode_f1_channel, latent_size, num_emd):
        super().__init__()
        # VQVAE的编码器，一般是先经过几个卷积，然后最后通过reshape或者view来铺平。
        self.encode = nn.Sequential(
            nn.Conv2d(in_channels=img_channel, out_channels=encode_f1_channel, kernel_size=4, stride=2, padding=1),
            # 卷积层
            nn.ReLU(),  # ReLU激活函数
            nn.Conv2d(in_channels=encode_f1_channel, out_channels=encode_f1_channel, kernel_size=4, stride=2,
                      padding=1),  # 卷积层
            nn.ReLU(),  # ReLU激活函数
            nn.Conv2d(in_channels=encode_f1_channel, out_channels=encode_f1_channel, kernel_size=3, stride=1,
                      padding=1),
            # # 再来两个Resblock保持原图像不用变的
            ResidualBlock(encode_f1_channel),
            ResidualBlock(encode_f1_channel),
        )

        self.emd_dic = nn.Embedding(num_embeddings=num_emd, embedding_dim=latent_size)
        # 进行归一化
        self.emd_dic.weight.data.uniform_(-1.0 / num_emd,
                                          1.0 / num_emd),

        self.decode = nn.Sequential(
            nn.Conv2d(latent_size, encode_f1_channel, kernel_size=3, stride=1, padding=1),  # 卷积层，用于恢复到输入通道数
            ResidualBlock(encode_f1_channel),
            ResidualBlock(encode_f1_channel),
            # nn.ReLU(),
            nn.ConvTranspose2d(encode_f1_channel, encode_f1_channel, kernel_size=4, stride=2, padding=1),  # 反卷积层
            nn.ReLU(),  # ReLU激活函数
            nn.ConvTranspose2d(encode_f1_channel, img_channel, kernel_size=4, stride=2, padding=1),  # 反卷积层
            # # 再来两个Resblock保持原图像不用变的

        )

    def forward(self, x):
        # part1 编码部分,得到编码后的向量
        encode_z = self.encode(x)  # (batch,encode_f1_channel,img_size,img_size) # (b,encode_f1_channel,7,7)

        emd_dic_data = self.emd_dic.weight.data  # (num_emd,latent_size)
        # 获取两者的维度大小，来进行下面的操作
        num_emd, latent_size = emd_dic_data.shape

        # part2 计算距离部分：然后需要利用通道计算和字典向量的距离,先扩展，便于广播计算
        '''
        # 为什么这样扩展
        # 1. 为了能够计算每张图像和每个向量之间的距离，将扩展后每张图像对应的维度和所有向量对应的维度进行对应。也就是图像batch后面的所有维度，需要和整个字典向量对应，因此
        # 2. encode_z应当会比总的字典向量的维度多一个batch维度，但是为了统一，所以需要把字典向量前面添加一个维度。
        # 3. 然后要保证每个图像要和每个向量进行对应，所以num_emd,也应该比一张图像的(C,H,W)前面一个维度,所以图像前面要插入一个维度。最终形成下面的局面，五维向量。
        '''

        encode_z_broadcast = encode_z.unsqueeze(1)  # (b, 1, encode_f1_channel, 7, 7)
        emd_dic_data_broad_cast = emd_dic_data.reshape(1, num_emd, latent_size, 1, 1)  # (1, num_emd, latent_size, 1, 1)

        # 开始计算距离
        dist = torch.sum((encode_z_broadcast - emd_dic_data_broad_cast) ** 2, dim=2)  # 从每张图像对应的维度开始进行距离结果(b, num_emd, 7, 7)

        # 取出num_emd个距离中最小的那个# (batch,img_size*img_size)
        min_dist_index = torch.argmin(dist, dim=1)  # (b, 7, 7)
        finnel_latent = self.emd_dic(min_dist_index).permute(0, 3, 2, 1)  # (b, 7, 7, 32) - > (b, 32, 7, 7)

        # part3 变换得到最终的潜层向量
        # (batch, latent_size, img_size, img_size)

        decoder_input = encode_z + (finnel_latent-encode_z).detach()


        # part4 解码部分
        x_hat = self.decode(decoder_input)  # (batch,channel*img_size*img_size)

        return x_hat, encode_z, finnel_latent  # 后面俩个用于计算损失


'''
# 开始测试
'''
if __name__ == '__main__':
    img, label = Mnist_dataset[0]
    vqvae = VQVAE(img_channel=1, encode_f1_channel=64, num_emd=512, latent_size=64)
    result, encode_z, finnel_latent = vqvae(img.unsqueeze(0))
    print(result.size())
