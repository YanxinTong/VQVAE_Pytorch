import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


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

    def __init__(self, input_dim, dim, n_embedding):  # input_dim:1   dim:32   n_embedding:32
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(input_dim, dim, 4, 2, 1),
                                     nn.ReLU(), nn.Conv2d(dim, dim, 4, 2, 1),
                                     nn.ReLU(), nn.Conv2d(dim, dim, 3, 1, 1),
                                     ResidualBlock(dim), ResidualBlock(dim))
        self.vq_embedding = nn.Embedding(n_embedding, dim)
        self.vq_embedding.weight.data.uniform_(-1.0 / n_embedding,
                                               1.0 / n_embedding)
        self.decoder = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            ResidualBlock(dim), ResidualBlock(dim),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1))


    def forward(self, x):  # [b,1,28,28]
        # encode
        ze = self.encoder(x)  # [b,32,7,7]

        # ze: [N, C, H, W]
        # embedding [K, C]
        embedding = self.vq_embedding.weight.data  # [32,32]

        N, C, H, W = ze.shape  # [b,32,7,7]
        K, _ = embedding.shape  # k=32

        embedding_broadcast = embedding.reshape(1, K, C, 1, 1)  # [1, 32, 32, 1, 1]

        ze_broadcast = ze.reshape(N, 1, C, H, W)  # [b, 1, 32, 7, 7]

        distance = torch.sum((embedding_broadcast - ze_broadcast) ** 2, 2)  # [b, 32, 7, 7]

        nearest_neighbor = torch.argmin(distance, 1)  # [b, 7, 7]

        # make C to the second dim #self.vq_embedding(nearest_neighbor)[b, 7, 7, 32]

        zq = self.vq_embedding(nearest_neighbor).permute(0, 3, 1, 2)  # [b, 32, 7, 7]

        # stop gradient
        decoder_input = ze + (zq - ze).detach()

        # decode
        x_hat = self.decoder(decoder_input)  # [32,1,28,28]
        return x_hat, ze, zq


# 超参数
batch_size = 128
epochs = 10
learning_rate = 1e-3
img_channel = 1
embedding_dim = 64
num_embeddings = 512

# 数据预处理和加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

train_dataset = datasets.MNIST('./Mnist', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 模型实例化
model = VQVAE(img_channel, embedding_dim, num_embeddings)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# 损失函数与优化器
def vqvae_loss(x, rec_x, encode_z, finnel_latent, beta):
    # 首先是重建损失
    loss1 = nn.MSELoss()
    rec_loss = loss1(rec_x, x)
    # 承诺损失：量化后的向量和编码器输出的向量之间的L2距离,更新dict
    loss2 = nn.MSELoss()
    commitment_loss = loss2(encode_z.detach(), finnel_latent)

    # 向量量化损失：量化后的向量和编码器输出的离散嵌入向量之间的L2距离,更新编码器参数
    loss3 = nn.MSELoss()
    quantization_loss = loss3(encode_z, finnel_latent.detach())
    return rec_loss + commitment_loss + beta * quantization_loss


optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# 训练模型
def train():
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for data, _ in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            # 前向传播
            recon_data, ze, zq = model(data)
            loss = vqvae_loss(data, recon_data, ze, zq, beta=0.5)

            # 反向传播与优化
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader)}")

        # 每隔一定次数保存图片
        if (epoch + 1) % 2 == 0:
            visualize_reconstruction(data, recon_data)


# 可视化重构结果
def visualize_reconstruction(original, reconstructed):
    original = original.cpu().detach().numpy()
    reconstructed = reconstructed.cpu().detach().numpy()
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(original[0][0], cmap='gray')
    axes[0].set_title('Original Image')
    axes[1].imshow(reconstructed[0][0], cmap='gray')
    axes[1].set_title('Reconstructed Image')
    plt.show()


# 开始训练
train()
