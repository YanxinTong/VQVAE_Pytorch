import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 超参数
batch_size = 128
epochs = 10
learning_rate = 1e-3
latent_dim = 64
embedding_dim = 64
num_embeddings = 512

# 数据预处理和加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

train_dataset = datasets.MNIST('./Mnist', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# VQ-VAE模型定义
class VQVAE(nn.Module):
    def __init__(self, latent_dim, embedding_dim, num_embeddings):
        super(VQVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),  # 14x14
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 7x7
            nn.ReLU(),
            nn.Conv2d(128, latent_dim, kernel_size=3, stride=1, padding=1),  # 7x7
        )

        self.quantizer = nn.Embedding(num_embeddings, embedding_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, kernel_size=4, stride=2, padding=1),  # 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 28x28
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),  # 28x28
        )

    def encode(self, x):
        z = self.encoder(x)
        z_flattened = z.view(z.size(0), z.size(1), -1).permute(0, 2, 1)
        z_quantized = self.quantize(z_flattened)
        return z_quantized.view(z.size(0), z.size(2), z.size(3), -1).permute(0, 3, 1, 2)

    def quantize(self, z):
        dist = torch.cdist(z, self.quantizer.weight)
        encoding_indices = dist.argmin(dim=-1)
        quantized = self.quantizer(encoding_indices)

        # Straight-through estimator(STE操作)，允许梯度通过非连续操作（如量化或离散化步骤）的技术
        z = z + (quantized - z).detach()
        return z

    def forward(self, x):
        z_quantized = self.encode(x)
        return self.decoder(z_quantized)


# 模型实例化
model = VQVAE(latent_dim, embedding_dim, num_embeddings)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 损失函数与优化器
reconstruction_loss_fn = nn.MSELoss()
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
            recon_data = model(data)
            loss = reconstruction_loss_fn(recon_data, data)

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
