import torch
from torchvision.transforms import Compose
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.utils import save_image

from Diffuser import Diffuser
from model import Unet


if __name__ == "__main__":
    # ハイパーパラメータの設定
    image_size = 28
    channels = 1
    batch_size = 128
    epochs = 10
    timesteps = 1000
    learning_rate = 1e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gamma = 3.0
    save_every = 1 # 何epochごとにモデルのパラメータを保存するかを決める値。

    # データセットとデータローダーの準備
    transform = Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1),
    ])

    dataset = datasets.MNIST(root='./data', download=True, transform=transform)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # モデルと最適化手法の設定
    model = Unet(dim=64, channels=channels, dim_mults=(1, 2, 4), num_labels=10)
    diffuser = Diffuser(model, timesteps=timesteps, device=device)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # モデルのトレーニング
    diffuser.train(dataloader, epochs, optimizer, save_every=save_every, save_path=f"./models/mnist/")

    # 画像の生成と保存
    samples, _ = diffuser.generate(image_size, batch_size=16, channels=channels, labels=None, gamma=gamma)
    samples = torch.tensor(samples[-1])
    samples = (samples + 1) / 2  # [-1, 1] -> [0, 1]
    samples = samples.clamp(0, 1)
    save_image(samples, f"./results/mnist/{epochs}epoch_training_generated_samples.png", nrow=4)