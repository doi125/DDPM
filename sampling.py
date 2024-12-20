import torch
import matplotlib.pyplot as plt

from Diffuser import Diffuser
from model import Unet


def show_images(images, labels=None, rows=2, cols=10):
    """
    画像と対応するラベルをプロットする関数。グレースケール画像にも対応。

    Args:
        images (torch.Tensor): 画像のテンソル。形状は (batch_size, channels, height, width)。
        labels (torch.Tensor, optional): 各画像に対応するラベル。形状は (batch_size,)。
        rows (int, optional): プロットする行数。デフォルトは 2。
        cols (int, optional): プロットする列数。デフォルトは 10。
    """
    batch_size = images.shape[0]
    total = rows * cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    fig.subplots_adjust(wspace=0.1, hspace=0.3)

    for i in range(total):
        ax = axes[i // cols, i % cols] if rows > 1 else axes[i]
        if i < batch_size:
            img = images[i]
            # チャンネル数が1の場合（グレースケール）
            if img.shape[0] == 1:
                img = img.squeeze(0)  # (1, H, W) -> (H, W)
                ax.imshow(img.cpu().numpy(), cmap='gray')
            else:
                img = img.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
                ax.imshow(img.cpu().numpy())
            if labels is not None:
                ax.set_title(str(labels[i].item()), fontsize=8)
            ax.axis('off')
        else:
            ax.axis('off')  # 不足分のサブプロットは非表示に

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_size = 28
    channels = 1
    batch_size = 16
    timesteps = 1000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gamma = 3.0


    model_path = "./models/mnist/model_epoch_10.pt"

    model = Unet(dim=64, channels=channels, dim_mults=(1, 2, 4), num_labels=10)

    model.load_state_dict(torch.load(model_path))

    diffuser = Diffuser(model, timesteps=timesteps, device=device)

    #labels = torch.full((32,), 2, device=device) # labelを自由に設定してください。labels=Noneの場合、diffuser内でlabelが勝手にランダムな値に設定されます。
    samples, labels = diffuser.generate(image_size, batch_size=batch_size, channels=channels, labels=None, gamma=gamma)
    samples = torch.tensor(samples[-1])
    samples = (samples + 1) / 2  # [-1, 1] -> [0, 1]
    samples = samples.clamp(0, 1)
    show_images(samples,labels, rows=2, cols=8)

