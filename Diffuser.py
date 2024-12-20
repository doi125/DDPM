import math
import numpy as np

from tqdm.auto import tqdm

import torch
import torch.nn.functional as F

from utils import extract

# スケジュールの定義
def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(
        ((x / timesteps) + s) / (1 + s) * math.pi * 0.5
    ) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (
        alphas_cumprod[1:] / alphas_cumprod[:-1]
    )
    return torch.clip(betas, 0.0001, 0.9999)

# 損失関数とサンプリング関数
def q_sample(x_start, t, noise, sqrt_alphas_cumprod_t, sqrt_one_minus_alphas_cumprod_t):
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

def p_losses(denoise_model, x_start, t, labels, noise, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, loss_type="l1"):
    x_noisy = q_sample(
        x_start=x_start,
        t=t,
        noise=noise,
        sqrt_alphas_cumprod_t=extract(sqrt_alphas_cumprod, t, x_start.shape),
        sqrt_one_minus_alphas_cumprod_t=extract(
            sqrt_one_minus_alphas_cumprod, t, x_start.shape
        ),
    )
    predicted_noise = denoise_model(x_noisy, t, labels=labels)
    if loss_type == "l1":
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == "l2":
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()
    return loss

@torch.no_grad()
def p_sample(
    model,
    x,
    t,
    labels,
    t_index,
    betas,
    sqrt_one_minus_alphas_cumprod,
    sqrt_recip_alphas,
    posterior_variance,
    gamma,
):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    eps_cond = model(x, t, labels=labels)
    eps_uncond = model(x, t)
    eps = eps_uncond + gamma * (eps_cond - eps_uncond)

    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * eps / sqrt_one_minus_alphas_cumprod_t
    )
    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def p_sample_loop(
    model,
    labels,
    shape,
    timesteps,
    betas,
    sqrt_one_minus_alphas_cumprod,
    sqrt_recip_alphas,
    posterior_variance,
    gamma,
):
    device = next(model.parameters()).device
    b = shape[0]
    img = torch.randn(shape, device=device)
    imgs = []
    if labels is None:
        labels = torch.randint(0,10,(b, ), device=device)

    for i in tqdm(
        reversed(range(0, timesteps)),
        desc="sampling loop time step",
        total=timesteps,
    ):
        img = p_sample(
            model,
            img,
            torch.full((b,), i, device=device, dtype=torch.long),
            labels,
            i,
            betas,
            sqrt_one_minus_alphas_cumprod,
            sqrt_recip_alphas,
            posterior_variance,
            gamma,
        )
        imgs.append(img.cpu().numpy())
    return imgs, labels

@torch.no_grad()
def sample(model, labels, image_size, batch_size, channels, **kwargs):
    return p_sample_loop(
        model,
        labels,
        shape=(batch_size, channels, image_size, image_size),
        **kwargs
    )

# Diffuserクラスの定義
class Diffuser:
    def __init__(
        self,
        model,
        timesteps=1000,
        beta_schedule="linear",
        loss_type="huber",
        device=None,
    ):
        self.model = model.to(device)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.timesteps = timesteps
        self.loss_type = loss_type

        # ベータスケジュールの選択
        if beta_schedule == "linear":
            self.betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            self.betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError("Unknown beta schedule")

        self.betas = self.betas.to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.0
        )
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1.0 - self.alphas_cumprod
        )
        self.posterior_variance = (
            self.betas
            * (1.0 - self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )

    def train(self, dataloader, epochs, optimizer, save_every=None, save_path=None):
        self.model.train()
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for images, labels in tqdm(dataloader):
                optimizer.zero_grad()
                batch = images.to(self.device)
                labels = labels.to(self.device)
                batch_size = batch.size(0)
                t = torch.randint(
                    0, self.timesteps, (batch_size,), device=self.device
                ).long()
                noise = torch.randn_like(batch)

                if np.random.random() < 0.1:
                    labels = None

                loss = p_losses(
                    self.model,
                    batch,
                    t,
                    labels,
                    noise,
                    self.sqrt_alphas_cumprod,
                    self.sqrt_one_minus_alphas_cumprod,
                    loss_type=self.loss_type,
                )
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}")

            # モデルの保存
            if save_every and (epoch + 1) % save_every == 0 and save_path:
                torch.save(self.model.state_dict(), f"{save_path}model_epoch_{epoch + 1}.pt")

        return losses

    @torch.no_grad()
    def generate(self, image_size, batch_size=16, channels=3,labels=None, gamma=0.2):
        self.model.eval()
        samples, labels = sample(
            model=self.model,
            labels=labels,
            image_size=image_size,
            batch_size=batch_size,
            channels=channels,
            timesteps=self.timesteps,
            betas=self.betas,
            sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod,
            sqrt_recip_alphas=self.sqrt_recip_alphas,
            posterior_variance=self.posterior_variance,
            gamma=gamma,
        )
        return samples, labels