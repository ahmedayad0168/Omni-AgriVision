import math
from pathlib import Path
from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal embeddings for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.bn1(self.conv1(x)))
        # Add time embedding
        time_proj = self.act(self.time_mlp(t_emb))
        h = h + time_proj[(..., ) + (None, ) * 2]
        h = self.act(self.bn2(self.conv2(h)))
        return h + self.res_conv(x)


class UNet(nn.Module):
    """Conditional UNet predicting noise in the diffusion process."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_classes: int = 10,
        base_channels: int = 32,
        time_emb_dim: int = 128,
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(inplace=True),
        )
        self.class_emb = nn.Embedding(num_classes, time_emb_dim)

        # Downsample
        self.inc = Block(in_channels, base_channels, time_emb_dim)
        self.down1 = Block(base_channels, base_channels * 2, time_emb_dim)
        self.down2 = Block(base_channels * 2, base_channels * 4, time_emb_dim)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bot1 = Block(base_channels * 4, base_channels * 8, time_emb_dim)
        self.bot2 = Block(base_channels * 8, base_channels * 4, time_emb_dim)

        # Upsample
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_conv1 = Block(base_channels * 4 + base_channels * 2, base_channels * 2, time_emb_dim)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.up_conv2 = Block(base_channels * 2 + base_channels, base_channels, time_emb_dim)

        self.out_conv = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, class_labels: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(timestep) + self.class_emb(class_labels)

        # Down
        x1 = self.inc(x, t_emb)
        x2 = self.down1(self.pool(x1), t_emb)
        x3 = self.down2(self.pool(x2), t_emb)

        # Middle
        m = self.bot1(x3, t_emb)
        m = self.bot2(m, t_emb)

        # Up
        u1 = self.up1(m)
        u1 = torch.cat([u1, x2], dim=1)
        u1 = self.up_conv1(u1, t_emb)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, x1], dim=1)
        u2 = self.up_conv2(u2, t_emb)

        return self.out_conv(u2)


class DiffusionModel(nn.Module):
    """
    Complete DDPM (Denoising Diffusion Probabilistic Model) for agricultural crop lesion synthesis.
    """

    def __init__(
        self,
        img_channels: int = 3,
        num_classes: int = 10,
        timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        image_size: int = 32,
        device: str = "auto",
    ):
        super().__init__()
        self.timesteps = timesteps
        self.img_channels = img_channels
        self.num_classes = num_classes
        self.image_size = image_size
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )

        self.model = UNet(
            in_channels=img_channels,
            out_channels=img_channels,
            num_classes=num_classes,
            base_channels=32,
        )

        # Precompute diffusion schedule
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )

        self.to(self.device)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward diffuse: sample q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha = self.sqrt_alphas_cumprod[t][(..., ) + (None, ) * 3]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][(..., ) + (None, ) * 3]
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

    def compute_loss(
        self, x_start: torch.Tensor, labels: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute training loss L_simple."""
        x_start = x_start.to(self.device)
        labels = labels.to(self.device)
        batch_size = x_start.size(0)

        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self.model(x_noisy, t, labels)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: int, labels: torch.Tensor) -> torch.Tensor:
        """Sample x_{t-1} from p_theta(x_{t-1} | x_t)."""
        t_batch = torch.full((x.size(0),), t, device=self.device, dtype=torch.long)
        pred_noise = self.model(x, t_batch, labels)

        beta_t = self.betas[t]
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]

        # Model mean
        model_mean = sqrt_recip_alpha_t * (
            x - beta_t / sqrt_one_minus_alpha_cumprod_t * pred_noise
        )

        if t == 0:
            return model_mean
        else:
            posterior_var = self.posterior_variance[t]
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_var) * noise

    @torch.no_grad()
    def sample(
        self,
        num_samples: int = 4,
        class_id: Optional[int] = None,
        device: Optional[str] = None,
    ) -> torch.Tensor:
        """Run full reverse diffusion sampling chain."""
        dev = torch.device(device) if device else self.device
        self.model.eval()

        if class_id is not None:
            labels = torch.full((num_samples,), class_id, dtype=torch.long, device=dev)
        else:
            labels = torch.randint(0, self.num_classes, (num_samples,), device=dev)

        img = torch.randn(
            (num_samples, self.img_channels, self.image_size, self.image_size),
            device=dev,
        )

        for t in reversed(range(self.timesteps)):
            img = self.p_sample(img, t, labels)

        # Denormalize [-1, 1] to [0, 1]
        return ((img + 1.0) / 2.0).clamp(0.0, 1.0)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "timesteps": self.timesteps,
            "img_channels": self.img_channels,
            "num_classes": self.num_classes,
            "image_size": self.image_size,
        }, path)
        logger.info(f"Saved DiffusionModel weights to {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded DiffusionModel weights from {path}")