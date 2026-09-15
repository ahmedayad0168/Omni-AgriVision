import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Generator(nn.Module):
    """Conditional Generator producing synthetic agricultural plant/disease patches."""

    def __init__(self, latent_dim: int = 100, num_classes: int = 10, img_channels: int = 3, feature_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.label_emb = nn.Embedding(num_classes, num_classes)
        in_dim = latent_dim + num_classes

        self.init_proj = nn.Sequential(
            nn.Linear(in_dim, feature_dim * 8 * 4 * 4),
            nn.BatchNorm1d(feature_dim * 8 * 4 * 4),
            nn.ReLU(True),
        )

        self.conv_blocks = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(feature_dim * 8, feature_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim * 4),
            nn.ReLU(True),
            # 8x8 -> 16x16
            nn.ConvTranspose2d(feature_dim * 4, feature_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim * 2),
            nn.ReLU(True),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(feature_dim * 2, feature_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(True),
            # 32x32 -> 64x64
            nn.ConvTranspose2d(feature_dim, img_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        c = self.label_emb(labels)
        x = torch.cat([z, c], dim=1)
        x = self.init_proj(x)
        x = x.view(x.size(0), -1, 4, 4)
        return self.conv_blocks(x)


class Discriminator(nn.Module):
    """Conditional Discriminator distinguishing real vs synthetic plant patches."""

    def __init__(self, num_classes: int = 10, img_channels: int = 3, feature_dim: int = 64):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, 64 * 64)

        self.conv_blocks = nn.Sequential(
            # (3 + 1 channels, 64x64) -> 32x32
            nn.Conv2d(img_channels + 1, feature_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 32x32 -> 16x16
            nn.Conv2d(feature_dim, feature_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 16x16 -> 8x8
            nn.Conv2d(feature_dim * 2, feature_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 8x8 -> 4x4
            nn.Conv2d(feature_dim * 4, feature_dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(feature_dim * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        c = self.label_emb(labels).view(labels.size(0), 1, 64, 64)
        x = torch.cat([img, c], dim=1)
        feat = self.conv_blocks(x)
        out = self.classifier(feat)
        return out.view(img.size(0), 1)


class ConditionalGAN(nn.Module):
    """Complete Conditional GAN for agricultural lesion / plant data synthesis."""

    def __init__(
        self,
        latent_dim: int = 100,
        num_classes: int = 10,
        img_channels: int = 3,
        feature_dim: int = 64,
        device: str = "auto",
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_channels = img_channels
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )

        self.generator = Generator(latent_dim, num_classes, img_channels, feature_dim)
        self.discriminator = Discriminator(num_classes, img_channels, feature_dim)
        self.to(self.device)

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Forward pass through generator."""
        return self.generator(z, labels)

    @torch.no_grad()
    def generate(
        self,
        num_samples: int = 8,
        class_id: Optional[int] = None,
        device: Optional[str] = None,
    ) -> torch.Tensor:
        """Generate synthetic images for a given class or random classes."""
        dev = torch.device(device) if device else self.device
        self.generator.eval()

        z = torch.randn(num_samples, self.latent_dim, device=dev)
        if class_id is not None:
            labels = torch.full((num_samples,), class_id, dtype=torch.long, device=dev)
        else:
            labels = torch.randint(0, self.num_classes, (num_samples,), device=dev)

        gen_imgs = self.generator(z, labels)
        # Denormalize [-1, 1] to [0, 1]
        return ((gen_imgs + 1.0) / 2.0).clamp(0.0, 1.0)

    def train_step(
        self,
        real_images: torch.Tensor,
        labels: torch.Tensor,
        optimizer_g: torch.optim.Optimizer,
        optimizer_d: torch.optim.Optimizer,
        criterion: nn.Module = None,
    ) -> Dict[str, float]:
        """Perform a single training step for Generator and Discriminator."""
        if criterion is None:
            criterion = nn.BCELoss()

        batch_size = real_images.size(0)
        real_images = real_images.to(self.device)
        labels = labels.to(self.device)

        real_target = torch.ones(batch_size, 1, device=self.device)
        fake_target = torch.zeros(batch_size, 1, device=self.device)

        # Train Discriminator
        optimizer_d.zero_grad()
        d_real = self.discriminator(real_images, labels)
        loss_d_real = criterion(d_real, real_target)

        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_images = self.generator(z, labels)
        d_fake = self.discriminator(fake_images.detach(), labels)
        loss_d_fake = criterion(d_fake, fake_target)

        loss_d = (loss_d_real + loss_d_fake) / 2
        loss_d.backward()
        optimizer_d.step()

        # Train Generator
        optimizer_g.zero_grad()
        d_fake_for_g = self.discriminator(fake_images, labels)
        loss_g = criterion(d_fake_for_g, real_target)
        loss_g.backward()
        optimizer_g.step()

        return {
            "loss_d": float(loss_d.item()),
            "loss_g": float(loss_g.item()),
            "d_real": float(d_real.detach().mean().item()),
            "d_fake": float(d_fake.detach().mean().item()),
        }

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "latent_dim": self.latent_dim,
            "num_classes": self.num_classes,
        }, path)
        logger.info(f"Saved ConditionalGAN weights to {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(ckpt["generator"])
        self.discriminator.load_state_dict(ckpt["discriminator"])
        logger.info(f"Loaded ConditionalGAN weights from {path}")