from __future__ import annotations

import os

import torch
from torchvision.utils import save_image


def _save_each(images: torch.Tensor, out_dir: str, start_idx: int = 0, prefix: str = "synthetic") -> int:
    """Save a batch of images (values in [0, 1]) as separate files. Returns count saved."""
    os.makedirs(out_dir, exist_ok=True)
    for i in range(images.size(0)):
        save_image(images[i], os.path.join(out_dir, f"{prefix}_{start_idx + i:05d}.png"))
    return images.size(0)


@torch.no_grad()
def generate_gan_images(generator, num_classes: int, samples_per_class: int,
                        output_dir: str, latent_dim: int, device: torch.device,
                        batch_size: int = 32) -> int:
    """Generate `samples_per_class` images for each class; one file per image."""
    generator.eval()
    total = 0
    for class_idx in range(num_classes):
        class_dir = os.path.join(output_dir, f"class_{class_idx:02d}")
        done = 0
        while done < samples_per_class:
            n = min(batch_size, samples_per_class - done)
            noise = torch.randn(n, latent_dim, 1, 1, device=device)
            labels = torch.full((n,), class_idx, dtype=torch.long, device=device)
            imgs = (generator(noise, labels) * 0.5 + 0.5).clamp(0, 1)  # tanh [-1,1] -> [0,1]
            _save_each(imgs, class_dir, start_idx=done)
            done += n
        total += done
    return total


@torch.no_grad()
def generate_vae_images(vae, num_images: int, output_dir: str,
                        latent_dim: int, device: torch.device, batch_size: int = 32) -> int:
    """Generate healthy-leaf images from the VAE decoder; one file per image."""
    vae.eval()
    done = 0
    while done < num_images:
        n = min(batch_size, num_images - done)
        z = torch.randn(n, latent_dim, device=device)
        imgs = vae.decoder(z).clamp(0, 1)   # decoder already ends in Sigmoid -> [0,1]
        _save_each(imgs, output_dir, start_idx=done, prefix="healthy")
        done += n
    return done
