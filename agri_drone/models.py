from __future__ import annotations

import os

import torch
import torch.nn as nn


def build_detector(path: str, device: str):
    from ultralytics import YOLO
    _require(path)
    return YOLO(path).to(device)


def build_classifier(path: str):
    from ultralytics import YOLO
    _require(path)
    return YOLO(path)


def build_leaf_seg(path: str, device: torch.device):
    import segmentation_models_pytorch as smp
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b3", encoder_weights=None, in_channels=3, classes=1
    )
    return _load_seg(model, path, device)


def build_disease_seg(path: str, device: torch.device):
    import segmentation_models_pytorch as smp
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1
    )
    return _load_seg(model, path, device)


def _load_seg(model: nn.Module, path: str, device: torch.device) -> nn.Module:
    _require(path)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


class ConditionalGenerator(nn.Module):
    def __init__(self, num_classes=11, latent_dim=100, embed_size=100, channels_img=3):
        super().__init__()
        self.embed = nn.Embedding(num_classes, embed_size)
        input_dim = latent_dim + embed_size

        self.net = nn.Sequential(
            
            nn.ConvTranspose2d(input_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, channels_img, 4, 2, 1, bias=False),
            nn.Tanh()           # output: 3 x 64 x 64 in [-1,1]
        )

    def forward(self, noise, labels):
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
        x = torch.cat([noise, embedding], dim=1)
        return self.net(x)


def build_generator(cfg, device: torch.device) -> ConditionalGenerator:
    model = ConditionalGenerator(
        num_classes=cfg.gan_num_classes,
        latent_dim=cfg.gan_latent_dim,
        embed_size=cfg.gan_embed_size,
        channels_img=cfg.image_channels,
    )
    _require(cfg.gan_path)
    model.load_state_dict(torch.load(cfg.gan_path, map_location=device))
    return model.to(device).eval()


class VAEEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 4, 2, 1)   # 128x128 -> 64x64
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)            # 64x64 -> 32x32
        self.conv3 = nn.Conv2d(64, 128, 4, 2, 1)           # 32x32 -> 16x16
        self.conv4 = nn.Conv2d(128, 256, 4, 2, 1)          # 16x16 -> 8x8
        self.relu = nn.ReLU(True)
        self.fc_mu = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(256 * 8 * 8, latent_dim)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = x.flatten(1)
        return self.fc_mu(x), self.fc_logvar(x)


class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=128, out_channels=3):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 8 * 8)
        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)  
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)   
        self.deconv3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)     
        self.deconv4 = nn.ConvTranspose2d(32, out_channels, 4, 2, 1) 
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        x = self.fc(z).view(-1, 256, 8, 8)
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.relu(self.deconv3(x))
        x = self.sigmoid(self.deconv4(x))
        return x


class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = VAEEncoder(latent_dim=latent_dim)
        self.decoder = VAEDecoder(latent_dim=latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def build_vae(cfg, device: torch.device) -> VAE:
    model = VAE(latent_dim=cfg.vae_latent_dim)
    _require(cfg.vae_path)
    model.load_state_dict(torch.load(cfg.vae_path, map_location=device))
    return model.to(device).eval()


# --------------------------------------------------------------------------------------
def _require(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required weights file not found: {path}")
