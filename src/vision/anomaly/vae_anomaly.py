import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), 
            nn.Linear(512, 256), nn.ReLU(), 
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.mu = nn.Linear(128, latent_dim)
        self.log_var = nn.Linear(128, latent_dim)
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, input_dim), nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.log_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var


class AnomalyDetector:
    def __init__(self, model_path: Optional[str] = None, input_size: int = 224,
                  latent_dim: int = 128, anomaly_threshold: float = 0.85, device: str = "auto"):
        self.input_dim = input_size * input_size * 3
        self.latent_dim = latent_dim
        self.anomaly_threshold = anomaly_threshold
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = VAE(self.input_dim, latent_dim)
        if model_path and Path(model_path).exists():
            state = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(state['model_state_dict'] if 'model_state_dict' in state else state)

        self.model.to(self.device)
        self.model.eval()
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, features: np.ndarray):
        self.scaler.fit(features)
        self.fitted = True

    def detect(self, features: np.ndarray) -> Dict[str, Any]:
        if self.fitted:
            features_norm = self.scaler.transform(features.reshape(1, -1))
        else:
            features_norm = features.reshape(1, -1)

        x = torch.FloatTensor(features_norm).to(self.device)
        with torch.no_grad():
            recon, mu, log_var = self.model(x)
        mse = F.mse_loss(recon, x, reduction='mean').item()
        anomaly_score = min(1.0, mse / self.anomaly_threshold)

        return {
            'is_anomaly': anomaly_score > self.anomaly_threshold,
            'anomaly_score': anomaly_score,
            'reconstruction_error': mse,
            'threshold': self.anomaly_threshold
        }

    def train_vae(self, dataloader, epochs: int = 100, lr: float = 1e-3):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                x = batch.to(self.device)
                recon, mu, log_var = self.model(x)
                recon_loss = F.mse_loss(recon, x, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                loss = (recon_loss + kl_loss) / x.size(0)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch+1) % 10 == 0:
                logger.info(f"VAE Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")