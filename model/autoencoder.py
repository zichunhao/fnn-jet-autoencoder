from .fnn import FNN

import sys
from pathlib import Path

project_path = Path(__file__).parent.parent
sys.path.insert(0, str(project_path))

import torch
from torch import nn
from typing import List, Optional
from utils import DEFAULT_DTYPE, DEFAULT_DEVICE


class FNNJetAutoencoder(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_particles: int,
        latent_size: int,
        hidden_sizes_encoder: Optional[List[int]] = None,
        hidden_sizes_decoder: Optional[List[int]] = None,
        dtype: torch.dtype = DEFAULT_DTYPE,
        device: torch.device = DEFAULT_DEVICE,
    ) -> None:
        super().__init__()
        self.total_size = feature_dim * num_particles

        self.latent_size = latent_size
        self.hidden_sizes_encoder = hidden_sizes_encoder
        self.hidden_sizes_decoder = hidden_sizes_decoder

        self.dtype = dtype
        self.device = device

        self.encoder = nn.Sequential(
            Reshape(-1, self.total_size),
            FNN(
                input_size=self.total_size,
                output_size=latent_size,
                hidden_sizes=hidden_sizes_encoder,
            ),
        ).to(device=device, dtype=dtype)
        self.decoder = nn.Sequential(
            FNN(
                input_size=latent_size,
                output_size=self.total_size,
                hidden_sizes=hidden_sizes_decoder,
            ),
            Reshape(-1, num_particles, feature_dim),
        ).to(device=device, dtype=dtype)

        self.__num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor, latent: bool = False) -> torch.Tensor:
        x = x.to(dtype=self.dtype, device=self.device)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        elif len(x.shape) == 3:
            pass
        else:
            raise ValueError(
                "Input shape must be "
                "(batch_size, num_particles, feature_dim) "
                "or (num_particles, feature_dim). "
                f"Got: {x.shape=}"
            )
        z = self.encoder(x)
        x = self.decoder(z)
        if latent:
            # (recons, latent)
            return x, z
        else:
            # recons
            return x

    @property
    def compression_rate(self) -> float:
        return self.latent_size / self.total_size

    def __str__(self) -> str:
        s = f"FNNJetAutoencoder(\n"
        if self.hidden_sizes_encoder is None:
            s += f"  Encoder: {self.total_size} -> {self.latent_size}\n"
        else:
            s += f"  Encoder: {self.total_size} -> {self.hidden_sizes_encoder} -> {self.latent_size}\n"

        if self.hidden_sizes_decoder is None:
            s += f"  Decoder: {self.latent_size} -> {self.total_size}\n"
        else:
            s += f"  Decoder: {self.latent_size} -> {self.hidden_sizes_decoder} -> {self.total_size}\n"

        s += f")"
        return s

    @property
    def num_learnable_params(self) -> int:
        return self.__num_params


class Reshape(nn.Module):
    def __init__(self, *args) -> None:
        """Reshape module to be compatible with torch.nn.Sequential.
        Initialized with the same arguments as torch.reshape.
        """
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(self.shape)

    def __str__(self) -> str:
        # ignore batch dimension
        return f"Reshape({self.shape[1:]})"
