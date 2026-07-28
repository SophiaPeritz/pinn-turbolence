"""
Baseline architecture: MLP with Fourier embedding.
Layer 1 uses the standard setup only;
"""

import torch
import torch.nn as nn


class FourierEmbedding(nn.Module):
    """
    Random Fourier Features embedding.
    Maps the input coordinates to a higher-dimensional space to reduce
    spectral bias, i.e. the tendency of the model to struggle with high
    frequencies.

    Input:  [t, x, y]  shape (N, 3)
    Output: [cos(Bx), sin(Bx)]  shape (N, 2*m)
    """

    def __init__(self, input_dim=3, embed_dim=64, scale=1.0):
        super().__init__()
        # Spatial periodicity is always enforced as a hard constraint:
        # [x, y] -> [cos(2πx), sin(2πx), cos(2πy), sin(2πy)].
        # Random Fourier features are applied only to time.
        if input_dim != 3:
            raise ValueError("FourierEmbedding expects input_dim=3 for [t, x, y]")
        B_t = torch.randn(1, embed_dim) * scale
        self.register_buffer("B_t", B_t)

    def forward(self, x):
        # Expected input shape: (N, input_dim) with columns [t, x, y].
        # Temporal embedding via random Fourier features.
        t = x[:, 0:1]  # (N, 1)
        proj_t = t @ self.B_t  # (N, embed_dim)
        emb_t = torch.cat([torch.cos(proj_t), torch.sin(proj_t)], dim=-1)

        # Spatial periodic hard-constraint mapping.
        x_sp = x[:, 1:2]
        y_sp = x[:, 2:3]
        two_pi = 2.0 * torch.pi
        sp_map = torch.cat([
            torch.cos(two_pi * x_sp), torch.sin(two_pi * x_sp),
            torch.cos(two_pi * y_sp), torch.sin(two_pi * y_sp)
        ], dim=-1)  # (N, 4)

        return torch.cat([emb_t, sp_map], dim=-1)


class PINN_MLP(nn.Module):
    """
    Baseline MLP for the 2D Navier-Stokes system.

    Input:  spatiotemporal coordinates [t, x, y]
    Output: physical fields [u, v, p]

    Architecture:
        Fourier embedding -> hidden layers -> output layer
    """

    def __init__(self, cfg):
        super().__init__()

        input_dim    = cfg.get("input_dim", 3)
        hidden_dim   = cfg.get("hidden_dim", 256)
        n_layers     = cfg.get("n_layers", 4)
        output_dim   = cfg.get("output_dim", 3)   # u, v, p
        embed_dim    = cfg.get("embed_dim", 64)
        fourier_scale = cfg.get("fourier_scale", 1.0)
        periodic_spatial = cfg.get("periodic_spatial", True)
        activation   = cfg.get("activation", "tanh")

        if not periodic_spatial:
            raise ValueError("periodic_spatial=false is no longer supported")

        # Fourier / periodic embedding.
        self.embedding = FourierEmbedding(input_dim, embed_dim, fourier_scale)

        # emb_t -> 2*embed_dim, sp_map -> 4, so the total dimension is
        # 2*embed_dim + 4.
        first_layer_dim = 2 * embed_dim + 4

        # Activation function.
        act_map = {
            "tanh":  nn.Tanh,
            "swish": nn.SiLU,
            "gelu":  nn.GELU,
        }
        Act = act_map.get(activation, nn.Tanh)

        # Hidden layers.
        layers = []
        layers.append(nn.Linear(first_layer_dim, hidden_dim))
        layers.append(Act())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(Act())
        self.hidden = nn.Sequential(*layers)

        # Output layer (no final activation).
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Xavier initialization for stability.
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x shape: (N, 3) -> [t, x, y].
        z = self.embedding(x)
        z = self.hidden(z)
        return self.output_layer(z)


def build_network(cfg: dict) -> PINN_MLP:
    """Factory function that builds the network from the configuration."""
    return PINN_MLP(cfg)