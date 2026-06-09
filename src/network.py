"""
Architettura baseline: MLP con Fourier embedding.
Strato 1 - nessuna innovazione avanzata (PirateNet verra dopo).
"""

import torch
import torch.nn as nn
import numpy as np


class FourierEmbedding(nn.Module):
    """
    Random Fourier Features embedding.
    Trasforma le coordinate di input in uno spazio ad alta dimensione
    per mitigare lo spectral bias (il modello fatica con frequenze alte).

    Input:  [t, x, y]  shape (N, 3)
    Output: [cos(Bx), sin(Bx)]  shape (N, 2*m)
    """

    def __init__(self, input_dim=3, embed_dim=64, scale=1.0, periodic_spatial=False):
        super().__init__()
        # Optionally enforce periodicity on spatial dimensions (x,y)
        # If periodic_spatial is True, we will replace x,y with their
        # periodic mapping [cos(2πx), sin(2πx), cos(2πy), sin(2πy)] and
        # only apply random Fourier features to the temporal component.
        self.periodic_spatial = periodic_spatial

        if self.periodic_spatial:
            # Expect input_dim == 3 (t, x, y). We'll create B only for t (1 dim)
            B_t = torch.randn(1, embed_dim) * scale
            self.register_buffer("B_t", B_t)
        else:
            # B e' fisso (non trainabile): campionato da N(0, scale^2)
            B = torch.randn(input_dim, embed_dim) * scale
            self.register_buffer("B", B)

    def forward(self, x):
        # x shape: (N, input_dim) expected [t, x, y]
        if self.periodic_spatial:
            # temporal embedding via RFF
            t = x[:, 0:1]  # (N,1)
            proj_t = t @ self.B_t  # (N, embed_dim)
            emb_t = torch.cat([torch.cos(proj_t), torch.sin(proj_t)], dim=-1)

            # spatial periodic hard-constraint mapping
            x_sp = x[:, 1:2]
            y_sp = x[:, 2:3]
            two_pi = 2.0 * torch.pi
            sp_map = torch.cat([
                torch.cos(two_pi * x_sp), torch.sin(two_pi * x_sp),
                torch.cos(two_pi * y_sp), torch.sin(two_pi * y_sp)
            ], dim=-1)  # (N,4)

            return torch.cat([emb_t, sp_map], dim=-1)

        else:
            proj = x @ self.B  # (N, embed_dim)
            return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)  # (N, 2*embed_dim)


class PINN_MLP(nn.Module):
    """
    MLP baseline per Navier-Stokes 2D.

    Input:  coordinate spatiotemporali [t, x, y]
    Output: campi fisici [u, v, p]

    Architettura:
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
        periodic_spatial = cfg.get("periodic_spatial", False)
        activation   = cfg.get("activation", "tanh")

        # Fourier / periodic embedding
        self.embedding = FourierEmbedding(input_dim, embed_dim, fourier_scale,
                                          periodic_spatial=periodic_spatial)

        if periodic_spatial:
            # emb_t -> 2*embed_dim, sp_map -> 4  => total dim = 2*embed_dim + 4
            first_layer_dim = 2 * embed_dim + 4
        else:
            # standard RFF embedding: input_dim -> 2*embed_dim
            first_layer_dim = 2 * embed_dim

        # Funzione di attivazione
        act_map = {
            "tanh":  nn.Tanh,
            "swish": nn.SiLU,
            "gelu":  nn.GELU,
        }
        Act = act_map.get(activation, nn.Tanh)

        # Hidden layers
        layers = []
        layers.append(nn.Linear(first_layer_dim, hidden_dim))
        layers.append(Act())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(Act())
        self.hidden = nn.Sequential(*layers)

        # Output layer (nessuna attivazione finale)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Inizializzazione Xavier per stabilita
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x shape: (N, 3)  ->  [t, x, y]
        z = self.embedding(x)
        z = self.hidden(z)
        return self.output_layer(z)


def build_network(cfg: dict) -> PINN_MLP:
    """Factory function — costruisce la rete dalla config."""
    return PINN_MLP(cfg)