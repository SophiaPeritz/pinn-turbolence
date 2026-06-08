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

    def __init__(self, input_dim=3, embed_dim=64, scale=1.0):
        super().__init__()
        # B e' fisso (non trainabile): campionato da N(0, scale^2)
        B = torch.randn(input_dim, embed_dim) * scale
        self.register_buffer("B", B)

    def forward(self, x):
        # x shape: (N, input_dim)
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
        activation   = cfg.get("activation", "tanh")

        # Fourier embedding: input_dim -> 2*embed_dim
        self.embedding = FourierEmbedding(input_dim, embed_dim, fourier_scale)
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