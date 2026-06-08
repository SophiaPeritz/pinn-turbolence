"""
Utilities: checkpoint, sampling dei punti di collocazione,
plotting e metriche fisiche.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os


# ─────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch, loss, path):
    """Salva pesi + stato optimizer su disco."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss":                 loss,
    }, path)
    print(f"[checkpoint] Salvato: {path}")


def load_checkpoint(model, optimizer, path, device="cpu"):
    """Carica checkpoint da disco."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"[checkpoint] Caricato: {path}  (epoch {ckpt['epoch']})")
    return ckpt["epoch"], ckpt["loss"]


# ─────────────────────────────────────────
# Sampling dei punti di collocazione
# ─────────────────────────────────────────

def sample_collocation_points(n_pde, t_range, domain, device):
    """
    Campiona punti casuali nel dominio spazio-temporale.

    Args:
        n_pde    : numero di punti
        t_range  : (t_min, t_max)
        domain   : (x_min, x_max, y_min, y_max)  default (0,1,0,1)
        device   : torch device

    Returns:
        x_pde : tensor shape (n_pde, 3) -> [t, x, y]
    """
    t = torch.rand(n_pde, 1) * (t_range[1] - t_range[0]) + t_range[0]
    x = torch.rand(n_pde, 1) * (domain[1] - domain[0]) + domain[0]
    y = torch.rand(n_pde, 1) * (domain[3] - domain[2]) + domain[2]
    return torch.cat([t, x, y], dim=1).to(device)


def sample_ic_points(n_ic, domain, u0_fn, device):
    """
    Campiona punti sulla condizione iniziale t=0.

    Args:
        n_ic   : numero di punti
        domain : (x_min, x_max, y_min, y_max)
        u0_fn  : funzione u0_fn(x, y) -> (u, v) tensori shape (N,1)

    Returns:
        x_ic : shape (n_ic, 3)  con t=0
        u_ic : shape (n_ic, 2)  valori [u, v]
    """
    x = torch.rand(n_ic, 1) * (domain[1] - domain[0]) + domain[0]
    y = torch.rand(n_ic, 1) * (domain[3] - domain[2]) + domain[2]
    t = torch.zeros(n_ic, 1)

    x_ic = torch.cat([t, x, y], dim=1).to(device)

    u0, v0 = u0_fn(x.to(device), y.to(device))
    u_ic = torch.cat([u0, v0], dim=1).to(device)

    return x_ic, u_ic


# ─────────────────────────────────────────
# Condizione iniziale Kolmogorov
# ─────────────────────────────────────────

def kolmogorov_ic(x, y, seed=42):
    """
    Campo di velocita iniziale casuale smussato per Kolmogorov flow.
    Usa una sovrapposizione di modi di Fourier a bassa frequenza.
    """
    torch.manual_seed(seed)
    n_modes = 4
    u = torch.zeros_like(x)
    v = torch.zeros_like(y)

    for k in range(1, n_modes + 1):
        ak = torch.randn(1).item() * 0.1
        bk = torch.randn(1).item() * 0.1
        ck = torch.randn(1).item() * 0.1
        dk = torch.randn(1).item() * 0.1
        u += ak * torch.sin(2 * torch.pi * k * x) + bk * torch.cos(2 * torch.pi * k * y)
        v += ck * torch.cos(2 * torch.pi * k * x) + dk * torch.sin(2 * torch.pi * k * y)

    return u, v


# ─────────────────────────────────────────
# Metriche fisiche
# ─────────────────────────────────────────

def kinetic_energy(u, v):
    """Energia cinetica media: KE = 0.5 * mean(u^2 + v^2)"""
    return 0.5 * torch.mean(u ** 2 + v ** 2).item()


def enstrophy(u, v, x):
    """
    Enstrofia: E = 0.5 * mean(omega^2)
    omega = dv/dx - du/dy  (vorticita 2D)
    """
    x = x.requires_grad_(True)
    # Serve il modello qui, quindi viene chiamata dall'esterno
    # Questa e' una versione semplificata che riceve omega direttamente
    raise NotImplementedError("Chiama enstrophy_from_model() con il modello.")


def enstrophy_from_model(model, x_eval, device):
    """
    Calcola enstrofia data la rete e i punti di valutazione.
    """
    x_eval = x_eval.to(device).requires_grad_(True)
    out = model(x_eval)

    u = out[:, 0:1]
    v = out[:, 1:2]

    du_dy = torch.autograd.grad(u, x_eval,
                                 grad_outputs=torch.ones_like(u),
                                 create_graph=False, retain_graph=True)[0][:, 2:3]
    dv_dx = torch.autograd.grad(v, x_eval,
                                 grad_outputs=torch.ones_like(v),
                                 create_graph=False, retain_graph=False)[0][:, 1:2]

    omega = dv_dx - du_dy
    return 0.5 * torch.mean(omega ** 2).item()


# ─────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────

def plot_loss_history(history: dict, save_path=None):
    """
    Plotta le curve di loss durante il training.

    Args:
        history : dict con chiavi 'total', 'ic', 'pde' e liste di valori
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].semilogy(history["total"], label="Total", color="black")
    axes[0].set_title("Total Loss")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(history["ic"],  label="IC loss",  color="blue")
    axes[1].semilogy(history["pde"], label="PDE loss", color="red")
    axes[1].set_title("Loss Components")
    axes[1].set_xlabel("Iteration")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[plot] Salvato: {save_path}")
    plt.show()


def plot_velocity_field(model, t_val, n_grid=64, device="cpu", save_path=None):
    """
    Plotta il campo di velocita u su una griglia 2D a tempo fisso t=t_val.
    """
    xi = torch.linspace(0, 1, n_grid)
    yi = torch.linspace(0, 1, n_grid)
    Xg, Yg = torch.meshgrid(xi, yi, indexing="ij")

    t_grid = torch.full((n_grid * n_grid, 1), t_val)
    x_flat = Xg.reshape(-1, 1)
    y_flat = Yg.reshape(-1, 1)
    pts = torch.cat([t_grid, x_flat, y_flat], dim=1).to(device)

    with torch.no_grad():
        out = model(pts)

    u = out[:, 0].reshape(n_grid, n_grid).cpu().numpy()
    v = out[:, 1].reshape(n_grid, n_grid).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im0 = axes[0].imshow(u.T, origin="lower", cmap="RdBu_r",
                          extent=[0, 1, 0, 1])
    axes[0].set_title(f"u  (t={t_val:.2f})")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(v.T, origin="lower", cmap="RdBu_r",
                          extent=[0, 1, 0, 1])
    axes[1].set_title(f"v  (t={t_val:.2f})")
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[plot] Salvato: {save_path}")
    plt.show()