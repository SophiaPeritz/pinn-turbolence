"""Evaluation script for the Kolmogorov PINN baseline.

Loads a trained checkpoint, evaluates it on a regular grid at a fixed time,
computes physics metrics, and saves both numeric results and plots.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml

from src.losses import compute_pde_residuals
from src.network import build_network
from src.utils import enstrophy_from_model, kinetic_energy, load_checkpoint


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def make_grid(n_grid: int, domain: list[float], t_val: float, device: torch.device):
    x_min, x_max, y_min, y_max = domain
    xs = torch.linspace(x_min, x_max, n_grid, device=device)
    ys = torch.linspace(y_min, y_max, n_grid, device=device)
    xx, yy = torch.meshgrid(xs, ys, indexing="ij")
    tt = torch.full((n_grid * n_grid, 1), t_val, device=device)
    pts = torch.cat([tt, xx.reshape(-1, 1), yy.reshape(-1, 1)], dim=1)
    return pts, xx, yy


def evaluate_model(cfg: dict, checkpoint_path: str, n_grid: int | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path(cfg.get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    model = build_network(cfg["network"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"].get("lr", 1e-3))
    _, _ = load_checkpoint(model, optimizer, checkpoint_path, device=device)
    model.eval()

    t_val = float(cfg["physics"]["t_end"])
    n_grid = int(n_grid or cfg.get("evaluate", {}).get("n_grid", 64))
    domain = cfg["physics"]["domain"]

    grid_pts, xx, yy = make_grid(n_grid=n_grid, domain=domain, t_val=t_val, device=device)

    with torch.no_grad():
        out = model(grid_pts)

    u = out[:, 0].reshape(n_grid, n_grid)
    v = out[:, 1].reshape(n_grid, n_grid)

    ke = kinetic_energy(u, v)
    en = enstrophy_from_model(model, grid_pts, device=device)

    # PDE residuals on the same grid (requires grad)
    pde_pts = grid_pts.detach().clone().requires_grad_(True)
    res_u, res_v, res_c = compute_pde_residuals(model, pde_pts, cfg["physics"]["Re"])
    pde_mse = (
        torch.mean(res_u**2) +
        torch.mean(res_v**2) +
        torch.mean(res_c**2)
    ).item()

    metrics = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "t_val": t_val,
        "n_grid": n_grid,
        "kinetic_energy": ke,
        "enstrophy": en,
        "pde_mse": pde_mse,
    }

    metrics_path = results_dir / "eval_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    # Plot velocity fields
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    im0 = axes[0].imshow(u.detach().cpu().numpy().T, origin="lower", cmap="RdBu_r",
                         extent=[domain[0], domain[1], domain[2], domain[3]])
    axes[0].set_title(f"u(t={t_val:.2f})")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(v.detach().cpu().numpy().T, origin="lower", cmap="RdBu_r",
                         extent=[domain[0], domain[1], domain[2], domain[3]])
    axes[1].set_title(f"v(t={t_val:.2f})")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    plt.colorbar(im1, ax=axes[1])

    plot_path = results_dir / "eval_velocity.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(json.dumps(metrics, indent=2))
    print(f"[evaluate] metrics saved to {metrics_path}")
    print(f"[evaluate] plot saved to {plot_path}")

    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained Kolmogorov PINN checkpoint.")
    parser.add_argument("--config", default="configs/kolmogorov.yaml", help="Path to YAML config")
    parser.add_argument("--checkpoint", default="results/baseline/weights/baseline_final.pt", help="Checkpoint path")
    parser.add_argument("--n-grid", type=int, default=None, help="Grid resolution for evaluation")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = load_config(args.config)
    evaluate_model(cfg, args.checkpoint, n_grid=args.n_grid)


if __name__ == "__main__":
    main()