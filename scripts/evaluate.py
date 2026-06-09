#!/usr/bin/env python3
"""Script di valutazione: carica checkpoint, calcola KE e enstrophy e salva plot."""
import argparse
import torch
import yaml
import os
from src.network import build_network
from src.utils import enstrophy_from_model, kinetic_energy, plot_velocity_field


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_network(cfg["network"]) .to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    # sample grid for metrics
    n_grid = cfg.get("evaluate", {}).get("n_grid", 64)
    xi = torch.linspace(0, 1, n_grid)
    yi = torch.linspace(0, 1, n_grid)
    Xg, Yg = torch.meshgrid(xi, yi, indexing="ij")
    t_grid = torch.full((n_grid * n_grid, 1), cfg["physics"].get("t_end", 1.0))
    pts = torch.cat([t_grid, Xg.reshape(-1,1), Yg.reshape(-1,1)], dim=1).to(device)

    with torch.no_grad():
        out = model(pts)
    u = out[:,0:1]
    v = out[:,1:2]

    ke = kinetic_energy(u, v)
    enst = enstrophy_from_model(model, pts, device)

    print(f"Kinetic energy: {ke:.6e}")
    print(f"Enstrophy: {enst:.6e}")

    results_dir = cfg.get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)
    plot_velocity_field(model, t_val=cfg["physics"].get("t_end",1.0), n_grid=n_grid, device=device,
                        save_path=os.path.join(results_dir, "eval_velocity.png"))


if __name__ == "__main__":
    main()
