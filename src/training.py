"""
Training loop baseline (Strato 1).
Adam optimizer, pesi fissi, nessun time-marching.
Gli strati successivi estenderanno questo file.
"""

import torch
import yaml
import os
from src.network import build_network
from src.losses  import compute_total_loss
from src.utils   import (save_checkpoint, load_checkpoint,
                          sample_collocation_points, sample_ic_points,
                          kolmogorov_ic, plot_loss_history,
                          plot_velocity_field)
from torch.utils.tensorboard import SummaryWriter
import shutil


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train(cfg: dict, cfg_path: str = None):
    """
    Training loop principale.

    Args:
        cfg : dizionario di configurazione (da kolmogorov.yaml)
    """

    # ── Device ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    # ── Rete ────────────────────────────────────────────────────────────
    model = build_network(cfg["network"]).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] Parametri: {n_params:,}")

    # ── Optimizer ───────────────────────────────────────────────────────
    lr = cfg["training"].get("lr", 1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler: decay esponenziale
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=cfg["training"].get("lr_decay", 0.9) ** (
            1.0 / cfg["training"].get("lr_decay_steps", 2000)
        )
    )

    # ── Configurazione ──────────────────────────────────────────────────
    Re           = cfg["physics"]["Re"]
    t_end        = cfg["physics"]["t_end"]
    domain       = cfg["physics"]["domain"]      # [x_min, x_max, y_min, y_max]
    n_pde        = cfg["training"]["n_pde"]
    n_ic         = cfg["training"]["n_ic"]
    n_iter       = cfg["training"]["n_iter"]
    log_every    = cfg["training"].get("log_every", 500)
    save_every   = cfg["training"].get("save_every", 5000)
    w_ic         = cfg["training"].get("w_ic", 100.0)   # peso IC alto: IC importante
    w_pde        = cfg["training"].get("w_pde", 1.0)
    causal_cfg   = cfg["training"].get("causal", {})
    causal       = causal_cfg.get("enabled", False)
    results_dir  = cfg.get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)

    # ── Condizione iniziale (fissa per tutto il training) ───────────────
    x_ic, u_ic = sample_ic_points(
        n_ic, domain,
        u0_fn=kolmogorov_ic,
        device=device
    )

    # ── TensorBoard writer
    tb_dir = os.path.join(results_dir, "tensorboard")
    writer = SummaryWriter(tb_dir)

    # Save a copy of config used
    if cfg_path:
        try:
            shutil.copy(cfg_path, os.path.join(results_dir, "config_used.yaml"))
        except Exception:
            pass

    # ── History per plotting ─────────────────────────────────────────────
    history = {"total": [], "ic": [], "pde": [], "bc": []}

    # ── Loop di training ─────────────────────────────────────────────────
    print(f"[train] Inizio training: {n_iter} iterazioni, Re={Re}")
    for it in range(1, n_iter + 1):

        # Ricampiona i punti PDE ad ogni iterazione (Monte Carlo)
        x_pde = sample_collocation_points(
            n_pde,
            t_range=(0.0, t_end),
            domain=domain,
            device=device
        )

        optimizer.zero_grad()

        loss_total, loss_ic, loss_pde, loss_bc, loss_details = compute_total_loss(
            model, x_ic, u_ic, x_pde, Re,
            w_ic=w_ic, w_pde=w_pde,
            w_bc=cfg["training"].get("w_bc", 1.0),
            domain=domain, n_bc=cfg["training"].get("n_bc", 256), device=device,
            causal=causal,
            causal_n_chunks=causal_cfg.get("n_chunks", 16),
            causal_epsilon=causal_cfg.get("epsilon", 1.0),
            return_details=True,
        )

        loss_total.backward()
        # Gradient clipping: evita esplosione dei gradienti
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Logging
        history["total"].append(loss_total.item())
        history["ic"].append(loss_ic.item())
        history["pde"].append(loss_pde.item())
        history["bc"].append(loss_bc.item())

        # TensorBoard
        it_global = it
        writer.add_scalar("loss/total", loss_total.item(), it_global)
        writer.add_scalar("loss/ic", loss_ic.item(), it_global)
        writer.add_scalar("loss/pde", loss_pde.item(), it_global)
        writer.add_scalar("loss/bc", loss_bc.item(), it_global)
        if causal:
            weights = loss_details["causal_weights"]
            writer.add_scalar("causal/min_weight", weights.min().item(), it_global)
            writer.add_scalar("causal/mean_weight", weights.mean().item(), it_global)

        if it % log_every == 0:
            causal_log = ""
            if causal:
                causal_log = f" | w_min={loss_details['causal_weights'].min().item():.2e}"
            print(f"  it {it:6d} | "
                  f"loss={loss_total.item():.3e} | "
                  f"IC={loss_ic.item():.3e} | "
                  f"PDE={loss_pde.item():.3e} | "
                  f"BC={loss_bc.item():.3e} | "
                  f"lr={scheduler.get_last_lr()[0]:.2e}"
                  f"{causal_log}")

        # Checkpoint
        if it % save_every == 0:
            ckpt_path = os.path.join(results_dir, "weights",
                                      f"baseline_it{it}.pt")
            save_checkpoint(model, optimizer, it,
                             loss_total.item(), ckpt_path)

    # ── Salva risultati finali ────────────────────────────────────────────
    print("[train] Training completato.")

    plot_loss_history(
        history,
        save_path=os.path.join(results_dir, "baseline_loss.png")
    )
    plot_velocity_field(
        model, t_val=t_end, device=device,
        save_path=os.path.join(results_dir, "baseline_velocity.png")
    )

    # Checkpoint finale
    save_checkpoint(
        model, optimizer, n_iter, history["total"][-1],
        os.path.join(results_dir, "weights", "baseline_final.pt")
    )

    writer.close()

    return model, history


if __name__ == "__main__":
    cfg = load_config("configs/kolmogorov.yaml")
    train(cfg)
