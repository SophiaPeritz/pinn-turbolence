"""
Baseline training loops for Kolmogorov PINN experiments.

This module contains the standard training routine plus the time-marching
variant used for long-horizon simulations.
"""

import torch
import yaml
import os
import copy
from src.network import build_network
from src.losses  import compute_total_loss, compute_ic_loss, compute_pde_loss
from src.utils   import (save_checkpoint, load_checkpoint,
                          sample_collocation_points, sample_ic_points,
                          kolmogorov_ic, plot_loss_history,
                          plot_velocity_field)
import shutil

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass


def compute_adaptive_weights(model, x_ic, u_ic, x_pde, Re, device):
    """
    Compute adaptive IC/PDE weights from gradient norms.
    """
    loss_ic = compute_ic_loss(model, x_ic, u_ic)
    loss_pde = compute_pde_loss(model, x_pde, Re)

    params = [p for p in model.parameters() if p.requires_grad]
    grad_ic = torch.autograd.grad(
        loss_ic, params,
        retain_graph=True, allow_unused=True
    )
    grad_pde = torch.autograd.grad(
        loss_pde, params,
        retain_graph=True, allow_unused=True
    )

    def grad_norm(grads):
        terms = [g.pow(2).sum() for g in grads if g is not None]
        if not terms:
            return torch.zeros((), device=device)
        return torch.sqrt(torch.stack(terms).sum())

    norm_ic = grad_norm(grad_ic)
    norm_pde = grad_norm(grad_pde)
    total = norm_ic + norm_pde

    w_ic = (total / (norm_ic + 1e-8)).detach()
    w_pde = (total / (norm_pde + 1e-8)).detach()

    return w_ic.item(), w_pde.item()


def sample_transfer_ic_points(model, n_ic, t_start, domain, device):
    """Sample the initial condition of a window from the previous window model."""
    x = torch.rand(n_ic, 1) * (domain[1] - domain[0]) + domain[0]
    y = torch.rand(n_ic, 1) * (domain[3] - domain[2]) + domain[2]
    t = torch.full((n_ic, 1), t_start)
    x_ic = torch.cat([t, x, y], dim=1).to(device)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        out = model(x_ic)
    if was_training:
        model.train()

    u_ic = out[:, 0:2].detach()
    return x_ic, u_ic


def _run_training_loop(
    cfg: dict,
    cfg_path: str = None,
    model=None,
    device=None,
    x_ic=None,
    u_ic=None,
    t_range=None,
    results_dir=None,
    log_prefix="[train]",
    checkpoint_prefix="baseline",
    plot_prefix="baseline",
    copy_config=True,
):
    """Run training for a single time window or the baseline full horizon."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{log_prefix} Device: {device}")

    if model is None:
        model = build_network(cfg["network"]).to(device)
    else:
        model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{log_prefix} Parameters: {n_params:,}")

    lr = cfg["training"].get("lr", 1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=cfg["training"].get("lr_decay", 0.9) ** (
            1.0 / cfg["training"].get("lr_decay_steps", 2000)
        )
    )

    Re = cfg["physics"]["Re"]
    t_end = cfg["physics"]["t_end"]
    domain = cfg["physics"]["domain"]
    n_pde = cfg["training"]["n_pde"]
    n_ic = cfg["training"]["n_ic"]
    n_iter = cfg["training"]["n_iter"]
    log_every = cfg["training"].get("log_every", 500)
    save_every = cfg["training"].get("save_every", 5000)
    w_ic = cfg["training"].get("w_ic", 100.0)
    w_pde = cfg["training"].get("w_pde", 1.0)
    use_adaptive = cfg["training"].get("adaptive_weighting", False)
    adaptive_every = cfg["training"].get("adaptive_every", 1000)
    causal_cfg = cfg["training"].get("causal", {})
    causal = causal_cfg.get("enabled", False)

    if t_range is None:
        t_start = 0.0
        t_stop = t_end
    else:
        t_start, t_stop = t_range

    if x_ic is None or u_ic is None:
        x_ic, u_ic = sample_ic_points(
            n_ic, domain,
            u0_fn=kolmogorov_ic,
            device=device
        )

    results_dir = results_dir or cfg.get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)

    tb_dir = os.path.join(results_dir, "tensorboard")
    writer = SummaryWriter(tb_dir)

    if cfg_path and copy_config:
        try:
            shutil.copy(cfg_path, os.path.join(results_dir, "config_used.yaml"))
        except Exception:
            pass

    history = {"total": [], "ic": [], "pde": []}

    print(f"{log_prefix} Training start: {n_iter} iterations, Re={Re}, t=[{t_start:.3f}, {t_stop:.3f}]")
    for it in range(1, n_iter + 1):
        x_pde = sample_collocation_points(
            n_pde,
            t_range=(t_start, t_stop),
            domain=domain,
            device=device
        )

        if use_adaptive and it % adaptive_every == 0:
            w_ic, w_pde = compute_adaptive_weights(
                model, x_ic, u_ic, x_pde, Re, device
            )

        optimizer.zero_grad()

        loss_total, loss_ic, loss_pde, loss_details = compute_total_loss(
            model, x_ic, u_ic, x_pde, Re,
            w_ic=w_ic, w_pde=w_pde,
            causal=causal,
            causal_n_chunks=causal_cfg.get("n_chunks", 16),
            causal_epsilon=causal_cfg.get("epsilon", 1.0),
            t_range=(t_start, t_stop),
            return_details=True,
        )

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        history["total"].append(loss_total.item())
        history["ic"].append(loss_ic.item())
        history["pde"].append(loss_pde.item())

        writer.add_scalar("loss/total", loss_total.item(), it)
        writer.add_scalar("loss/ic", loss_ic.item(), it)
        writer.add_scalar("loss/pde", loss_pde.item(), it)
        if causal:
            weights = loss_details["causal_weights"]
            writer.add_scalar("causal/min_weight", weights.min().item(), it)
            writer.add_scalar("causal/mean_weight", weights.mean().item(), it)
        if use_adaptive and it % adaptive_every == 0:
            writer.add_scalar("adaptive/w_ic", w_ic, it)
            writer.add_scalar("adaptive/w_pde", w_pde, it)

        if it % log_every == 0:
            causal_log = ""
            adaptive_log = ""
            if causal:
                causal_log = f" | w_min={loss_details['causal_weights'].min().item():.2e}"
            if use_adaptive:
                adaptive_log = f" | w_ic={w_ic:.2f} | w_pde={w_pde:.2f}"
            print(f"  it {it:6d} | "
                  f"loss={loss_total.item():.3e} | "
                  f"IC={loss_ic.item():.3e} | "
                  f"PDE={loss_pde.item():.3e} | "
                  f"lr={scheduler.get_last_lr()[0]:.2e}"
                  f"{causal_log}"
                  f"{adaptive_log}")

        if it % save_every == 0:
            ckpt_path = os.path.join(results_dir, "weights",
                                     f"{checkpoint_prefix}_it{it}.pt")
            save_checkpoint(model, optimizer, it,
                            loss_total.item(), ckpt_path)

    print(f"{log_prefix} Training completed.")

    plot_loss_history(
        history,
        save_path=os.path.join(results_dir, f"{plot_prefix}_loss.png")
    )
    plot_velocity_field(
        model, t_val=t_stop, device=device,
        save_path=os.path.join(results_dir, f"{plot_prefix}_velocity.png")
    )

    save_checkpoint(
        model, optimizer, n_iter, history["total"][-1],
        os.path.join(results_dir, "weights", f"{checkpoint_prefix}_final.pt")
    )

    writer.close()
    return model, history, optimizer


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train(cfg: dict, cfg_path: str = None):
    """
    Main single-window training loop.

    Args:
        cfg: Training configuration dictionary.
    """

    model, history, _ = _run_training_loop(
        cfg,
        cfg_path=cfg_path,
        log_prefix="[train]",
        checkpoint_prefix="baseline",
        plot_prefix="baseline",
    )
    return model, history


def train_time_marching(cfg: dict, cfg_path: str = None):
    """Time-marching training with transfer learning across time windows."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_total = cfg["physics"]["t_end"]
    window_size = cfg["training"].get("window_size", 0.1)
    n_windows = int(round(t_total / window_size))
    if n_windows < 1:
        raise ValueError("window_size must be positive and smaller than t_end")

    windows = []
    for i in range(n_windows):
        t_start = i * window_size
        t_stop = t_total if i == n_windows - 1 else min(t_total, (i + 1) * window_size)
        windows.append((t_start, t_stop))

    results_root = cfg.get("results_dir", "results")
    os.makedirs(results_root, exist_ok=True)

    model = build_network(cfg["network"]).to(device)
    all_history = []
    x_ic = None
    u_ic = None

    for i, (t_start, t_stop) in enumerate(windows):
        print(f"\n[time-marching] Window {i + 1}/{n_windows}: t=[{t_start:.1f}, {t_stop:.1f}]")
        if i == 0:
            x_ic, u_ic = sample_ic_points(
                cfg["training"]["n_ic"],
                cfg["physics"]["domain"],
                u0_fn=kolmogorov_ic,
                device=device,
            )
        else:
            x_ic, u_ic = sample_transfer_ic_points(
                model,
                cfg["training"]["n_ic"],
                t_start,
                cfg["physics"]["domain"],
                device=device,
            )

        window_results_dir = os.path.join(results_root, f"window_{i:02d}")
        model, history, _ = _run_training_loop(
            cfg,
            cfg_path=cfg_path,
            model=model,
            device=device,
            x_ic=x_ic,
            u_ic=u_ic,
            t_range=(t_start, t_stop),
            results_dir=window_results_dir,
            log_prefix=f"[time-marching] window {i + 1}/{n_windows}",
            checkpoint_prefix=f"window_{i:02d}",
            plot_prefix=f"window_{i:02d}",
            copy_config=(i == 0),
        )
        all_history.append(history)

    return model, all_history


if __name__ == "__main__":
    cfg = load_config("configs/kolmogorov.yaml")
    train(cfg)
