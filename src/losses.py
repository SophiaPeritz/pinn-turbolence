"""
Loss functions per il PINN baseline (Strato 1).
Loss composita: IC + BC + PDE, pesi fissi.
Gli strati successivi aggiungeranno causal weighting e adaptive weights.
"""

import torch
import torch.nn as nn


def compute_pde_residuals(model, x_pde, Re):
    """
    Calcola i residui PDE usando autograd di PyTorch.

    Args:
        model : la rete PINN
        x_pde : collocation points shape (N, 3) -> [t, x, y]
        Re    : Reynolds number

    Returns:
        res_u, res_v, res_c : residui momentum u, momentum v, continuita
    """
    x_pde = x_pde.requires_grad_(True)
    out = model(x_pde)  # (N, 3) -> [u, v, p]

    u = out[:, 0:1]
    v = out[:, 1:2]
    p = out[:, 2:3]

    def grad(f, x, create=True):
        return torch.autograd.grad(
            f, x,
            grad_outputs=torch.ones_like(f),
            create_graph=create,
            retain_graph=True
        )[0]

    # Gradiente dell'output rispetto a [t, x, y]
    du = grad(u, x_pde)
    dv = grad(v, x_pde)
    dp = grad(p, x_pde)

    du_dt, du_dx, du_dy = du[:, 0:1], du[:, 1:2], du[:, 2:3]
    dv_dt, dv_dx, dv_dy = dv[:, 0:1], dv[:, 1:2], dv[:, 2:3]
    dp_dx = dp[:, 1:2]
    dp_dy = dp[:, 2:3]

    # Derivate seconde (Laplaciano)
    du_dxx = grad(du_dx, x_pde)[:, 1:2]
    du_dyy = grad(du_dy, x_pde)[:, 2:3]
    dv_dxx = grad(dv_dx, x_pde)[:, 1:2]
    dv_dyy = grad(dv_dy, x_pde)[:, 2:3]

    # Forzante di Kolmogorov
    fx = 0.1 * torch.sin(4.0 * torch.pi * x_pde[:, 2:3])

    # Residui
    res_u = du_dt + u * du_dx + v * du_dy + dp_dx - (1.0 / Re) * (du_dxx + du_dyy) - fx
    res_v = dv_dt + u * dv_dx + v * dv_dy + dp_dy - (1.0 / Re) * (dv_dxx + dv_dyy)
    res_c = du_dx + dv_dy

    return res_u, res_v, res_c


def compute_ic_loss(model, x_ic, u_ic):
    """
    Loss sulla condizione iniziale.

    Args:
        model : rete PINN
        x_ic  : punti IC shape (N, 3), t=0
        u_ic  : valori di riferimento shape (N, 2) -> [u, v]
    """
    out = model(x_ic)
    pred_u = out[:, 0:1]
    pred_v = out[:, 1:2]
    loss = torch.mean((pred_u - u_ic[:, 0:1]) ** 2) + \
           torch.mean((pred_v - u_ic[:, 1:2]) ** 2)
    return loss


def compute_pde_loss(model, x_pde, Re):
    """
    Loss PDE (media dei quadrati dei residui).
    Strato 1: pesi uniformi, nessun causal weighting.
    """
    res_u, res_v, res_c = compute_pde_residuals(model, x_pde, Re)
    loss = (
        torch.mean(res_u ** 2) +
        torch.mean(res_v ** 2) +
        torch.mean(res_c ** 2)
    )
    return loss


def compute_total_loss(model, x_ic, u_ic, x_pde, Re,
                       w_ic=1.0, w_pde=1.0):
    """
    Loss totale composita (Strato 1 - pesi fissi).

    Args:
        w_ic  : peso per la loss IC
        w_pde : peso per la loss PDE

    Returns:
        loss_total, loss_ic, loss_pde  (per logging)
    """
    loss_ic  = compute_ic_loss(model, x_ic, u_ic)
    loss_pde = compute_pde_loss(model, x_pde, Re)

    loss_total = w_ic * loss_ic + w_pde * loss_pde

    return loss_total, loss_ic.detach(), loss_pde.detach()