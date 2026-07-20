"""
Loss functions per il PINN baseline (Strato 1).
Loss composita: IC + PDE, con BC periodiche gestite come hard constraint
tramite embedding spaziale in `src/network.py`.
"""

import torch


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


def compute_causal_pde_loss(model, x_pde, Re, n_chunks=16, epsilon=1.0):
    """Causality-aware PDE loss from Wang et al. (2025).

    Collocation points are ordered in time and split into disjoint chunks.
    The loss of chunk ``i`` is weighted by

        exp(-epsilon * sum(losses of chunks before i)).

    The weights are detached from autograd, as they are scheduling weights
    rather than an additional optimization target.
    """
    if n_chunks < 1:
        raise ValueError("n_chunks must be at least 1")
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if x_pde.shape[0] < n_chunks:
        raise ValueError("n_chunks cannot exceed the number of PDE points")

    # Sorting produces equally populated, disjoint temporal chunks even when
    # the Monte Carlo sampler does not place the same number of points in each
    # fixed-width time interval.
    time_order = torch.argsort(x_pde[:, 0])
    x_sorted = x_pde[time_order]
    res_u, res_v, res_c = compute_pde_residuals(model, x_sorted, Re)

    residual_energy = res_u.square() + res_v.square() + res_c.square()
    chunk_losses = torch.stack([
        chunk.mean() for chunk in torch.tensor_split(residual_energy, n_chunks)
    ])

    preceding_loss = torch.cat([
        torch.zeros_like(chunk_losses[:1]),
        torch.cumsum(chunk_losses.detach()[:-1], dim=0),
    ])
    causal_weights = torch.exp(-epsilon * preceding_loss)
    loss = torch.mean(causal_weights * chunk_losses)
    return loss, chunk_losses.detach(), causal_weights.detach()


def compute_total_loss(
    model,
    x_ic,
    u_ic,
    x_pde,
    Re,
    w_ic=1.0,
    w_pde=1.0,
    w_bc=0.0,
    domain=None,
    n_bc=0,
    device="cpu",
    causal=False,
    causal_n_chunks=16,
    causal_epsilon=1.0,
    return_details=False,
):
    """
    Loss totale composita.

    La periodicità spaziale è imposta come hard constraint nel network.
    Per compatibilità, la funzione ritorna comunque `loss_bc = 0.0`.
    """
    loss_ic = compute_ic_loss(model, x_ic, u_ic)
    if causal:
        loss_pde, chunk_losses, causal_weights = compute_causal_pde_loss(
            model, x_pde, Re,
            n_chunks=causal_n_chunks,
            epsilon=causal_epsilon,
        )
    else:
        loss_pde = compute_pde_loss(model, x_pde, Re)
        chunk_losses = torch.empty(0, device=x_pde.device)
        causal_weights = torch.empty(0, device=x_pde.device)
    loss_bc = torch.zeros((), device=x_ic.device if torch.is_tensor(x_ic) else device)

    loss_total = w_ic * loss_ic + w_pde * loss_pde + w_bc * loss_bc
    result = (loss_total, loss_ic.detach(), loss_pde.detach(), loss_bc.detach())
    if return_details:
        details = {
            "causal_chunk_losses": chunk_losses,
            "causal_weights": causal_weights,
        }
        return (*result, details)
    return result
