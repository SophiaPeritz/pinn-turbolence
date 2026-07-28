"""
Loss functions for the Kolmogorov PINN experiments.

The composite objective combines the initial-condition loss and the Navier--Stokes
residual loss.
Periodic boundary conditions are not imposed through an additional penalty term; 
instead, they are enforced as hard constraints by transforming the spatial 
coordinates using a periodic Fourier embedding before being provided to the
neural network.
"""


import torch


def compute_pde_residuals(model, x_pde, Re):
    """
    Compute the PDE residuals with PyTorch autograd.

    Args:
        model: PINN model.
        x_pde: Collocation points with shape (N, 3) and columns [t, x, y].
        Re: Reynolds number.

    Returns:
        Residuals for the u-momentum, v-momentum, and continuity equations.
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

    # Gradients of the network outputs with respect to [t, x, y].
    du = grad(u, x_pde)
    dv = grad(v, x_pde)
    dp = grad(p, x_pde)

    du_dt, du_dx, du_dy = du[:, 0:1], du[:, 1:2], du[:, 2:3]
    dv_dt, dv_dx, dv_dy = dv[:, 0:1], dv[:, 1:2], dv[:, 2:3]
    dp_dx = dp[:, 1:2]
    dp_dy = dp[:, 2:3]

    # Second derivatives used in the Laplacian terms.
    du_dxx = grad(du_dx, x_pde)[:, 1:2]
    du_dyy = grad(du_dy, x_pde)[:, 2:3]
    dv_dxx = grad(dv_dx, x_pde)[:, 1:2]
    dv_dyy = grad(dv_dy, x_pde)[:, 2:3]

    # Kolmogorov forcing term.
    fx = 0.1 * torch.sin(4.0 * torch.pi * x_pde[:, 2:3])

    # PDE residuals.
    res_u = du_dt + u * du_dx + v * du_dy + dp_dx - (1.0 / Re) * (du_dxx + du_dyy) - fx
    res_v = dv_dt + u * dv_dx + v * dv_dy + dp_dy - (1.0 / Re) * (dv_dxx + dv_dyy)
    res_c = du_dx + dv_dy

    return res_u, res_v, res_c


def compute_ic_loss(model, x_ic, u_ic):
    """
    Initial-condition loss.

    Args:
        model: PINN model.
        x_ic: Initial-condition points with shape (N, 3) and t = 0.
        u_ic: Reference values with shape (N, 2) and columns [u, v].
    """
    out = model(x_ic)
    pred_u = out[:, 0:1]
    pred_v = out[:, 1:2]
    loss = torch.mean((pred_u - u_ic[:, 0:1]) ** 2) + \
           torch.mean((pred_v - u_ic[:, 1:2]) ** 2)
    return loss


def compute_pde_loss(model, x_pde, Re):
    """
    Mean-squared PDE residual loss.

    This is the standard non-causal loss used by the baseline setup.
    """
    res_u, res_v, res_c = compute_pde_residuals(model, x_pde, Re)
    loss = (
        torch.mean(res_u ** 2) +
        torch.mean(res_v ** 2) +
        torch.mean(res_c ** 2)
    )
    return loss


def compute_causal_pde_loss(model, x_pde, Re, n_chunks=16, epsilon=1.0, t_range=None):
    """Causality-aware PDE loss.

    Collocation points are grouped into fixed-width time bins over the explicit
    time range [t_min, t_max]. The loss of chunk ``i`` is weighted by

        exp(-epsilon * sum(losses of chunks before i)).

    The weights are detached from autograd because they act as scheduling
    coefficients rather than an optimization target.
    """
    if n_chunks < 1:
        raise ValueError("n_chunks must be at least 1")
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if x_pde.shape[0] < n_chunks:
        raise ValueError("n_chunks cannot exceed the number of PDE points")
    if t_range is None:
        raise ValueError(
            "An explicit t_range=(t0, t1) is required to reproduce the Wang et al. chunking"
        )

    t_min, t_max = t_range
    t_min = torch.as_tensor(t_min, device=x_pde.device, dtype=x_pde.dtype)
    t_max = torch.as_tensor(t_max, device=x_pde.device, dtype=x_pde.dtype)

    if t_max <= t_min:
        raise ValueError("t_range must satisfy t_max > t_min")

    bin_edges = torch.linspace(t_min, t_max, n_chunks + 1, device=x_pde.device, dtype=x_pde.dtype)
    res_u, res_v, res_c = compute_pde_residuals(model, x_pde, Re)

    # |R|^2 = Ru^2 + Rv^2 + Rc^2.
    residual_energy = res_u.square() + res_v.square() + res_c.square()

    chunk_losses = []
    time_values = x_pde[:, 0]
    for i in range(n_chunks):
        left = bin_edges[i]
        right = bin_edges[i + 1]
        if i == n_chunks - 1:
            chunk_mask = (time_values >= left) & (time_values <= right)
        else:
            chunk_mask = (time_values >= left) & (time_values < right)

        if chunk_mask.any().item():
            chunk_losses.append(residual_energy[chunk_mask].mean())
        else:
            # Defensive fallback: empty bins should be rare with uniform sampling.
            chunk_losses.append(torch.zeros((), device=x_pde.device, dtype=residual_energy.dtype))

    chunk_losses = torch.stack(chunk_losses)

    preceding_loss = torch.cat([
        torch.zeros_like(chunk_losses[:1]),
        torch.cumsum(chunk_losses.detach()[:-1], dim=0),
    ])
    causal_weights = torch.exp(-epsilon * preceding_loss)
    loss = torch.sum(causal_weights * chunk_losses) / n_chunks
    return loss, chunk_losses.detach(), causal_weights.detach()


def compute_total_loss(
    model,
    x_ic,
    u_ic,
    x_pde,
    Re,
    w_ic=1.0,
    w_pde=1.0,
    causal=False,
    causal_n_chunks=16,
    causal_epsilon=1.0,
    t_range=None,
    return_details=False,
):
    """
    Composite loss used by the training loops.

    Spatial periodicity is enforced as a hard constraint in the network,
    therefore no boundary-condition penalty term is included.
    """
    loss_ic = compute_ic_loss(model, x_ic, u_ic)
    if causal:
        loss_pde, chunk_losses, causal_weights = compute_causal_pde_loss(
            model, x_pde, Re,
            n_chunks=causal_n_chunks,
            epsilon=causal_epsilon,
            t_range=t_range,
        )
    else:
        loss_pde = compute_pde_loss(model, x_pde, Re)
        chunk_losses = torch.empty(0, device=x_pde.device)
        causal_weights = torch.empty(0, device=x_pde.device)

    loss_total = w_ic * loss_ic + w_pde * loss_pde
    result = (loss_total, loss_ic.detach(), loss_pde.detach())
    if return_details:
        details = {
            "causal_chunk_losses": chunk_losses,
            "causal_weights": causal_weights,
        }
        return (*result, details)
    return result
