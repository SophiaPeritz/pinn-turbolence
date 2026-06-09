"""
Navier-Stokes 2D incompressible - Kolmogorov flow
Equazioni governanti e forzante esterno
"""

"""
File opzionale: definizioni PDE per DeepXDE.
Nota: il training baseline usa PyTorch autograd (`src/losses.py`).
Questo file fornisce una versione alternativa basata su DeepXDE se si vuole
esperimentare con quella libreria.
"""

try:
    import deepxde as dde
except Exception:
    dde = None
import torch


def navier_stokes_residuals(x, y):
    """
    Residui delle equazioni di Navier-Stokes 2D incomprimibili.

    Input:
        x : collocation points [t, x, y]
        y : output della rete [u, v, p]

    Equazioni:
        du/dt + u*du/dx + v*du/dy = -dp/dx + (1/Re)*(d2u/dx2 + d2u/dy2) + fx
        dv/dt + u*dv/dx + v*dv/dy = -dp/dy + (1/Re)*(d2v/dx2 + d2v/dy2)
        du/dx + dv/dy = 0  (incomprimibilita)
    """
    Re = dde.config.get("Re", 1000.0)

    # Variabili di output
    u = y[:, 0:1]
    v = y[:, 1:2]
    p = y[:, 2:3]

    # Derivate prime rispetto a t, x, y
    du_dt = dde.grad.jacobian(y, x, i=0, j=0)
    du_dx = dde.grad.jacobian(y, x, i=0, j=1)
    du_dy = dde.grad.jacobian(y, x, i=0, j=2)

    dv_dt = dde.grad.jacobian(y, x, i=1, j=0)
    dv_dx = dde.grad.jacobian(y, x, i=1, j=1)
    dv_dy = dde.grad.jacobian(y, x, i=1, j=2)

    dp_dx = dde.grad.jacobian(y, x, i=2, j=1)
    dp_dy = dde.grad.jacobian(y, x, i=2, j=2)

    # Derivate seconde (termine diffusivo)
    du_dxx = dde.grad.hessian(y, x, component=0, i=1, j=1)
    du_dyy = dde.grad.hessian(y, x, component=0, i=2, j=2)

    dv_dxx = dde.grad.hessian(y, x, component=1, i=1, j=1)
    dv_dyy = dde.grad.hessian(y, x, component=1, i=2, j=2)

    # Forzante di Kolmogorov: f(x,y) = [0.1 * sin(4*pi*y), 0]
    # Inietta energia al numero d'onda k=2
    fx = 0.1 * torch.sin(4.0 * torch.pi * x[:, 2:3])
    fy = torch.zeros_like(fx)

    # Residui
    momentum_u = (
        du_dt
        + u * du_dx
        + v * du_dy
        + dp_dx
        - (1.0 / Re) * (du_dxx + du_dyy)
        - fx
    )

    momentum_v = (
        dv_dt
        + u * dv_dx
        + v * dv_dy
        + dp_dy
        - (1.0 / Re) * (dv_dxx + dv_dyy)
        - fy
    )

    continuity = du_dx + dv_dy

    return momentum_u, momentum_v, continuity