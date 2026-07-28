import torch
import pytest
from src.network import build_network
from src.losses import compute_causal_pde_loss, compute_total_loss
from src.training import compute_adaptive_weights, sample_transfer_ic_points
from src.utils import sample_collocation_points, sample_ic_points, kolmogorov_ic


def test_network_forward():
    cfg = {"network": {"input_dim": 3, "embed_dim": 8, "hidden_dim": 32, "n_layers": 2, "output_dim": 3}}
    model = build_network(cfg["network"]) 
    x = torch.randn(10, 3)
    out = model(x)
    assert out.shape == (10, 3)


def test_loss_and_sampling_shapes():
    cfg_net = {"input_dim":3, "embed_dim":8, "hidden_dim":32, "n_layers":2, "output_dim":3}
    model = build_network(cfg_net)

    domain = (0,1,0,1)
    x_ic, u_ic = sample_ic_points(16, domain, kolmogorov_ic, device="cpu")
    x_pde = sample_collocation_points(32, (0.0,1.0), domain, device="cpu")

    loss_total, loss_ic, loss_pde = compute_total_loss(
        model,
        x_ic,
        u_ic,
        x_pde,
        Re=100.0,
        w_ic=1.0,
        w_pde=1.0,
    )
    assert torch.isfinite(loss_total)


def test_causal_pde_weights_respect_time_order():
    cfg_net = {"input_dim": 3, "embed_dim": 8, "hidden_dim": 16,
               "n_layers": 2, "output_dim": 3}
    model = build_network(cfg_net)
    x_pde = sample_collocation_points(16, (0.0, 1.0), (0, 1, 0, 1), device="cpu")

    loss, chunk_losses, weights = compute_causal_pde_loss(
        model, x_pde, Re=100.0, n_chunks=4, epsilon=1.0, t_range=(0.0, 1.0)
    )

    assert torch.isfinite(loss)
    assert chunk_losses.shape == (4,)
    assert weights.shape == (4,)
    assert torch.isclose(weights[0], torch.tensor(1.0))
    assert torch.all(weights[1:] <= weights[:-1])
    assert not weights.requires_grad


def test_causal_pde_chunks_follow_fixed_time_bins(monkeypatch):
    cfg_net = {"input_dim": 3, "embed_dim": 8, "hidden_dim": 16,
               "n_layers": 2, "output_dim": 3}
    model = build_network(cfg_net)

    x_pde = torch.tensor(
        [
            [0.05, 0.0, 0.0],
            [0.15, 0.0, 0.0],
            [0.20, 0.0, 0.0],
            [0.90, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    def fake_residuals(model, x_pde, Re):
        t = x_pde[:, 0:1]
        zeros = torch.zeros_like(t)
        return t, zeros, zeros

    monkeypatch.setattr("src.losses.compute_pde_residuals", fake_residuals)

    loss, chunk_losses, weights = compute_causal_pde_loss(
        model,
        x_pde,
        Re=100.0,
        n_chunks=2,
        epsilon=1.0,
        t_range=(0.0, 1.0),
    )

    expected = torch.tensor([
        (0.05 ** 2 + 0.15 ** 2 + 0.20 ** 2) / 3.0,
        0.90 ** 2,
    ])

    assert torch.isfinite(loss)
    assert torch.allclose(chunk_losses, expected, atol=1e-6)
    assert weights.shape == (2,)
    assert torch.isclose(weights[0], torch.tensor(1.0))


def test_adaptive_weights_are_finite_and_positive():
    cfg_net = {"input_dim": 3, "embed_dim": 8, "hidden_dim": 16,
               "n_layers": 2, "output_dim": 3}
    model = build_network(cfg_net)

    domain = (0, 1, 0, 1)
    x_ic, u_ic = sample_ic_points(8, domain, kolmogorov_ic, device="cpu")
    x_pde = sample_collocation_points(16, (0.0, 1.0), domain, device="cpu")

    w_ic, w_pde = compute_adaptive_weights(model, x_ic, u_ic, x_pde, Re=100.0, device="cpu")

    assert torch.isfinite(torch.tensor(w_ic))
    assert torch.isfinite(torch.tensor(w_pde))
    assert w_ic > 0.0
    assert w_pde > 0.0


def test_transfer_ic_sampling_uses_window_start_time():
    cfg_net = {"input_dim": 3, "embed_dim": 8, "hidden_dim": 16,
               "n_layers": 2, "output_dim": 3}
    model = build_network(cfg_net)

    domain = (0, 1, 0, 1)
    t_start = 0.5
    x_ic, u_ic = sample_transfer_ic_points(model, 8, t_start, domain, device="cpu")

    assert x_ic.shape == (8, 3)
    assert u_ic.shape == (8, 2)
    assert torch.allclose(x_ic[:, 0], torch.full((8,), t_start))
    assert torch.isfinite(u_ic).all()
