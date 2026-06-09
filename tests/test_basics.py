import torch
from src.network import build_network
from src.losses import compute_total_loss
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

    loss_total, loss_ic, loss_pde, loss_bc = compute_total_loss(model, x_ic, u_ic, x_pde, Re=100.0,
                                                               w_ic=1.0, w_pde=1.0, w_bc=1.0,
                                                               domain=domain, n_bc=8, device="cpu")
    assert torch.isfinite(loss_total)
