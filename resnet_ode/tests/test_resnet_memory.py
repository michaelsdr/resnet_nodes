import torch
import torch.nn as nn

from resnet_ode import ResNetBack

torch.manual_seed(1)


def test_dimension_layers():
    function = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 3), nn.Linear(3, 3))
    net_nb = ResNetBack(function, use_backprop=False)
    net_b = ResNetBack(function, use_backprop=True)
    x = torch.rand(3)
    assert net_b(x).shape == x.shape
    assert net_nb(x).shape == x.shape


def test_outputs_memory():
    functions = [
        nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 3)) for _ in range(36)
    ]
    net_nb = ResNetBack(functions, use_backprop=False)
    net_b = ResNetBack(functions, use_backprop=False)
    x = torch.rand(3, requires_grad=True)
    assert torch.allclose(net_nb(x), net_b(x), atol=1e-5, rtol=1e-4)
    net_b_output = (net_b(x) ** 2).sum()
    net_nb_output = (net_nb(x) ** 2).sum()
    assert (
        torch.autograd.grad(net_b_output, x)[0].shape
        == torch.autograd.grad(net_nb_output, x)[0].shape
    )
