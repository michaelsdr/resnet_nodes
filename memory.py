import torch
import torch.nn as nn
from resnet_ode import ResNetBack
import matplotlib.pyplot as plt
from memory_profiler import memory_usage
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(1)

Depths = np.arange(1, 301, 150)


hidden = 8
d = 500

function = nn.Sequential(nn.Linear(d, hidden), nn.Tanh(), nn.Linear(hidden, d))
function_res = nn.Sequential(nn.Linear(d, hidden), nn.Tanh(), nn.Linear(hidden, d))

X = torch.rand(500, 500)


def train(net):
    Loss = (net(X) ** 2).mean()
    Loss.backward()


if __name__ == "__main__":
    Mem_list_nb = []

    for n_iters in Depths:
        net_nb = ResNetBack(
            [
                function,
            ]
            * n_iters,
            use_backprop=False,
        )
        used_mem = np.max(memory_usage((train, (net_nb,))))
        Mem_list_nb.append(used_mem)

    Mem_list_b = []

    for n_iters in Depths:
        net_b = ResNetBack(
            [
                function,
            ]
            * n_iters,
            use_backprop=True,
        )
        used_mem = np.max(memory_usage((train, (net_b,))))
        Mem_list_b.append(used_mem)

    plt.figure(figsize=(4, 1.5))

    plt.plot(Depths, Mem_list_b, label="Backpropagation", linewidth=4, color="purple")
    plt.plot(Depths, Mem_list_nb, label="Adjoint Method", linewidth=4, color="blue")
    plt.yscale("log")
    y_ = plt.ylabel("Memory (MiB)")
    x_ = plt.xlabel("Depth")
    plt.legend()
    plt.savefig(
        "figures/memory_theory.pdf", bbox_inches="tight", bbox_extra_artists=[x_, y_]
    )
