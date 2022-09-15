import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random

import numpy as np


def set_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


set_all_seeds(6)
d = 2

n = 100

B = torch.eye(d) + torch.randn(d, d)

np.linalg.det(B)

B = B


class Net(nn.Module):
    def __init__(self, A, d=d, n=n):
        super(Net, self).__init__()
        self.A = A
        self.d = d
        self.n = n

    def forward(self):
        d = self.d
        n = self.n
        I = torch.eye(d)
        X = torch.eye(d)
        for A_i in self.A:
            X = X.mm(I + A_i / n)
        return X


class NetBackward(nn.Module):
    def __init__(self, A, d=d, n=n):
        super(NetBackward, self).__init__()
        self.A = A
        self.d = d
        self.n = n

    def forward(self):
        d = self.d
        n = self.n
        I = torch.eye(d)
        X = torch.eye(d)
        for A_i in list(self.A)[::-1]:
            X = X.mm(I - A_i / n)
        return X


A = torch.zeros(n, d, d)  # + (1/n) * torch.randn(n, d, d)
A = nn.Parameter(A)

net = Net(A, d, n)

optimizer = torch.optim.SGD(net.parameters(), lr=100)

n_epochs = 10000

A_list = []
for k in range(n_epochs):
    optimizer.zero_grad()
    output = (net() - B).square().mean()
    A_list.append(A.clone().detach())
    output.backward()
    optimizer.step()
    if k % 100 == 0:
        print(output)
        if output < 1e-5:
            break
p = 0.5
plt.figure(figsize=(7 * p, 5 * p))


opt_indices = np.sort(np.random.randint(1, len(A_list), 4))
for x in opt_indices:
    plt.plot(
        A_list[x][:, np.random.randint(d), np.random.randint(d)],
        linewidth=3,
        label="Step: " + str(x),
    )
plt.xlabel("Depth $N$")
plt.ylabel("Weight value")
plt.tight_layout()
plt.legend(ncol=2)
plt.savefig("figures/linear_weights.pdf")
