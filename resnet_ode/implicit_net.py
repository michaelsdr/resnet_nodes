import torch
import torch.nn as nn


class ExplicitNet(nn.Module):
    def __init__(self, function, n_iters=1):
        super(ExplicitNet, self).__init__()
        self.add_module(str(1), function)
        self.add_module(str(1), function)
        self.n_iters = n_iters

    def forward(self, x):
        for i in range(self.n_iters):
            x = x + self.function(x)
        return x

    @property
    def function(self):
        return self._modules[str(1)]


class ImplicitNet(nn.Module):
    def __init__(self, function, n_iters=1, implicit=False, rms=False):
        super(ImplicitNet, self).__init__()
        self.add_module(str(1), function)
        self.add_module(str(1), function)
        self.n_iters = n_iters
        self.implicit = implicit
        self.rms = rms

    def forward(self, x):
        if self.rms:
            v = 0
            gamma = 0.9
            eps = 1
            for i in range(self.n_iters):
                f_x = self.function(x)
                v = gamma * v + (f_x ** 2) * (1 - gamma)
                x = x + f_x / ((v + eps).sqrt()) * (1 / self.n_iters)
        if self.implicit:
            with torch.no_grad():
                for i in range(self.n_iters - 1):
                    x = x + self.function(x)
            x = x + self.function(x)
        else:
            for i in range(self.n_iters):
                x = x + self.function(x) / self.n_iters
        return x

    @property
    def function(self):
        return self._modules[str(1)]
