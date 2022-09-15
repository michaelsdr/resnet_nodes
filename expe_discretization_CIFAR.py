import torch
from resnet_ode.models import ResNetMemoryCustom
import numpy as np

images_cifar = np.load("img_cifar.npy")[:64]
images_cifar = torch.tensor(images_cifar)

N_list = [
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    2 ** 12,
    2 ** 13,
    2 ** 14,
    2 ** 15,
]

errors = []
with torch.no_grad():
    for N in N_list:
        net = ResNetMemoryCustom(10, [N, 2, 2, 2])
        net = net.cuda()
        inputs = images_cifar.cuda()
        x = net.conv1(inputs)
        x = net.bn1(x)
        x = net.relu(x)
        if net.large:
            x = net.maxpool(x)
        x = net.layer1[0](x)
        functions = net.layer1[1].network.functions
        n_iters = len(net.layer1[1].network.functions)
        z = x.clone()
        for f in functions:
            x = x + f(x) / n_iters
        y = x.clone()
        for f in functions[::-1]:
            y = y - f(y) / n_iters
        error = (z - y).square().mean().sqrt() / (y.square().mean().sqrt())
        print(N, error)
        errors.append(error)

errors = torch.tensor(errors).cpu()
errors = np.asarray(errors.cpu())
np.save("errors_cifar.npy", errors)
