import numpy as np
import matplotlib.pyplot as plt
import torch

rng = np.random.RandomState(0)


depths = [4, 8, 16, 32, 64]

repo_adr = "metrics_gradient_2"

grads_1 = []
grads_2 = []
idx = 2
for d in depths:
    grads_1.append(torch.load("%s/depth_%d.to" % (repo_adr, d))[idx])
    grads_2.append(torch.load("%s/depth_%d_heun.to" % (repo_adr, d))[idx])

n_points = 20
scale = 2e-1
offset = 0.5


f, ax = plt.subplots(figsize=(3, 2))

for j, (depth, grad1, grad2) in enumerate(zip(depths, grads_1, grads_2)):
    for i, grad in enumerate([grad1, grad2]):
        n_p = len(grad1)
        n = scale * (rng.rand(n_p) + offset) * (2 * i - 1)
        x = n + j
        color = ["darkblue", "red"][i]
        label = ["ResNet", "HeunNet"][i]
        label = None if depth != depths[-1] else label
        plt.scatter(x, grad, color=color, s=0.3, alpha=0.6)
        plt.scatter([], [], color=color, s=5, label=label)

plt.legend()
plt.yscale("log")
plt.xticks(np.arange(len(depths)), ["%d" % depth for depth in depths])
x_ = plt.xlabel("Depth")
y_ = plt.ylabel("Gradient error")
plt.grid()
plt.savefig("gradient_errors.pdf", bbox_inches="tight", bbox_extra_artists=[x_, y_])
