import torch

import matplotlib.pyplot as plt
import numpy as np


n_tries = 5
p = 0.85
depths = [2, 4, 8, 16, 32, 64]

res = {}
for h in ["False", "True"]:
    for b in ["False", "True"]:
        for d in depths:
            res[h, b, d] = []


plt.figure(figsize=(4 * p, 4 * p))

for i in range(1, n_tries):
    for h in ["False", "True"]:
        for b in ["False", "True"]:
            acc = []
            for d in depths:
                test_acc = max(
                    torch.load(
                        "results/%s_heun_%s_backprop_%s_seed_%s.to" % (d, h, b, i)
                    )["test_acc"]
                )
                acc.append(100 - test_acc)
                res[h, b, d].append(test_acc)


for h in ["False", "True"]:
    for b in ["False", "True"]:
        acc = []

        for d in depths:
            acc.append(100 - np.array(res[h, b, d]))
            if h == "False" and b == "False":
                label = "ResNet + Adjoint Method"
                color = "darkblue"
            if h == "False" and b == "True":
                label = "ResNet + Backprop"
                color = "darkblue"
            if h == "True" and b == "False":
                label = "HeunNet + Adjoint Method"
                color = "red"

            if h == "True" and b == "True":
                label = "HeunNet + Backprop"
                color = "red"

        if b == "False":
            linestyle = "--"
        else:
            linestyle = "-"
        plt.semilogy(
            np.mean(acc, axis=-1),
            linestyle=linestyle,
            linewidth=3,
            label=label,
            color=color,
        )
        plt.fill_between(
            np.arange(len(acc)),
            np.quantile(acc, q=0.25, axis=-1),
            np.quantile(acc, q=0.75, axis=-1),
            alpha=0.2,
            linewidth=3,
            color=color,
        )


plt.yticks(
    [15, 20, 30, 40, 50, 60, 70, 80, 90],
    [
        r"$15 \%$",
        r"$20 \%$",
        r"$30 \%$",
        r"$40 \%$",
        r"$50 \%$",
        r"$60 \%$",
        r"$70 \%$",
        r"$80 \%$",
        r"$90 \%$",
    ],
)

plt.xticks(np.arange(len(depths)), depths)
plt.ylim(12, 60)
plt.ylabel("Test error")
plt.xlabel("Depth")
plt.grid(which="major")
plt.legend(handlelength=2.0)
plt.tight_layout()
plt.savefig("figures/cifar_acc.pdf")
