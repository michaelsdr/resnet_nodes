import numpy as np
import argparse
from resnet_ode.trainer_CIFAR_10 import train_resnet
import os

parser = argparse.ArgumentParser(description="test")
parser.add_argument("-m", default="resnetmemory101", type=str)
parser.add_argument("-s", default="results_cifar_10")
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--n_layers", default=8, type=int)
parser.add_argument("--save", default=True, type=bool)
parser.add_argument("--adapt_lr", default=False, type=bool)
parser.add_argument("--lr_schedule", default=True, type=bool)
parser.add_argument("--checkdir", default="checkpoint_CIFAR", type=str)
parser.add_argument("--use_backprop", default=True, type=bool)


args = parser.parse_args()


model = args.m  # change to 'iresnetauto' to use tied weights
save_adr = args.s
n_layers = args.n_layers  # useful only if weights are tied
save = args.save
seed = args.seed
cd = args.checkdir
adaptative_lr = args.adapt_lr
use_lr_schedule = args.lr_schedule
use_backprop = args.use_backprop

depth = "101" if model == "resnetmemory101" else "auto"

expe_name = "compare_convergence_%s" % depth
save_at = "results_CIFAR"
save_adr = "%s/%s/" % (save_at, expe_name)

try:
    os.makedirs(save_adr)
except:
    pass

"""
lr_list, this would affect training only if lr_schedule is False, otherwise the trainer uses cosine scheduler
"""

n_iters = 200
lr = np.ones(n_iters) * 0.1
lr[100:] /= 10
lr[150:] /= 10

if __name__ == "__main__":

    train_accs, train_losss, test_accs, test_losss = train_resnet(
        lr,
        model,
        seed=seed,
        save=save,
        save_adr=save_adr,
        save_at=save_at,
        n_layers=n_layers,
        checkpoint_dir=cd,
        adaptative_lr=adaptative_lr,
        use_lr_schedule=use_lr_schedule,
        use_backprop=use_backprop,
    )
