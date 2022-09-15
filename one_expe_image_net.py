import argparse
from resnet_ode.trainer_imagenet import main
import os

parser = argparse.ArgumentParser(description="test")
parser.add_argument("-m", default="resnetmemory101", type=str)
parser.add_argument("--n_layers", default=8, type=int)
parser.add_argument("-e", default="comparison_101")
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--use_backprop", default=True, type=bool)
args = parser.parse_args()


model = args.m  # change to 'iresnetauto' to use tied weights
n_layers = args.n_layers  # useful only if weights are tied
expe_name = args.e
seed = args.seed
use_backprop = args.use_backprop

datapath = (
    "/gpfsdswork/dataset/imagenet/RawImages/"  # Provide your path to IMAGENET here
)

save_adr = "results_imagenet/%s" % expe_name

try:
    os.makedirs(save_adr)
except:
    pass

n_iters = 102
lr = 0.1
train_accs, train_losss, test_accs, test_losss = main(
    datapath,
    save_adr,
    expe_name=expe_name,
    arch=model,
    n_workers=10,
    n_layers=n_layers,
    n_epochs=n_iters,
    batch_size=256,
    lr=lr,
    use_backprop=use_backprop,
)
