import argparse
import os


parser = argparse.ArgumentParser(description="test")
parser.add_argument("--backprop", default=1, type=int)
parser.add_argument("--heun", default=0, type=int)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--depth", default=2, type=int)

args = parser.parse_args()

backprop = args.backprop
heun = args.heun
if backprop == 0:
    backprop = False
else:
    backprop = True

if heun == 0:
    heun = False
else:
    heun = True
depth = args.depth
seed = args.seed
n_epochs = 90
n_epochs_ft = 60
print(args.heun)

if __name__ == "__main__":
    if heun and backprop:
        os.system(
            "python train_tiny.py --depth %d --n_epochs %d --backprop --heun --seed %d"
            % (depth, n_epochs, seed)
        )
    elif heun and backprop is False:
        os.system(
            "python train_tiny.py --depth %d --n_epochs %d --heun --seed %d"
            % (depth, n_epochs, seed)
        )
    elif heun is False and backprop:
        os.system(
            "python train_tiny.py --depth %d --n_epochs %d --backprop --seed %d"
            % (depth, n_epochs, seed)
        )
    else:
        os.system(
            "python train_tiny.py --depth %d --n_epochs %d --seed %d"
            % (depth, n_epochs, seed)
        )
