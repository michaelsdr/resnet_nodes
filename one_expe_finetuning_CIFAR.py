import argparse
from is_ode_like_CIFAR import main


parser = argparse.ArgumentParser(description="test")
parser.add_argument("-n", default=50, type=int)
parser.add_argument("-s", default="fine_tuning")
parser.add_argument("-c", default="checkpoint_CIFAR_finetuning")
parser.add_argument("-l", default=32, type=int)
parser.add_argument("--seed", default=0, type=int)
args = parser.parse_args()


n_epochs = args.n
save_adr = args.s
n_layer = args.l
checkpoint_dir = args.c
seed = args.seed

lr = 1e-3
if __name__ == "__main__":
    train_accs, train_losss, test_accs, test_losss = main(
        n_epochs=n_epochs,
        n_layer=n_layer,
        lr=lr,
        save_adr=save_adr,
        checkpoint_dir=checkpoint_dir,
        seed=seed,
    )
