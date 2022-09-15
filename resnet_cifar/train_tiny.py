"""
Adapted from https://github.com/kuangliu/pytorch-cifar
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import copy

from models import *


parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
parser.add_argument(
    "--resume", "-r", action="store_true", help="resume from checkpoint"
)
parser.add_argument("--zero-init", "-z", default=True, type=bool)
parser.add_argument("--adaptive", action="store_true")
parser.add_argument("--backprop", action="store_true")
parser.add_argument("--heun", action="store_true")
parser.add_argument("--device", "-d", default=0, type=int)
parser.add_argument("--depth", default=8, type=int)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--base_depth", default=8, type=int)
parser.add_argument("--planes", default=16, type=int)
parser.add_argument("--mtinyresnet", default=False, type=bool)
parser.add_argument("--n_epochs", default=90, type=int)


def main():
    args = parser.parse_args()
    seed = args.seed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device > 0:
        device = "cuda:%d" % args.device
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print("==> Preparing data..")
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=".data/CIFAR10", train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=256, shuffle=True, num_workers=10
    )

    testset = torchvision.datasets.CIFAR10(
        root=".data/CIFAR10", train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    bn_dict = {True: "with_bn", False: "no_bn"}
    zero_dict = {True: "zero_init", False: "no_init"}
    # expe_name = "%s_%s_%s" % (bn_dict[args.use_bn], zero_dict[args.zero_init], str(time.time()))
    # Model
    print("==> Building model..")
    net = iTinyResnet(
        args.depth,
        in_planes=args.planes,
        zero_init=args.zero_init,
        use_heun=args.heun,
        use_backprop=args.backprop,
    )
    net_base = iTinyResnet(
        args.base_depth,
        in_planes=args.planes,
        zero_init=args.zero_init,
        use_heun=args.heun,
        use_backprop=args.backprop,
    )
    if args.mtinyresnet:
        net = mTinyResnet(
            args.depth,
            in_planes=args.planes,
            zero_init=args.zero_init,
            use_heun=args.heun,
            use_backprop=args.backprop,
        )
        net_base = mTinyResnet(
            args.base_depth,
            in_planes=args.planes,
            zero_init=args.zero_init,
            use_heun=args.heun,
            use_backprop=args.backprop,
        )

    net = net.to(device)
    net_base = net_base.to(device)
    cudnn.benchmark = True

    log_dict = {"test_loss": [], "train_loss": [], "test_acc": [], "train_acc": []}

    if args.resume:
        # Load checkpoint.
        print("==> Resuming from checkpoint..")
        assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
        checkpoint = torch.load(
            "./checkpoint/ckpt_%s_heun_%s_backprop_%s_seed_%s.pth"
            % (args.base_depth, args.heun, args.backprop, seed)
        )
        net_base.load_state_dict(checkpoint["net"])
        best_acc = checkpoint["acc"]
        start_epoch = checkpoint["epoch"]
        log_dict = checkpoint["log_dict"]

    if args.mtinyresnet:
        net.conv1 = copy.deepcopy(net_base.conv1)
        net.bn1 = copy.deepcopy(net_base.bn1)
        net.linear = copy.deepcopy(net_base.linear)
        ratio = args.base_depth / args.depth
        for i in range(args.depth):
            target_layer = net_base.residual_layers.functions[int(i * ratio)]
            net.residual_layers.functions[i].conv1 = copy.deepcopy(target_layer.conv1)
            net.residual_layers.functions[i].bn1 = copy.deepcopy(target_layer.bn1)
            net.residual_layers.functions[i].conv2 = copy.deepcopy(target_layer.conv2)
            net.residual_layers.functions[i].bn2 = copy.deepcopy(target_layer.bn2)

    else:
        net.conv1 = copy.deepcopy(net_base.conv1)
        net.bn1 = copy.deepcopy(net_base.bn1)
        net.linear = copy.deepcopy(net_base.linear)
        ratio = args.base_depth / args.depth
        for i in range(args.depth):
            target_layer = net_base.residual_layers[int(i * ratio)]
            net.residual_layers[i].conv1 = copy.deepcopy(target_layer.conv1)
            net.residual_layers[i].bn1 = copy.deepcopy(target_layer.bn1)
            net.residual_layers[i].conv2 = copy.deepcopy(target_layer.conv2)
            net.residual_layers[i].bn2 = copy.deepcopy(target_layer.bn2)

    criterion = nn.CrossEntropyLoss()

    residual_parameters = []
    other_parameters = []
    for name, param in net.named_parameters():
        if "residual_layers" in name:
            residual_parameters.append(param)
        else:
            other_parameters.append(param)
    optimizer = optim.SGD(other_parameters, lr=args.lr, momentum=0.9, weight_decay=5e-4)

    residual_lr = args.lr * args.depth if args.adaptive else args.lr
    optimizer_residual = optim.SGD(
        residual_parameters, lr=residual_lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # Training
    def train(epoch):
        print("\nEpoch: %d" % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            optimizer_residual.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            optimizer_residual.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print(train_loss / (batch_idx + 1), 100.0 * correct / total, correct, total)

        train_loss = train_loss / (batch_idx + 1)
        train_acc = 100.0 * correct / total
        log_dict["train_loss"].append(train_loss)
        log_dict["train_acc"].append(train_acc)

    def test(epoch):
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        print(test_loss / (batch_idx + 1), 100.0 * correct / total)

        # Save checkpoint.
        acc = 100.0 * correct / total
        test_loss = test_loss / (batch_idx + 1)
        test_acc = 100.0 * correct / total
        log_dict["test_loss"].append(test_loss)
        log_dict["test_acc"].append(test_acc)
        print("Saving..")
        state = {
            "net": net.state_dict(),
            "acc": acc,
            "epoch": epoch,
            "log_dict": log_dict,
        }
        if not os.path.isdir("checkpoint"):
            os.mkdir("checkpoint")
        torch.save(
            state,
            "./checkpoint/ckpt_%s_heun_%s_backprop_%s_seed_%s.pth"
            % (args.depth, args.heun, args.backprop, seed),
        )
        best_acc = acc

    for epoch in range(start_epoch, start_epoch + args.n_epochs):
        train(epoch)
        test(epoch)
        if epoch == args.n_epochs // 3:
            for g in optimizer.param_groups:
                g["lr"] /= 10
            for g in optimizer_residual.param_groups:
                g["lr"] /= 10
        if epoch == 2 * args.n_epochs // 3:
            for g in optimizer.param_groups:
                g["lr"] /= 10
            for g in optimizer_residual.param_groups:
                g["lr"] /= 10


if __name__ == "__main__":
    main()
