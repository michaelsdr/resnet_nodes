"""
Adapted from https://github.com/kuangliu/pytorch-cifar
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import torchvision.transforms as transforms
import os

from .models import (
    ResNet101,
    IResNet101,
    ResNet18,
    IResNet18,
    IResNet34,
    ResNet34,
    IResNet152,
    ResNet152,
    ResNetAuto,
    IResNetAuto,
    ResNetMemory18,
    ResNetMemory101,
    ResNetMemoryCustom,
    ResNetLipMemoryCustom,
)

n_workers = 10


def train_resnet(
    lr_list,
    model="resnet18",
    cifar100=False,
    save_adr=None,
    seed=0,
    save=True,
    n_layers=1,
    save_at="results",
    checkpoint_dir="checkpoint_CIFAR10_resnet",
    adaptative_lr=False,
    use_backprop=True,
    use_lr_schedule=False,
):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Data
    expe_name = "%s_ckpt_model_%s_seed_%d_nl_%d.pth" % (
        save_at,
        model,
        seed,
        n_layers,
    )
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
    if cifar100:
        Loader = torchvision.datasets.CIFAR100
        root = ".data/CIFAR100"
    else:
        Loader = torchvision.datasets.CIFAR10
        root = ".data/CIFAR10"
    trainset = Loader(
        root=root,
        train=True,
        download=True,
        transform=transform_train,
    )
    testset = Loader(
        root=root,
        train=False,
        download=True,
        transform=transform_test,
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=128,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True,
    )

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=2048, shuffle=False, num_workers=n_workers
    )

    # Model
    print("==> Building model..")
    print(model)
    if model == "iresnet18":
        net = IResNet18
    if model == "resnet18":
        net = ResNet18
    if model == "iresnet34":
        net = IResNet34
    if model == "resnet34":
        net = ResNet34
    if model == "iresnet101":
        net = IResNet101
    if model == "resnet101":
        net = ResNet101
    if model == "iresnet152":
        net = IResNet152
    if model == "resnet152":
        net = ResNet152
    if model == "resnetauto":
        net = ResNetAuto
    if model == "iresnetauto":
        net = IResNetAuto
    if model == "resnetmemory":
        net = ResNetMemory101
    if model == "resnetmemory18":
        net = ResNetMemory18
    if model == "resnetmemory101":
        net = ResNetMemory101
    if model == "resnetmemorycustom":
        net = ResNetMemoryCustom
    if model == "resnetmemorycustom":
        net = ResNetMemoryCustom
    if model == "resnetmemorycifar":
        net = ResNetMemoryCIFAR
    if model == "resnetlipmemorycustom":
        net = ResNetLipMemoryCustom
    num_classes = 100 if cifar100 else 10
    if model == "iresnetauto":
        net = net(num_classes=num_classes, n_layers=n_layers)
    elif model == "resnetmemory101" or model == "resnetmemory18":
        net = net(
            num_classes=num_classes,
            zero_init_residual=True,
            use_backprop=True,
        )
        print("zero_init_residual")
    elif model == "resnetmemorycustom" or model == "resnetlipmemorycustom":
        net = net(
            num_classes=num_classes,
            zero_init_residual=False,
            use_backprop=True,
            layers=[3, 4, n_layers, 3],
        )
        print("custom", "zero_init_residual")
    elif model == "resnetmemorycifar":
        net = net(
            num_classes=num_classes,
            zero_init_residual=True,
            use_backprop=use_backprop,
            layers=[n_layers],
        )
        print("memorycifar", "zero_init_residual")
    else:
        net = net(num_classes=num_classes)
    net = net.to(device)
    print(net)
    if device == "cuda":
        net = torch.nn.DataParallel(net).cuda()
    resume = os.path.isdir(checkpoint_dir)
    ep = 0
    train_accs = []
    train_losss = []
    test_losss = []
    test_accs = []
    if resume == True:
        assert os.path.isdir(checkpoint_dir), "Error: no checkpoint directory found!"
        try:
            to_load = "./%s/%s" % (checkpoint_dir, expe_name)
            checkpoint = torch.load(to_load)
            net.load_state_dict(checkpoint["net"])
            ep = checkpoint["epoch"]
            metrics = checkpoint["metrics"]
            train_accs, train_losss, test_accs, test_losss = (
                list(metrics[0]),
                list(metrics[1]),
                list(metrics[2]),
                list(metrics[3]),
            )
            print("==> Resuming from checkpoint..")
        except:
            pass

    optimizer = optim.SGD(
        net.parameters(),
        lr=lr_list[0],
        momentum=0.9,
        weight_decay=5e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(lr_list),
    )

    if adaptative_lr is True and use_lr_schedule is False:
        print("test")

        layer_1_named_params = list(
            dict(net.module.layer1[1].named_parameters()).keys()
        )
        my_list = [
            "module.layer1.1." + layer_1_named_params[i]
            for i in range(len(layer_1_named_params))
        ]
        params = list(filter(lambda kv: kv[0] in my_list, net.named_parameters()))
        base_params = list(
            filter(
                lambda kv: kv[0] not in my_list,
                net.named_parameters(),
            )
        )

        params = dict(params).values()
        base_params = dict(base_params).values()
        lr_mult = len(net.module.layer1[1].network.functions)
        lr = lr_list[0]
        optimizer = torch.optim.SGD(
            [
                {"params": base_params},
                {"params": params, "lr": lr_mult * lr},
            ],
            lr=lr,
            momentum=0.9,
            weight_decay=5e-4,
        )

    criterion = nn.CrossEntropyLoss().cuda()
    # Training
    def train(net, trainloader, epoch):
        print("\nEpoch: %d" % epoch)
        if use_lr_schedule is False:
            for i, param_group in enumerate(optimizer.param_groups):
                if i == 0:
                    param_group["lr"] = lr_list[epoch]
                else:
                    param_group["lr"] = lr_list[epoch] * lr_mult
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            # print(dict(net.named_parameters())['module.layer3.1.network.0.bn3.weight'])
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print(
            "Epoch %d: %.2e, %.2e"
            % (
                epoch,
                train_loss / (batch_idx + 1),
                100.0 * correct / total,
            )
        )
        return train_loss / (batch_idx + 1), 100.0 * correct / total

    def test(epoch):
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

        print(
            "Test  : %.2e, %.2e"
            % (test_loss / (batch_idx + 1), 100.0 * correct / total)
        )

        return test_loss / (batch_idx + 1), 100.0 * correct / total

    for epoch in range(ep, len(lr_list)):
        train_loss, train_acc = train(net, trainloader, epoch)
        test_loss, test_acc = test(epoch)
        train_losss.append(train_loss)
        train_accs.append(train_acc)
        test_losss.append(test_loss)
        test_accs.append(test_acc)
        metrics = np.array([train_accs, train_losss, test_accs, test_losss])
        if use_lr_schedule:
            scheduler.step()
            for param_groups in optimizer.param_groups:
                print("lr= ", param_groups["lr"])
        if save == True:
            if save_adr is not None:
                print("saving")
                np.save(save_adr, metrics)
            state = {
                "net": net.state_dict(),
                "acc": test_acc,
                "epoch": epoch,
                "metrics": metrics,
            }
            if not os.path.isdir(checkpoint_dir):
                os.mkdir(checkpoint_dir)
            torch.save(state, "./%s/%s" % (checkpoint_dir, expe_name))

    return train_accs, train_losss, test_accs, test_losss
