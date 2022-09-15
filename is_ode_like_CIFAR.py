import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import torchvision.transforms as transforms
import os
import copy

from resnet_ode.models import IResNetAuto2, ResNetMemoryCustom

device = "cuda" if torch.cuda.is_available() else "cpu"


def main(
    n_epochs=50,
    n_layer=32,
    lr=1e-3,
    save_adr="fine_tuning",
    checkpoint_dir="checkpoint_CIFAR_finetuning",
    seed=0,
):

    n_layers = [4, 4, n_layer, 4]
    n_workers = 8

    net = IResNetAuto2
    net = net(num_classes=10, n_layers=n_layers)

    checkpoint = torch.load("checkpoint_cifar_10.pth", map_location=device)
    net = torch.nn.DataParallel(net).to(device)
    net.load_state_dict(checkpoint["net"])

    net_2 = ResNetMemoryCustom(num_classes=10, layers=n_layers, use_backprop=False)
    net_2 = torch.nn.DataParallel(net_2).to(device)

    layer = net.module.layer1[0]
    net_2.module.layer1[0].conv1 = copy.deepcopy(layer.conv1)
    net_2.module.layer1[0].bn1 = copy.deepcopy(layer.bn1)
    net_2.module.layer1[0].conv2 = copy.deepcopy(layer.conv2)
    net_2.module.layer1[0].bn2 = copy.deepcopy(layer.bn2)
    net_2.module.layer1[0].conv3 = copy.deepcopy(layer.conv3)
    net_2.module.layer1[0].bn3 = copy.deepcopy(layer.bn3)
    net_2.module.layer1[0].downsample[0] = copy.deepcopy(layer.downsample[0])
    net_2.module.layer1[0].downsample[1] = copy.deepcopy(layer.downsample[1])

    layer = net.module.layer2[0]
    net_2.module.layer2[0].conv1 = copy.deepcopy(layer.conv1)
    net_2.module.layer2[0].bn1 = copy.deepcopy(layer.bn1)
    net_2.module.layer2[0].conv2 = copy.deepcopy(layer.conv2)
    net_2.module.layer2[0].bn2 = copy.deepcopy(layer.bn2)
    net_2.module.layer2[0].conv3 = copy.deepcopy(layer.conv3)
    net_2.module.layer2[0].bn3 = copy.deepcopy(layer.bn3)
    net_2.module.layer2[0].downsample[0] = copy.deepcopy(layer.downsample[0])
    net_2.module.layer2[0].downsample[1] = copy.deepcopy(layer.downsample[1])

    layer = net.module.layer3[0]
    net_2.module.layer3[0].conv1 = copy.deepcopy(layer.conv1)
    net_2.module.layer3[0].bn1 = copy.deepcopy(layer.bn1)
    net_2.module.layer3[0].conv2 = copy.deepcopy(layer.conv2)
    net_2.module.layer3[0].bn2 = copy.deepcopy(layer.bn2)
    net_2.module.layer3[0].conv3 = copy.deepcopy(layer.conv3)
    net_2.module.layer3[0].bn3 = copy.deepcopy(layer.bn3)
    net_2.module.layer3[0].downsample[0] = copy.deepcopy(layer.downsample[0])
    net_2.module.layer3[0].downsample[1] = copy.deepcopy(layer.downsample[1])

    layer = net.module.layer4[0]
    net_2.module.layer4[0].conv1 = copy.deepcopy(layer.conv1)
    net_2.module.layer4[0].bn1 = copy.deepcopy(layer.bn1)
    net_2.module.layer4[0].conv2 = copy.deepcopy(layer.conv2)
    net_2.module.layer4[0].bn2 = copy.deepcopy(layer.bn2)
    net_2.module.layer4[0].conv3 = copy.deepcopy(layer.conv3)
    net_2.module.layer4[0].bn3 = copy.deepcopy(layer.bn3)
    net_2.module.layer4[0].downsample[0] = copy.deepcopy(layer.downsample[0])
    net_2.module.layer4[0].downsample[1] = copy.deepcopy(layer.downsample[1])

    layer = net.module.layer1[1].function
    for i, _ in enumerate(net_2.module.layer1[1].network.functions):
        net_2.module.layer1[1].network.functions[i].conv1 = copy.deepcopy(layer.conv1)
        net_2.module.layer1[1].network.functions[i].bn1 = copy.deepcopy(layer.bn1)
        net_2.module.layer1[1].network.functions[i].conv2 = copy.deepcopy(layer.conv2)
        net_2.module.layer1[1].network.functions[i].bn2 = copy.deepcopy(layer.bn2)
        net_2.module.layer1[1].network.functions[i].conv3 = copy.deepcopy(layer.conv3)
        net_2.module.layer1[1].network.functions[i].bn3 = copy.deepcopy(layer.bn3)

    layer = net.module.layer2[1].function
    for i, _ in enumerate(net_2.module.layer2[1].network.functions):
        net_2.module.layer2[1].network.functions[i].conv1 = copy.deepcopy(layer.conv1)
        net_2.module.layer2[1].network.functions[i].bn1 = copy.deepcopy(layer.bn1)
        net_2.module.layer2[1].network.functions[i].conv2 = copy.deepcopy(layer.conv2)
        net_2.module.layer2[1].network.functions[i].bn2 = copy.deepcopy(layer.bn2)
        net_2.module.layer2[1].network.functions[i].conv3 = copy.deepcopy(layer.conv3)
        net_2.module.layer2[1].network.functions[i].bn3 = copy.deepcopy(layer.bn3)

    layer = net.module.layer3[1].function
    for i, _ in enumerate(net_2.module.layer3[1].network.functions):
        net_2.module.layer3[1].network.functions[i].conv1 = copy.deepcopy(layer.conv1)
        net_2.module.layer3[1].network.functions[i].bn1 = copy.deepcopy(layer.bn1)
        net_2.module.layer3[1].network.functions[i].conv2 = copy.deepcopy(layer.conv2)
        net_2.module.layer3[1].network.functions[i].bn2 = copy.deepcopy(layer.bn2)
        net_2.module.layer3[1].network.functions[i].conv3 = copy.deepcopy(layer.conv3)
        net_2.module.layer3[1].network.functions[i].bn3 = copy.deepcopy(layer.bn3)

    layer = net.module.layer4[1].function
    for i, _ in enumerate(net_2.module.layer4[1].network.functions):
        net_2.module.layer4[1].network.functions[i].conv1 = copy.deepcopy(layer.conv1)
        net_2.module.layer4[1].network.functions[i].bn1 = copy.deepcopy(layer.bn1)
        net_2.module.layer4[1].network.functions[i].conv2 = copy.deepcopy(layer.conv2)
        net_2.module.layer4[1].network.functions[i].bn2 = copy.deepcopy(layer.bn2)
        net_2.module.layer4[1].network.functions[i].conv3 = copy.deepcopy(layer.conv3)
        net_2.module.layer4[1].network.functions[i].bn3 = copy.deepcopy(layer.bn3)

    net_2.module.conv1 = copy.deepcopy(net.module.conv1)

    net_2.module.bn1 = copy.deepcopy(net.module.bn1)

    net_2.module.maxpool = copy.deepcopy(net.module.maxpool)

    net_2.module.avgpool = copy.deepcopy(net.module.avgpool)

    net_2.module.fc = copy.deepcopy(net.module.fc)

    net = net_2

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
        trainset, batch_size=128, shuffle=True, num_workers=n_workers
    )

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=2048, shuffle=False, num_workers=n_workers
    )

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = optim.SGD(
        net.module.layer3.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=5e-4,
    )

    def train():
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
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
        return train_loss / (batch_idx + 1), 100.0 * correct / total

    def test():
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

    train_losss, train_accs, test_losss, test_accs = [], [], [], []

    for epoch in range(n_epochs):
        if epoch == 40:
            print("diminushing lr")
            for i, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr / 10
        train_loss, train_acc = train()
        print("train", train_loss, train_acc)
        test_loss, test_acc = test()
        train_losss.append(train_loss)
        train_losss.append(train_acc)
        test_losss.append(test_loss)
        test_accs.append(test_acc)
        metrics = np.array([train_accs, train_losss, test_accs, test_losss])
        if save_adr is not None:
            print("saving")
            np.save(
                "results_finetuning_CIFAR/%s_nl_%d_seed_%d.npy"
                % (save_adr, n_layer, seed),
                metrics,
            )
        state = {
            "net": net.state_dict(),
            "acc": test_acc,
            "epoch": epoch,
            "metrics": metrics,
        }
        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        torch.save(
            state,
            "%s/%s_nl_%d_seed_%d.pth" % (checkpoint_dir, save_adr, n_layer, seed),
        )

    return train_accs, train_losss, test_accs, test_losss
