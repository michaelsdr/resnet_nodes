"""Train CIFAR10 with PyTorch."""
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import copy

from models import *

# from utils import progress_bar


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
parser.add_argument("--depth", default=32, type=int)
parser.add_argument("--seed", default=1, type=int)
parser.add_argument("--base_depth", default=32, type=int)
parser.add_argument("--planes", default=16, type=int)
parser.add_argument("--mtinyresnet", default=False, type=bool)
parser.add_argument("--n_epochs", default=60, type=int)

args = parser.parse_args()
print(args.heun)
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


net_b = iTinyResnet(
    args.depth,
    in_planes=args.planes,
    zero_init=args.zero_init,
    use_heun=args.heun,
    use_backprop=True,
)

net_nb = iTinyResnet(
    args.depth,
    in_planes=args.planes,
    zero_init=args.zero_init,
    use_heun=args.heun,
    use_backprop=False,
)

net_b.to(device)
net_b.load_state_dict(net.state_dict())

net_nb.to(device)
net_nb.load_state_dict(net.state_dict())

for batch_idx, (inputs, targets) in enumerate(trainloader):
    inputs, targets = inputs.to(device), targets.to(device)
    outputs = net_b(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    outputs = net_nb(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    break


def cos(a, b):
    return torch.dot(a, b) / (torch.sqrt(torch.dot(a, a)) * torch.sqrt(torch.dot(b, b)))


def cos_sims_params(net_b, net_nb):
    coss = []
    norm_grads = []
    relative_norm_grads = []
    for (n1, p1), (n2, p2) in zip(net_b.named_parameters(), net_nb.named_parameters()):
        if "bias" not in n1:
            p1_rs = p1.grad.flatten()
            p2_rs = p2.grad.flatten()
            print(n1, p1_rs.norm(), p2_rs.norm())
            r_p = cos(p1_rs, p2_rs)
            coss.append(r_p.cpu())
            norm_grads.append(((p1_rs - p2_rs).square().mean()).cpu())
            relative_norm_grads.append(
                ((p1_rs - p2_rs).square().mean() / p1_rs.square().mean()).cpu()
            )
    return coss, norm_grads, relative_norm_grads


coss, norm_grads, relative_norm_grads = cos_sims_params(net_b, net_nb)
if args.heun:
    torch.save(
        [coss, norm_grads, relative_norm_grads],
        "metrics_gradient/depth_%s.to" % args.depth,
    )
else:
    torch.save(
        [coss, norm_grads, relative_norm_grads],
        "metrics_gradient/depth_%s_heun.to" % args.depth,
    )
