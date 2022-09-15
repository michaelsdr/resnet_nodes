import os
import datetime
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import copy

from resnet_ode.models import (
    IResNetAuto2,
    ResNetMemoryCustom,
)

device = "cuda"

n_workers = 10

n_layers = [64, 64, 64, 64]
net = IResNetAuto2
net = net(num_classes=1000, n_layers=n_layers)
"""
LOAD HERE the checkpoint e.g:
checkpoint =
torch.load("models/comparison_auto/results_imagenet_auto_compare/
comparison_auto/iresnet_0_nl_8/checkpoint.pth.tar")
"""
net = torch.nn.DataParallel(net).cuda()
net.load_state_dict(checkpoint["state_dict"])


net_2 = ResNetMemoryCustom(num_classes=1000, layers=n_layers, use_backprop=False)
net_2 = torch.nn.DataParallel(net_2).cuda()


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

## IMAGENET

# net = ResNet101
# net = net(num_classes=1000)
#
# checkpoint = torch.load('models/checkpoint_fine_tuning_imagenet/resnet101_v2_0.pth.tar')
#
# net = torch.nn.DataParallel(net).cuda()
# net.load_state_dict(checkpoint['state_dict'])


datapath = "/gpfsdswork/dataset/imagenet/RawImages/"
expe_name = "finetuning"
arch = net
batch_size = 256
gpu = None
weight_decay = 1e-4
print_freq = 100
momentum = 0.9
lr = 0.0005

criterion = nn.CrossEntropyLoss().cuda(gpu)
optimizer = torch.optim.SGD(
    net.parameters(), lr, momentum=momentum, weight_decay=weight_decay
)

traindir = os.path.join(datapath, "train")
valdir = os.path.join(datapath, "val")

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    ),
)

train_sampler = None

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=(train_sampler is None),
    num_workers=n_workers,
    pin_memory=True,
    sampler=train_sampler,
)

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(
        valdir,
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    ),
    batch_size=batch_size,
    shuffle=False,
    num_workers=n_workers,
    pin_memory=True,
)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if gpu is not None:
            images = images.cuda(gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)
    return losses.avg, top1.avg.item(), top5.avg.item()


def validate(val_loader, model, criterion):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix="Test: ",
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if gpu is not None:
                images = images.cuda(gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i)

        print(
            " * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(top1=top1, top5=top5)
        )

    return losses.avg, top1.avg.item(), top5.avg.item()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.33 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print("learning rate:", lr)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


train_losss, train_top1s, train_top5s = [], [], []
test_losss, test_top1s, test_top5s = [], [], []
start_epoch = 0
n_epochs = 16

saveadr = "results_fine_tuning_imagenet_adjoint/resnet_64_lr_0005"
model_save_adr = (
    "models/checkpoint_fine_tuning_imagenet_adjoint/resnet_64_lr_0005.pth.tar"
)


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)


def main():
    for epoch in range(start_epoch, start_epoch + n_epochs):
        t0 = time.time()
        adjust_learning_rate(optimizer, epoch, lr)

        # train for one epoch
        test_loss, test_top1, test_top5 = validate(val_loader, net, criterion)
        train_loss, train_top1, train_top5 = train(
            train_loader, net, criterion, optimizer, epoch
        )

        # evaluate on validation set

        train_losss.append(train_loss)
        train_top1s.append(train_top1)
        train_top5s.append(train_top5)
        test_losss.append(test_loss)
        test_top1s.append(test_top1)
        test_top5s.append(test_top5)
        print(test_loss, test_top1, test_top5)
        np.save(
            "%s.npy" % saveadr,
            np.array(
                [
                    train_losss,
                    train_top1s,
                    train_top5s,
                    test_losss,
                    test_top1s,
                    test_top5s,
                ]
            ),
        )
        print(
            "Epoch %d took %s"
            % (
                epoch,
                str(datetime.timedelta(seconds=int(time.time() - t0))),
            )
        )

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "arch": arch,
                "state_dict": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_loss": train_losss,
                "train_top1": train_top1s,
                "train_top5": train_top5s,
                "test_loss": test_losss,
                "test_top1": test_top1s,
                "test_top5": test_top5s,
            },
            filename=model_save_adr,
        )
    return train_top1s, train_losss, test_top1s, test_losss


main()
