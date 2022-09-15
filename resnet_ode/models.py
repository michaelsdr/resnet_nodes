"""
Adapted from https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html
"""

import torch
import torch.nn as nn
from .implicit_net import ImplicitNet
from .resnet_memory import ResNetBack

import numpy as np


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        depth=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.depth = depth

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        if self.depth is not None:
            out *= 1 / self.depth
        out += identity
        if self.depth is None:
            out = self.relu(out)

        return out


class MBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(MBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        depth=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.depth = depth

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        if self.depth is not None:
            out *= 1 / self.depth
        out += identity
        if self.depth is None:
            out = self.relu(out)
        return out


class MBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(MBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.bn3.use_weight_decay = True
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super(ResNet, self).__init__()
        self.large = num_classes == 1000
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        if self.large:
            self.conv1 = nn.Conv2d(
                3,
                self.inplanes,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
        else:
            self.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride=1,
        dilate=False,
        depth=None,
    ):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                depth=None,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    depth=blocks - 1,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.large:
            x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


class ImplicitResNet(nn.Module):
    def __init__(
        self,
        block,
        mblock,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        rms=False,
    ):
        super(ImplicitResNet, self).__init__()
        self.large = num_classes == 1000
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        if self.large:
            self.conv1 = nn.Conv2d(
                3,
                self.inplanes,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
        else:
            self.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, mblock, 64, layers[0], rms=rms)
        self.layer2 = self._make_layer(
            block,
            mblock,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
            rms=rms,
        )
        self.layer3 = self._make_layer(
            block,
            mblock,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            rms=rms,
        )
        self.layer4 = self._make_layer(
            block,
            mblock,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
            rms=rms,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block,
        mblock,
        planes,
        blocks,
        stride=1,
        dilate=False,
        rms=False,
    ):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        first_function = block(
            self.inplanes,
            planes,
            stride,
            downsample,
            self.groups,
            self.base_width,
            previous_dilation,
            norm_layer,
        )
        self.inplanes = planes * block.expansion
        Maker = ImplicitNet
        function = mblock(
            self.inplanes,
            planes,
            groups=self.groups,
            base_width=self.base_width,
            dilation=self.dilation,
            norm_layer=norm_layer,
        )
        return nn.Sequential(
            first_function,
            Maker(function, n_iters=blocks - 1, rms=rms),
        )

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.large:
            x = self.maxpool(x)
        x = self.layer1(x)
        x = self.relu2(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        x = self.relu2(x)
        x = self.layer4(x)
        x = self.relu2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


class ResNetMemory(nn.Module):
    def __init__(
        self,
        block,
        mblock,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        use_backprop=False,
        init_fixup=False,
        lip=False,
        normalize=True,
        depth=None,
    ):
        super(ResNetMemory, self).__init__()
        self.num_layers = sum(layers)
        self.large = num_classes == 1000
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        self.lip = lip
        self.normalize = normalize
        self.use_backprop = use_backprop
        self.depth = depth
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be"
                " None or a 3-element tuple, got {}".format(
                    replace_stride_with_dilation
                )
            )
        self.groups = groups
        self.base_width = width_per_group
        if self.large:
            self.conv1 = nn.Conv2d(
                3,
                self.inplanes,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
        else:
            self.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, mblock, 64, layers[0])
        self.layer2 = self._make_layer(
            block,
            mblock,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
        )
        self.layer3 = self._make_layer(
            block,
            mblock,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            depth=self.depth,
        )
        self.layer4 = self._make_layer(
            block,
            mblock,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros,
        # and each residual block behaves like an identity.
        # This improves the model by
        # 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, MBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, MBottleneck):
        #             nn.init.constant_(m.conv3.weight, 0)
        if init_fixup:
            for m in self.modules():
                if isinstance(m, MBottleneck):
                    nn.init.normal_(
                        m.conv1.weight,
                        mean=0,
                        std=np.sqrt(
                            2
                            / (
                                m.conv1.weight.shape[0]
                                * np.prod(m.conv1.weight.shape[2:])
                            )
                        )
                        * self.num_layers ** (-0.25),
                    )
                    nn.init.normal_(
                        m.conv2.weight,
                        mean=0,
                        std=np.sqrt(
                            2
                            / (
                                m.conv2.weight.shape[0]
                                * np.prod(m.conv2.weight.shape[2:])
                            )
                        )
                        * self.num_layers ** (-0.25),
                    )
                    nn.init.constant_(m.conv3.weight, 0)
                    if m.downsample is not None:
                        nn.init.normal_(
                            m.downsample.weight,
                            mean=0,
                            std=np.sqrt(
                                2
                                / (
                                    m.downsample.weight.shape[0]
                                    * np.prod(m.downsample.weight.shape[2:])
                                )
                            ),
                        )
                elif isinstance(m, nn.Linear):
                    nn.init.constant_(m.weight, 0)
                    nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block,
        mblock,
        planes,
        blocks,
        stride=1,
        dilate=False,
        depth=None,
    ):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                depth=depth,
            )
        )
        self.inplanes = planes * block.expansion
        if blocks < 2:
            return nn.Sequential(
                layers[0],
            )
        for _ in range(1, blocks):
            layers.append(
                mblock(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        if self.lip:
            Maker = ResNetLipBack
            return nn.Sequential(
                layers[0],
                Maker(
                    layers[1:],
                    use_backprop=self.use_backprop,
                    normalize=self.normalize,
                ),
            )
        else:
            Maker = ResNetBack
            return nn.Sequential(
                layers[0],
                Maker(layers[1:], use_backprop=self.use_backprop),
            )

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.large:
            x = self.maxpool(x)
        x = self.layer1(x)
        x = self.relu2(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.layer3(x)
        x = self.relu2(x)
        x = self.layer4(x)
        x = self.relu2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


def ResNet18(num_classes=1000, bn=True):
    if bn:
        norm_layer = None
    else:
        norm_layer = nn.Identity
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, norm_layer=norm_layer)


def ResNetMemory18(
    num_classes=1000,
    use_backprop=True,
    init_fixup=False,
    zero_init_residual=False,
):
    return ResNetMemory(
        BasicBlock,
        MBasicBlock,
        [2, 2, 2, 2],
        num_classes,
        use_backprop=use_backprop,
        init_fixup=init_fixup,
        zero_init_residual=zero_init_residual,
    )


def IResNet18(num_classes=1000, bn=True):
    if bn:
        norm_layer = None
    else:
        norm_layer = nn.Identity
    return ImplicitResNet(
        BasicBlock,
        MBasicBlock,
        [2, 2, 2, 2],
        num_classes,
        norm_layer=norm_layer,
    )


def ResNet34(num_classes=1000):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def IResNet34(num_classes=1000, bn=True):
    if bn:
        norm_layer = None
    else:
        norm_layer = nn.Identity
    return ImplicitResNet(
        BasicBlock,
        MBasicBlock,
        [3, 4, 6, 3],
        num_classes,
        norm_layer=norm_layer,
    )


def ResNet101(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)


def ResNetMemory101(
    num_classes=1000,
    use_backprop=False,
    init_fixup=False,
    zero_init_residual=False,
):
    return ResNetMemory(
        Bottleneck,
        MBottleneck,
        [3, 4, 23, 3],
        num_classes,
        use_backprop=use_backprop,
        init_fixup=init_fixup,
        zero_init_residual=zero_init_residual,
    )


def ResNetMemoryCustom(
    num_classes=1000,
    layers=[3, 4, 10, 3],
    use_backprop=False,
    init_fixup=False,
    zero_init_residual=False,
    depth=None,
):
    return ResNetMemory(
        Bottleneck,
        MBottleneck,
        layers,
        num_classes,
        use_backprop=use_backprop,
        init_fixup=init_fixup,
        zero_init_residual=zero_init_residual,
        depth=depth,
    )


def ResNetLipMemoryCustom(
    num_classes=1000,
    layers=[2, 2, 10, 2],
    use_backprop=False,
    init_fixup=False,
    zero_init_residual=False,
    normalize=True,
    depth=None,
):
    return ResNetMemory(
        Bottleneck,
        MBottleneck,
        layers,
        num_classes,
        use_backprop=use_backprop,
        init_fixup=init_fixup,
        zero_init_residual=zero_init_residual,
        lip=True,
        normalize=normalize,
        depth=depth,
    )


def IResNet101(num_classes=1000):
    return ImplicitResNet(
        Bottleneck,
        MBottleneck,
        [3, 4, 23, 3],
        num_classes=num_classes,
    )


def ResNet152(num_classes=1000):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)


def ResNetDeep(num_classes=1000, n_layers=36):
    return ResNet(Bottleneck, [3, 8, n_layers, 3], num_classes)


def IResNet152(num_classes=1000):
    return ImplicitResNet(
        Bottleneck,
        MBottleneck,
        [3, 8, 36, 3],
        num_classes=num_classes,
    )


def ResNetAuto(num_classes=1000, n_layers=1):
    return ResNet(
        Bottleneck,
        [n_layers, n_layers, n_layers, n_layers],
        num_classes,
    )


def IResNetAuto(num_classes=1000, n_layers=1, rms=False):
    if n_layers > 2:
        return ImplicitResNet(
            Bottleneck,
            MBottleneck,
            [n_layers, n_layers, n_layers, n_layers],
            num_classes=num_classes,
            rms=rms,
        )
    else:
        return ImplicitResNet(
            BasicBlock,
            MBasicBlock,
            [n_layers, n_layers, n_layers, n_layers],
            num_classes=num_classes,
            rms=rms,
        )


def IResNetAuto2(num_classes=1000, n_layers=[10, 10, 10, 10], rms=False):
    return ImplicitResNet(
        Bottleneck,
        MBottleneck,
        n_layers,
        num_classes=num_classes,
        rms=rms,
    )
