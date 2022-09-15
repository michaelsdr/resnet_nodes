"""ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Code adapted from https://github.com/kuangliu/pytorch-cifar

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from memcnn import InvertibleModuleWrapper


class HeunFwd(nn.Module):
    def __init__(self, function1, function2, depth):
        super().__init__()
        self.function1 = function1
        self.function2 = function2
        self.depth = depth

    def forward(self, x):
        depth = self.depth
        f1 = self.function1(x)
        y = x + f1 / depth
        return x + (f1 + self.function2(y)) / 2 / depth

    def inverse(self, x):
        depth = self.depth
        f2 = self.function2(x)
        y = x - f2 / depth
        return x - (f2 + self.function1(y)) / 2 / depth


class EulerFwd(nn.Module):
    def __init__(self, function, depth):
        super().__init__()
        self.function = function
        self.depth = depth

    def forward(self, x):
        depth = self.depth
        return x + self.function(x) / depth

    def inverse(self, x):
        depth = self.depth
        return x - self.function(x) / depth


class InvertibleNet(nn.Module):
    def __init__(self, functions, use_backprop=True, num_bwd_passes=1, use_heun=False):
        super().__init__()
        for i, function in enumerate(functions):
            self.add_module(str(i), function)
        self.functions = functions
        self.depth = len(functions)
        depth = len(functions)
        self.use_heun = use_heun
        self.use_backprop = use_backprop
        disable = use_backprop
        if self.use_heun:
            self.nets = [
                InvertibleModuleWrapper(
                    HeunFwd(functions[i], functions[i + 1], depth),
                    num_bwd_passes=num_bwd_passes,
                    disable=disable,
                )
                for i in range(depth - 1)
            ]
        else:
            self.nets = [
                InvertibleModuleWrapper(
                    EulerFwd(functions[i], depth),
                    num_bwd_passes=num_bwd_passes,
                    disable=disable,
                )
                for i in range(depth)
            ]

    def forward(self, x):
        for net in self.nets:
            x = net(x.clone())
        return x

    def inverse(self, x):
        for net in self.nets[::-1]:
            x = net.inverse(x)
        return x

    def __getitem__(self, idx):
        return self.functions[idx]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, in_planes, planes, stride=1, zero_conv_init=False, use_batch_norm=True
    ):
        super(BasicBlock, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        if use_batch_norm:
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.conv3 = nn.Conv2d(
                in_planes,
                self.expansion * planes,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
            if use_batch_norm:
                self.shortcut = nn.Sequential(
                    self.conv3, nn.BatchNorm2d(self.expansion * planes)
                )
            else:
                self.shortcut = self.conv3

        else:
            if zero_conv_init:
                nn.init.constant_(self.conv2.weight, 0)

    def forward(self, x):
        if self.use_batch_norm:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
        else:
            out = F.relu(self.conv1(x))
            out = self.conv2(out)
        out += self.shortcut(x)
        # out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        num_classes=10,
        zero_conv_init=False,
        use_batch_norm=True,
    ):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        if use_batch_norm:
            self.bn1 = nn.BatchNorm2d(64)
        self.use_batch_norm = use_batch_norm
        self.layer1 = self._make_layer(
            block,
            64,
            num_blocks[0],
            stride=1,
            zero_conv_init=zero_conv_init,
            use_batch_norm=use_batch_norm,
        )
        self.layer2 = self._make_layer(
            block,
            128,
            num_blocks[1],
            stride=2,
            zero_conv_init=zero_conv_init,
            use_batch_norm=use_batch_norm,
        )
        self.layer3 = self._make_layer(
            block,
            256,
            num_blocks[2],
            stride=2,
            zero_conv_init=zero_conv_init,
            use_batch_norm=use_batch_norm,
        )
        self.layer4 = self._make_layer(
            block,
            512,
            num_blocks[3],
            stride=2,
            zero_conv_init=zero_conv_init,
            use_batch_norm=use_batch_norm,
        )
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self,
        block,
        planes,
        num_blocks,
        stride,
        zero_conv_init=False,
        use_batch_norm=True,
    ):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    stride,
                    zero_conv_init=zero_conv_init,
                    use_batch_norm=use_batch_norm,
                )
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.use_batch_norm:
            out = F.relu(self.bn1(self.conv1(x)))
        else:
            out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class TinyResidual(nn.Module):
    expansion = 1

    def __init__(
        self,
        planes,
        n_layers,
        stride=1,
        zero_conv_init=False,
        use_relu=True,
        use_bn=True,
    ):
        super(TinyResidual, self).__init__()
        self.conv1 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=True
        )
        if use_bn:
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=True
        )
        # self.conv1 = nn.Identity()
        if zero_conv_init:
            if use_bn:
                nn.init.constant_(self.bn2.weight, 0)
                nn.init.constant_(self.bn2.bias, 0)
            # nn.init.xavier_uniform_(self.conv2.weight, gain=1e-4)
            # nn.init.xavier_uniform_(self.conv1.weight, gain=1e-4)
        self.n_layers = n_layers
        self.use_relu = use_relu
        self.use_bn = use_bn
        if self.use_relu:
            self.non_lin = nn.ReLU()
        else:
            self.non_lin = nn.Identity()

    def forward(self, x):
        out = self.non_lin(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return out


class ResNetTransformMemory(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, functions, n_fun_args, *params):
        fun_args = params[:n_fun_args]
        ctx.functions = functions

        ctx.fun_args = fun_args
        ctx.params_require_grad = [
            param.requires_grad for param in params if param is not None
        ]
        n_iters = len(functions)
        with torch.no_grad():
            for i in range(n_iters):
                f = functions[i](x, *fun_args)
                x = x + f / n_iters
                if i == n_iters - 1:
                    ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        functions = ctx.functions
        fun_args = ctx.fun_args
        fun_args_requires_grad = [param.requires_grad for param in fun_args]
        n_fun_grad = sum(fun_args_requires_grad)
        params_require_grad = ctx.params_require_grad
        n_iters = len(functions)
        x = ctx.saved_tensors[0]
        grad_x = grad_output
        grad_params = []
        with torch.set_grad_enabled(True):
            for i in range(n_iters):
                function = functions[n_iters - 1 - i]
                x = x.detach().requires_grad_(False)
                with torch.no_grad():
                    f_eval = function(x, *fun_args) / n_iters
                    x = x - f_eval.data
                x = x.detach().requires_grad_(True)
                f_eval = function(x, *fun_args) / n_iters
                backward_list = []
                for requires_grad, param in zip(
                    params_require_grad,
                    fun_args + tuple(find_parameters(function)),
                ):
                    if requires_grad:
                        backward_list.append(param)
                vjps = torch.autograd.grad(f_eval, (x,) + tuple(backward_list), grad_x)
                # print(vjps[1:])
                grad_params.append(vjps[1:])
                grad_x = grad_x + vjps[0]

        flat_params_vjp = []

        for param in grad_params[::-1]:
            flat_params_vjp += param[n_fun_grad:]
        flat_param_fun = grad_params[::-1][0][:n_fun_grad]
        for param in grad_params[::-1][1:]:
            for j in range(n_fun_grad):
                flat_param_fun[j] = flat_param_fun[j] + param[j]
        flat_params_vjp = list(flat_param_fun) + flat_params_vjp
        flat_params = []
        i = 0
        for requires_grad in params_require_grad:
            if requires_grad:
                flat_params.append(flat_params_vjp[i])
                i += 1
            else:
                flat_params.append(None)
        return (grad_x, None, None, *flat_params)


class ResNetNoBackprop(nn.Module):
    """
    A class used to define a ResNet with the memory tricks
    Parameters
    ----------
    functions : list of modules, list of Sequential or Sequential
        a list of Sequential to define the transformation at each layer

    Methods
    -------
    forward(x)
        maps x to the output of the network
    """

    def __init__(self, functions, use_heun=False):
        super(ResNetNoBackprop, self).__init__()

        for i, function in enumerate(functions):
            self.add_module(str(i), function)
        self.n_functions = len(functions)
        self.use_heun = use_heun

    def forward(self, x, *function_args):
        functions = self.functions
        params = tuple(find_parameters(self))
        n_fun_args = len(function_args)
        forward_model = ResNetTransformMemory
        output = forward_model.apply(
            x,
            functions,
            n_fun_args,
            *function_args,
            *params,
        )
        return output

    @property
    def functions(self):
        return [self._modules[str(i)] for i in range(self.n_functions)]

    @property
    def init_function(self):
        return self._modules["init"]


class TinyResnet(nn.Module):
    def __init__(
        self,
        n_layers,
        in_planes=64,
        num_classes=10,
        zero_init=True,
        use_heun=False,
        use_relu=True,
        use_bn=True,
        use_backprop=False,
    ):
        super(TinyResnet, self).__init__()
        self.in_planes = in_planes

        self.conv1 = nn.Conv2d(
            3, in_planes, kernel_size=5, stride=3, padding=1, bias=False
        )

        self.bn1 = nn.BatchNorm2d(in_planes)
        functions = [
            TinyResidual(
                in_planes,
                n_layers=n_layers,
                zero_conv_init=zero_init,
                use_relu=use_relu,
                use_bn=use_bn,
            )
            for _ in range(n_layers)
        ]
        if not use_backprop:
            self.residual_layers = ResNetNoBackprop(functions, use_heun=True)
        else:
            self.residual_layers = nn.Sequential(*functions)
        self.linear = nn.Linear(4 * in_planes, num_classes)
        self.n_layers = n_layers
        self.zero_init = zero_init
        self.use_heun = use_heun
        self.use_relu = use_relu
        self.use_backprop = use_backprop

    def forward(self, x, early_exit=False):
        out = F.relu(self.bn1(self.conv1(x)))
        if early_exit:
            in_ = out.clone()
        if self.use_heun:
            if self.use_backprop:
                for i in range(self.n_layers - 1):
                    fi = self.residual_layers[i]
                    fii = self.residual_layers[i + 1]
                    fx = fi(out)
                    y = out + fx / self.n_layers
                    out = out + (fii(y) + fx) / 2 / self.n_layers
            else:
                out = self.residual_layers(out)
        else:
            for function in self.residual_layers:
                out = out + function(out) / self.n_layers
        if early_exit:
            return in_, out
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def backward(self, x):
        if self.use_heun:
            for i in range(self.n_layers - 1):
                fi = self.residual_layers[self.n_layers - i - 1]
                fii = self.residual_layers[self.n_layers - i - 2]
                fx = fi(x)
                y = x - fx / self.n_layers
                x = x - (fii(y) + fx) / 2 / self.n_layers
        else:
            for i in range(self.n_layers):
                fx = self.residual_layers[self.n_layers - i - 1](x)
                x = x - fx / self.n_layers
        return x


class iTinyResnet(nn.Module):
    def __init__(
        self,
        n_layers,
        in_planes=64,
        num_classes=10,
        zero_init=True,
        use_heun=False,
        use_relu=True,
        use_bn=True,
        use_backprop=False,
    ):
        super(iTinyResnet, self).__init__()
        self.in_planes = in_planes

        self.conv1 = nn.Conv2d(
            3, in_planes, kernel_size=5, stride=3, padding=1, bias=False
        )

        self.bn1 = nn.BatchNorm2d(in_planes)
        functions = [
            TinyResidual(
                in_planes,
                n_layers=n_layers,
                zero_conv_init=zero_init,
                use_relu=use_relu,
                use_bn=use_bn,
            )
            for _ in range(n_layers)
        ]
        self.residual_layers = InvertibleNet(
            functions, use_backprop=use_backprop, use_heun=use_heun
        )
        self.linear = nn.Linear(4 * in_planes, num_classes)
        self.n_layers = n_layers
        self.zero_init = zero_init
        self.use_heun = use_heun
        self.use_relu = use_relu
        self.use_backprop = use_backprop

    def forward(self, x, early_exit=False):
        out = F.relu(self.bn1(self.conv1(x)))
        if early_exit:
            in_ = out.clone()
        out = self.residual_layers(out)
        if early_exit:
            return in_, out
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def backward(self, x):
        return self.residual_layers.inverse(x)


class mTinyResnet(nn.Module):
    def __init__(
        self,
        n_layers,
        in_planes=64,
        num_classes=10,
        zero_init=True,
        use_heun=False,
        use_relu=True,
        use_bn=True,
        use_backprop=False,
    ):
        super(mTinyResnet, self).__init__()
        self.in_planes = in_planes

        self.conv1 = nn.Conv2d(
            3, in_planes, kernel_size=5, stride=3, padding=1, bias=False
        )

        self.bn1 = nn.BatchNorm2d(in_planes)
        functions = [
            TinyResidual(
                in_planes,
                n_layers=n_layers,
                zero_conv_init=zero_init,
                use_relu=use_relu,
                use_bn=use_bn,
            )
            for _ in range(n_layers)
        ]
        self.residual_layers = ResNetNoBackprop(functions, use_heun=use_heun)
        self.linear = nn.Linear(4 * in_planes, num_classes)
        self.n_layers = n_layers
        self.zero_init = zero_init
        self.use_heun = use_heun
        self.use_relu = use_relu
        self.use_backprop = use_backprop

    def forward(self, x, early_exit=False):
        out = F.relu(self.bn1(self.conv1(x)))
        if early_exit:
            in_ = out.clone()
        out = self.residual_layers(out)
        if early_exit:
            return in_, out
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def backward(self, x):
        return self.residual_layers.inverse(x)


def ResNet18(zero_conv_init=False, use_batch_norm=True):
    return ResNet(
        BasicBlock,
        [2, 2, 2, 2],
        zero_conv_init=zero_conv_init,
        use_batch_norm=use_batch_norm,
    )


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())


# test()


def find_parameters(module):

    assert isinstance(module, nn.Module)

    # If called within DataParallel, parameters won't appear in module.parameters().
    if getattr(module, "_is_replica", False):

        def find_tensor_attributes(module):
            tuples = [
                (k, v)
                for k, v in module.__dict__.items()
                if torch.is_tensor(v) and v.requires_grad
            ]
            return tuples

        gen = module._named_members(get_members_fn=find_tensor_attributes)
        return [param for _, param in gen]
    else:
        return list(module.parameters())


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(0)
    net = iTinyResnet(
        100, 16, use_relu=True, zero_init=False, use_heun=True, use_backprop=False
    )
    x = torch.randn(10, 3, 32, 32)
    in_, out_ = net(x, early_exit=True)
    loss = (out_ ** 2).sum()
    loss.backward()
    print(net.residual_layers[1].conv2.weight.grad[0, 0])
