import torch
import torch.nn as nn


class ResNetWithBackprop(nn.Module):
    """
    A class used to define a Momentum ResNet
    Parameters
    ----------
    functions : list of modules, list of Sequential or Sequential
        a list of Sequential to define the transformation at each layer
    Methods
    -------
    forward(x)
        maps x to the output of the network
    """

    def __init__(
        self,
        functions,
    ):
        super(ResNetWithBackprop, self).__init__()
        self.n_functions = len(functions)
        for i, function in enumerate(functions):
            self.add_module(str(i), function)

    def forward(self, x, *function_args):
        n_iters = len(self.functions)
        for i in range(n_iters):
            f = self.functions[i](x, *function_args) / n_iters
            x = x + f
        return x

    @property
    def functions(self):
        return [self._modules[str(i)] for i in range(self.n_functions)]

    @property
    def init_function(self):
        return self._modules["init"]


class ResNetTransformMemory(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x,
        functions,
        n_fun_args,
        *params
    ):
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
                if i == n_iters - 1 :
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
                function = functions[n_iters - 1 - i].eval()
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
                vjps = torch.autograd.grad(
                    f_eval, (x,) + tuple(backward_list), grad_x
                )
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
                flat_params.append(
                    None
                )

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

    def __init__(
        self,
        functions,
        use_heun=False
    ):
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


class ResNetBack(nn.Module):
    """
    Create a Residual Network.
    Parameters
    ----------
    functions : list of modules, list of Sequential or Sequential
        a list of Sequential to define the transformation at each layer.
        Each function in the list can take additional inputs. 'x' is assumed
        to be the first input.
    """

    def __init__(
        self,
        functions,
        use_backprop=True,
    ):

        super(ResNetBack, self).__init__()
        if use_backprop is False and len(functions) > 10:
            self.network = ResNetNoBackprop(
                functions,
            )
        else:
            self.network = ResNetWithBackprop(
                functions,
            )
        self.use_backprop = use_backprop

    def forward(self, x, *args, **kwargs):
        return self.network.forward(x, *args)

    @property
    def functions(self):
        return self.network.functions

    @property
    def init_function(self):
        return self.network.init_function


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


class ResNetTransformMemoryHeun(torch.autograd.Function):
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
            for i in range(n_iters - 1):
                f = functions[i](x, *fun_args) / n_iters
                y = x + f
                f2 = functions[i + 1](y, *fun_args) / n_iters
                x = x + (f + f2) / 2
                if i == n_iters - 2:
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
            for i in range(n_iters - 1):
                function = functions[n_iters - 1 - i]
                function2 = functions[n_iters - 2 - i]
                x = x.detach().requires_grad_(False)
                with torch.no_grad():
                    f_eval = -function(x, *fun_args) / n_iters  # beware, minus sign
                    y = x + f_eval.data
                    f_eval2 = -function2(y, *fun_args) / n_iters
                    x = x + (f_eval2.data + f_eval.data) / 2
                x = x.detach().requires_grad_(True)
                f_eval = function2(x, *fun_args) / n_iters
                y = x + f_eval
                f_eval2 = function(y, *fun_args) / n_iters
                output = (f_eval2 + f_eval) / 2
                backward_list = []
                for requires_grad, param in zip(
                    params_require_grad,
                    fun_args
                    + tuple(find_parameters(function))
                    + tuple(find_parameters(function2)),
                ):
                    if requires_grad:
                        backward_list.append(param)
                vjps = torch.autograd.grad(output, (x,) + tuple(backward_list), grad_x)
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


if __name__ == "__main__":
    torch.manual_seed(0)
    use_heun = True
    function = [nn.Linear(3, 3) for _ in range(100)]
    net = ResNetNoBackprop(function, use_heun=use_heun)
    x = torch.randn(10, 3)
    op = net(x).sum()
    op.backward()
    frac = 1 if use_heun else 2
    print(function[0].weight.grad / frac)
