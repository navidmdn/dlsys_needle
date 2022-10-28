"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype)
        self.has_bias = bias

        if self.has_bias:
            self.bias = ops.reshape(init.kaiming_uniform(out_features, 1,  device=device, dtype=dtype), (1, out_features))

    def forward(self, X: Tensor) -> Tensor:
        out = ops.matmul(X, self.weight)
        if self.has_bias:
            bias = ops.broadcast_to(self.bias, out.shape)
            out += bias

        return out


class Flatten(Module):
    def forward(self, X: Tensor):
        return X.reshape((X.shape[0], -1))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)
        return x


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        nclass = logits.shape[-1]
        zy = ops.summation(init.one_hot(nclass, y) * logits, axes=(1,), keepdims=True)
        loss = ops.logsumexp(logits - ops.broadcast_to(zy, logits.shape), axes=(1,))
        loss = ops.summation(loss) / loss.shape[0]
        return loss



class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = init.ones(dim, device=device, dtype=dtype, requires_grad=True)
        self.bias = init.zeros(dim, device=device, dtype=dtype, requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        mean = ops.summation(x, axes=(1,), keepdims=True) / x.shape[1] #bsx1
        var = ops.summation((x - ops.broadcast_to(mean, x.shape))**2, axes=(1,), keepdims=True) / x.shape[1]
        norm = (x - ops.broadcast_to(mean, x.shape)) / ((ops.broadcast_to(var, x.shape) + self.eps) ** 0.5)
        y = ops.broadcast_to(self.weight, x.shape) * norm + ops.broadcast_to(self.bias, x.shape)

        return y


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION



