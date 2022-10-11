"""Operator implementations."""

from numbers import Number
from typing import Optional, List

import numpy as np

from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp

import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks
import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a ** self.scalar

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a: NDArray, b: NDArray):
        return a / b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        adj_a = out_grad / rhs
        adj_b = -1 * out_grad * lhs / (rhs ** 2)
        return adj_a, adj_b


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad / self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        order = list(range(a.ndim))
        if self.axes is None:
            order[-2], order[-1] = order[-1], order[-2]
        else:
            order[self.axes[0]], order[self.axes[1]] = order[self.axes[1]], order[self.axes[0]]

        return np.transpose(a, order)

    def gradient(self, out_grad, node):
        return transpose(out_grad, self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return a.reshape(self.shape)

    def gradient(self, out_grad: Tensor, node: Tensor):
        inp = node.inputs[0]
        return reshape(out_grad, inp.shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad: Tensor, node: Tensor):
        inp = node.inputs[0]
        shape_v2 = list(reversed(out_grad.shape))
        shape_v1 = list(reversed(inp.shape))

        squeeze_idxs = []
        i = 0

        while i < len(shape_v2):
            if i < len(shape_v1) and shape_v1[i] != shape_v2[i]:
                assert shape_v1[i] == 1
                squeeze_idxs.append(len(shape_v2)-1-i)
            elif i >= len(shape_v1):
                squeeze_idxs.append(len(shape_v2)-1-i)
            i += 1

        grad = reshape(summation(out_grad, tuple(squeeze_idxs)), inp.shape)

        return grad


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.sum(a, axis=self.axes)

    def gradient(self, out_grad: Tensor, node):
        inp = node.inputs[0]
        shape = list(inp.shape)
        if self.axes is None:
            axes = range(len(shape))
        elif isinstance(self.axes, int):
            axes = [self.axes]
        else:
            axes = self.axes
        for i in axes:
            shape[i] = 1
        res = broadcast_to(reshape(out_grad, shape), inp.shape)
        return res


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return array_api.matmul(a, b)

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        lhs_adj = matmul(out_grad, transpose(rhs))
        rhs_adj = matmul(transpose(lhs), out_grad)

        start_idx = abs(len(lhs_adj.shape) - len(lhs.shape))
        indices = list(range(len(lhs_adj.shape)))[:start_idx]
        lhs_adj = summation(lhs_adj, axes=tuple(indices))

        start_idx = abs(len(rhs_adj.shape) - len(rhs.shape))
        indices = list(range(len(rhs_adj.shape)))[:start_idx]
        rhs_adj = summation(rhs_adj, axes=tuple(indices))

        return lhs_adj, rhs_adj


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -a

    def gradient(self, out_grad, node):
        return negate(out_grad)


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        return multiply(divide(broadcast_to(Tensor([1.0,]), node.inputs[0].shape), node.inputs[0]), out_grad)


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return np.exp(a)

    def gradient(self, out_grad, node):
        return multiply(out_grad, exp(node.inputs[0]))


def exp(a):
    return Exp()(a)


# TODO
class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(a, np.array(0))

    def gradient(self, out_grad, node):
        inp = node.inputs[0].realize_cached_data()
        inp_derivative = Tensor(np.where(inp <= 0, 0.0, 1.0))
        return multiply(out_grad, inp_derivative)


def relu(a):
    return ReLU()(a)
