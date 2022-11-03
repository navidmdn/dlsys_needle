"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will numpy as the array_api
# to backup our computations, this line will change in later homeworks
import numpy as array_api


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple(*[out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(array_api.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


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
        return self.scalar * power_scalar(node.inputs[0], self.scalar-1) * out_grad


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
        return array_api.array(a / self.scalar, dtype=a.dtype)

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

        return array_api.transpose(a, order)

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
    def __init__(self, axes: Optional[tuple] = None, keepdims=False):
        self.axes = axes
        self.keepdims = keepdims

    def compute(self, a):
        return array_api.sum(a, axis=self.axes, keepdims=self.keepdims)

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


def summation(a, axes=None, keepdims=False):
    return Summation(axes, keepdims)(a)


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
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        return multiply(out_grad, exp(node.inputs[0]))


def exp(a):
    return Exp()(a)


# TODO
class ReLU(TensorOp):
    def compute(self, a):
        return array_api.maximum(a, array_api.array(0))

    def gradient(self, out_grad, node):

        inp = node.inputs[0].realize_cached_data()
        inp_derivative = Tensor(array_api.where(inp <= 0, 0.0, 1.0))
        return multiply(out_grad, inp_derivative)


def relu(a):
    return ReLU()(a)


class Max(TensorOp):
    def __init__(self, axes: Optional[tuple] = None, keepdims=False):
        self.axes = axes
        self.keepdims = keepdims

    def compute(self, Z):
        return array_api.max(Z, axis=self.axes, keepdims=self.keepdims)

    def gradient(self, out_grad, node):
        x = node.inputs[0]
        zeros = array_api.zeros(x.shape)
        zeros[array_api.argmax(x.realize_cached_data())] = 1.0
        max_t = Tensor(zeros)
        return broadcast_to(out_grad, x.shape) * max_t


def maximum(a, axes=None, keepdims=False):
    return Max(axes, keepdims)(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        out_shape = array_api.sum(Z, axis=self.axes).shape
        _logsumexp = array_api.log(array_api.sum(array_api.exp(Z - array_api.max(Z, axis=self.axes, keepdims=True)),
                                                 axis=self.axes, keepdims=True))
        _logsumexp += array_api.max(Z, axis=self.axes, keepdims=True)
        return _logsumexp.reshape(out_shape)

    def gradient(self, out_grad, node):
        x = node.inputs[0]
        _max = maximum(x, axes=self.axes, keepdims=True)
        x = x - broadcast_to(_max, x.shape)
        nomtr = exp(x)
        _sum = summation(exp(x), axes=self.axes, keepdims=True)
        denomtr = broadcast_to(_sum, nomtr.shape)
        return (nomtr/(denomtr + 1e-9))*broadcast_to(reshape(out_grad, _sum.shape), denomtr.shape)




def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
