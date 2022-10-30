"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        for param in self.params:
            if param.grad is None:
                continue
            if param not in self.u:
                self.u[param] = 0.0
            self.u[param] = self.momentum * self.u[param] + (1 - self.momentum) * \
                            (param.grad.data + self.weight_decay * param.data)
            param.data = ndl.Tensor(param.data - self.lr * self.u[param], dtype='float32')


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
        bias_correction=True,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.bias_correction = bias_correction
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1
        for param in self.params:
            if param.grad is None:
                continue
            if param not in self.m:
                self.m[param] = 0.0
                self.v[param] = 0.0

            self.m[param] = self.beta1 * self.m[param] + (1 - self.beta1) * (param.grad.data.numpy() + self.weight_decay * param.data.numpy())
            self.v[param] = self.beta2 * self.v[param] + (1 - self.beta2) * (param.grad.data.numpy() + self.weight_decay * param.data.numpy())**2

            if self.bias_correction:
                m_hat = self.m[param] / (1 - self.beta1 ** self.t)
                v_hat = self.v[param] / (1 - self.beta2 ** self.t)
            else:
                m_hat = self.m[param]
                v_hat = self.v[param]

            param.data = ndl.Tensor(param.data.numpy() - ((self.lr * m_hat) / (v_hat ** 0.5 + self.eps)),
                                    dtype='float32')

