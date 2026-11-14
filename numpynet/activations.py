import numpy as np

from .layers import Layer


def softmax(z, axis=-1):
    e = np.exp(z - z.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


class ReLU(Layer):
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad):
        return grad * self.mask

    def __repr__(self):
        return "ReLU()"


class LeakyReLU(Layer):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x):
        self.mask = x > 0
        return np.where(self.mask, x, self.alpha * x)

    def backward(self, grad):
        return np.where(self.mask, grad, self.alpha * grad)

    def __repr__(self):
        return f"LeakyReLU(alpha={self.alpha})"


class Sigmoid(Layer):
    def forward(self, x):
        # sigmoid(x) = (1 + tanh(x/2)) / 2, which never overflows unlike 1/(1+exp(-x))
        self.out = 0.5 * (1.0 + np.tanh(0.5 * x))
        return self.out

    def backward(self, grad):
        return grad * self.out * (1.0 - self.out)

    def __repr__(self):
        return "Sigmoid()"


class Tanh(Layer):
    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, grad):
        return grad * (1.0 - self.out ** 2)

    def __repr__(self):
        return "Tanh()"
