import numpy as np


class Parameter:
    def __init__(self, data, name=""):
        self.data = data
        self.grad = np.zeros_like(data)
        self.name = name

    def __repr__(self):
        return f"Parameter({self.name}, shape={self.data.shape}, dtype={self.data.dtype})"


class Layer:
    training = True
    # names of non-trainable arrays (e.g. running stats) that should be saved and cast
    buffers = ()

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def params(self):
        return []

    def __call__(self, x):
        return self.forward(x)


class Dense(Layer):
    def __init__(self, in_features, out_features, init="he", rng=None):
        rng = np.random.default_rng() if rng is None else rng
        if init == "he":
            std = np.sqrt(2.0 / in_features)
        elif init == "xavier":
            std = np.sqrt(2.0 / (in_features + out_features))
        elif init == "normal":
            std = 0.01
        else:
            raise ValueError(f"unknown init '{init}', expected 'he', 'xavier' or 'normal'")

        self.in_features = in_features
        self.out_features = out_features
        self.W = Parameter(rng.standard_normal((in_features, out_features)) * std, "W")
        self.b = Parameter(np.zeros(out_features), "b")

    def forward(self, x):
        self.x = x
        return x @ self.W.data + self.b.data

    def backward(self, grad):
        self.W.grad[...] = self.x.T @ grad
        self.b.grad[...] = grad.sum(axis=0)
        return grad @ self.W.data.T

    def params(self):
        return [self.W, self.b]

    def __repr__(self):
        return f"Dense({self.in_features}, {self.out_features})"


class Dropout(Layer):
    def __init__(self, p=0.5, rng=None):
        if not 0.0 <= p < 1.0:
            raise ValueError("dropout probability must be in [0, 1)")
        self.p = p
        self.rng = np.random.default_rng() if rng is None else rng
        self.mask = None

    def forward(self, x):
        if not self.training or self.p == 0.0:
            self.mask = None
            return x
        keep = 1.0 - self.p
        # inverted dropout: scale at train time so eval is a plain identity
        self.mask = (self.rng.random(x.shape) < keep).astype(x.dtype) / keep
        return x * self.mask

    def backward(self, grad):
        if self.mask is None:
            return grad
        return grad * self.mask

    def __repr__(self):
        return f"Dropout(p={self.p})"


class BatchNorm(Layer):
    buffers = ("running_mean", "running_var")

    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.gamma = Parameter(np.ones(num_features), "gamma")
        self.beta = Parameter(np.zeros(num_features), "beta")
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, x):
        if self.training:
            mean = x.mean(axis=0)
            var = x.var(axis=0)
            m = self.momentum
            self.running_mean = m * self.running_mean + (1 - m) * mean
            self.running_var = m * self.running_var + (1 - m) * var
        else:
            mean, var = self.running_mean, self.running_var

        self.std = np.sqrt(var + self.eps)
        self.x_hat = (x - mean) / self.std
        return self.gamma.data * self.x_hat + self.beta.data

    def backward(self, grad):
        self.gamma.grad[...] = (grad * self.x_hat).sum(axis=0)
        self.beta.grad[...] = grad.sum(axis=0)
        dx_hat = grad * self.gamma.data
        if not self.training:
            # running stats are constants, so this is just an affine map
            return dx_hat / self.std
        # mean and var depend on every sample in the batch; after pushing the
        # gradient through both, the terms collapse to this compact form
        return (dx_hat - dx_hat.mean(axis=0)
                - self.x_hat * (dx_hat * self.x_hat).mean(axis=0)) / self.std

    def params(self):
        return [self.gamma, self.beta]

    def __repr__(self):
        return f"BatchNorm({self.num_features})"


class Flatten(Layer):
    def forward(self, x):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad):
        return grad.reshape(self.shape)

    def __repr__(self):
        return "Flatten()"
