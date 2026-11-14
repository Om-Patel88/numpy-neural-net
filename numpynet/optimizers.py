import numpy as np


class Optimizer:
    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.grad.fill(0)

    def step(self):
        raise NotImplementedError

    def _grads(self, weight_decay):
        for p in self.params:
            g = p.grad
            if weight_decay:
                g = g + weight_decay * p.data
            yield p, g


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, nesterov=False, weight_decay=0.0):
        super().__init__(params, lr)
        if nesterov and momentum == 0:
            raise ValueError("nesterov needs a nonzero momentum")
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.velocity = None

    def step(self):
        if self.momentum and self.velocity is None:
            self.velocity = [np.zeros_like(p.data) for p in self.params]

        for i, (p, g) in enumerate(self._grads(self.weight_decay)):
            if self.momentum:
                v = self.velocity[i]
                v *= self.momentum
                v += g
                # nesterov looks ahead along the updated velocity
                g = g + self.momentum * v if self.nesterov else v
            p.data -= self.lr * g


class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = None
        self.v = None

    def step(self):
        if self.m is None:
            self.m = [np.zeros_like(p.data) for p in self.params]
            self.v = [np.zeros_like(p.data) for p in self.params]

        self.t += 1
        b1, b2 = self.beta1, self.beta2
        # m and v start at zero, so early on they are biased toward zero;
        # dividing by (1 - beta^t) undoes that
        corr1 = 1.0 - b1 ** self.t
        corr2 = 1.0 - b2 ** self.t

        for i, (p, g) in enumerate(self._grads(self.weight_decay)):
            m, v = self.m[i], self.v[i]
            m *= b1
            m += (1.0 - b1) * g
            v *= b2
            v += (1.0 - b2) * g * g
            p.data -= self.lr * (m / corr1) / (np.sqrt(v / corr2) + self.eps)
