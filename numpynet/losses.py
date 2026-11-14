import numpy as np


class Loss:
    def forward(self, pred, target):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def __call__(self, pred, target):
        return self.forward(pred, target)


class MSE(Loss):
    """Mean squared error, averaged over every element of the batch."""

    def forward(self, pred, target):
        target = np.asarray(target, dtype=pred.dtype)
        if target.shape != pred.shape and target.size == pred.size:
            target = target.reshape(pred.shape)  # e.g. (N,) targets vs (N, 1) outputs
        self.diff = pred - target
        return float(np.mean(self.diff ** 2))

    def backward(self):
        return 2.0 * self.diff / self.diff.size


class SoftmaxCrossEntropy(Loss):
    """Softmax over raw logits followed by cross-entropy.

    Targets can be integer labels of shape (N,) or one-hot rows of shape (N, C).
    """

    def forward(self, logits, target):
        n = logits.shape[0]
        shifted = logits - logits.max(axis=1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))

        target = np.asarray(target)
        if target.ndim == 1:
            onehot = np.zeros_like(logits)
            onehot[np.arange(n), target] = 1
        else:
            onehot = target.astype(logits.dtype, copy=False)

        self.probs = np.exp(log_probs)
        self.onehot = onehot
        return float(-(onehot * log_probs).sum() / n)

    def backward(self):
        # d/dz of -sum(y * log softmax(z)) is softmax(z) - y when the rows of y sum
        # to one, which is why the two are fused instead of chained separately
        return (self.probs - self.onehot) / self.probs.shape[0]
