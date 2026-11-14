import numpy as np


def numerical_gradient(f, x, eps=1e-5):
    grad = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        old = x[idx]
        x[idx] = old + eps
        f_plus = f()
        x[idx] = old - eps
        f_minus = f()
        x[idx] = old
        grad[idx] = (f_plus - f_minus) / (2 * eps)
    return grad


def relative_error(a, b):
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(a) + np.abs(b), 1e-8)))


def check_layer(layer, x, eps=1e-5, seed=0):
    x = np.array(x, dtype=np.float64)
    out = layer.forward(x)
    # project the output onto a fixed random direction to get a scalar "loss";
    # its gradient w.r.t. the output is then just R
    R = np.random.default_rng(seed).standard_normal(out.shape)

    def f():
        return float(np.sum(layer.forward(x) * R))

    layer.forward(x)
    dx = layer.backward(R)
    analytic = {p.name: p.grad.copy() for p in layer.params()}

    errors = {"input": relative_error(dx, numerical_gradient(f, x, eps))}
    for p in layer.params():
        errors[p.name] = relative_error(analytic[p.name], numerical_gradient(f, p.data, eps))
    return errors


def check_model(model, loss_fn, x, y, eps=1e-5):
    def f():
        return loss_fn.forward(model.forward(x), y)

    f()
    model.backward(loss_fn.backward())
    named = model.named_params()
    analytic = {key: p.grad.copy() for key, p in named}

    return {key: relative_error(analytic[key], numerical_gradient(f, p.data, eps))
            for key, p in named}
