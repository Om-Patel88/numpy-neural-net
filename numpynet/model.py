import numpy as np


class Sequential:
    def __init__(self, layers):
        self.layers = list(layers)
        self.training = True

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def __call__(self, x):
        return self.forward(x)

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params(self):
        return [p for layer in self.layers for p in layer.params()]

    def named_params(self):
        return [(f"layer{i}.{p.name}", p)
                for i, layer in enumerate(self.layers) for p in layer.params()]

    def num_params(self):
        return sum(p.data.size for p in self.params())

    def train(self):
        return self._set_mode(True)

    def eval(self):
        return self._set_mode(False)

    def _set_mode(self, training):
        self.training = training
        for layer in self.layers:
            layer.training = training
        return self

    def predict(self, x, batch_size=512):
        was_training = self.training
        self.eval()
        try:
            out = [self.forward(x[i:i + batch_size]) for i in range(0, len(x), batch_size)]
        finally:
            self._set_mode(was_training)
        return np.concatenate(out, axis=0)

    def astype(self, dtype):
        for p in self.params():
            p.data = p.data.astype(dtype)
            p.grad = p.grad.astype(dtype)
        for layer in self.layers:
            for name in layer.buffers:
                setattr(layer, name, getattr(layer, name).astype(dtype))
        return self

    def _state(self):
        # every array worth saving, keyed by layer position, e.g. "layer2.running_mean"
        state = {key: p.data for key, p in self.named_params()}
        for i, layer in enumerate(self.layers):
            for name in layer.buffers:
                state[f"layer{i}.{name}"] = getattr(layer, name)
        return state

    def save(self, path):
        np.savez(_npz_path(path), **self._state())

    def load(self, path):
        state = self._state()
        with np.load(_npz_path(path)) as f:
            missing = sorted(set(state) - set(f.files))
            extra = sorted(set(f.files) - set(state))
            if missing or extra:
                raise ValueError(f"checkpoint does not match model architecture "
                                 f"(missing: {missing}, unexpected: {extra})")
            loaded = {key: f[key] for key in state}

        for key, arr in loaded.items():
            if arr.shape != state[key].shape:
                raise ValueError(f"shape mismatch for '{key}': checkpoint has "
                                 f"{arr.shape}, model expects {state[key].shape}")
        # copy in place so optimizers holding these arrays stay in sync
        for key, arr in loaded.items():
            state[key][...] = arr
        return self

    def __repr__(self):
        inner = ",\n  ".join(repr(layer) for layer in self.layers)
        return f"Sequential(\n  {inner}\n)"


def _npz_path(path):
    path = str(path)
    return path if path.endswith(".npz") else path + ".npz"
