import time

import numpy as np


def iterate_minibatches(X, y, batch_size, shuffle=True, rng=None):
    n = len(X)
    if shuffle:
        rng = np.random.default_rng() if rng is None else rng
        idx = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch = idx[start:start + batch_size]
            yield X[batch], y[batch]
    else:
        for start in range(0, n, batch_size):
            yield X[start:start + batch_size], y[start:start + batch_size]


def accuracy(logits, labels):
    labels = np.asarray(labels)
    if labels.ndim == 2:  # allow one-hot as well
        labels = labels.argmax(axis=1)
    return float(np.mean(logits.argmax(axis=1) == labels))


def evaluate(model, loss_fn, X, y, batch_size=512):
    was_training = model.training
    model.eval()
    total_loss = total_correct = 0.0
    try:
        for xb, yb in iterate_minibatches(X, y, batch_size, shuffle=False):
            out = model.forward(xb)
            # each batch loss is a mean, so weight by size to get the true mean
            total_loss += loss_fn.forward(out, yb) * len(xb)
            total_correct += accuracy(out, yb) * len(xb)
    finally:
        if was_training:
            model.train()
    return total_loss / len(X), total_correct / len(X)


def fit(model, loss_fn, optimizer, X_train, y_train, epochs=10, batch_size=64,
        val_data=None, shuffle=True, rng=None, verbose=True, on_epoch_end=None):
    rng = np.random.default_rng() if rng is None else rng
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        model.train()
        loss_sum = correct = 0.0
        seen = 0

        for xb, yb in iterate_minibatches(X_train, y_train, batch_size, shuffle, rng):
            out = model.forward(xb)
            loss = loss_fn.forward(out, yb)
            optimizer.zero_grad()
            model.backward(loss_fn.backward())
            optimizer.step()

            loss_sum += loss * len(xb)
            correct += accuracy(out, yb) * len(xb)
            seen += len(xb)

        history["train_loss"].append(loss_sum / seen)
        history["train_acc"].append(correct / seen)
        line = (f"epoch {epoch:>{len(str(epochs))}}/{epochs}  "
                f"loss {history['train_loss'][-1]:.4f}  acc {history['train_acc'][-1]:.4f}")

        if val_data is not None:
            val_loss, val_acc = evaluate(model, loss_fn, *val_data, batch_size=max(batch_size, 512))
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            line += f"  val_loss {val_loss:.4f}  val_acc {val_acc:.4f}"

        if verbose:
            print(f"{line}  ({time.perf_counter() - start:.1f}s)")
        if on_epoch_end is not None:
            on_epoch_end(epoch, history)

    return history
