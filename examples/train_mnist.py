"""Train an MLP on MNIST and save the weights, history and training curves.

    python examples/train_mnist.py
    python examples/train_mnist.py --epochs 30 --hidden 512 256 --export
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from numpynet import Adam, Dense, Dropout, ReLU, Sequential, SoftmaxCrossEntropy  # noqa: E402
from numpynet import accuracy, evaluate, iterate_minibatches  # noqa: E402
from numpynet.data import load_mnist, random_shift  # noqa: E402


def build_model(hidden, dropout, rng):
    sizes = [784, *hidden]
    layers = []
    for n_in, n_out in zip(sizes, sizes[1:]):
        layers += [Dense(n_in, n_out, rng=rng), ReLU()]
        if dropout > 0:
            layers.append(Dropout(dropout, rng=rng))
    layers.append(Dense(sizes[-1], 10, init="xavier", rng=rng))
    return Sequential(layers).astype(np.float32)


def export_for_browser(model, test_acc, out_dir):
    """Writes the dense weights as raw float32 plus a small json description,
    which is what the demo page in docs/ reads."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dense = [layer for layer in model.layers if isinstance(layer, Dense)]
    meta = {"layers": [], "test_accuracy": round(test_acc, 4), "weights": "weights.bin"}
    with open(out_dir / "weights.bin", "wb") as f:
        for i, layer in enumerate(dense):
            f.write(layer.W.data.astype("<f4").tobytes())
            f.write(layer.b.data.astype("<f4").tobytes())
            meta["layers"].append({
                "in": layer.in_features,
                "out": layer.out_features,
                "activation": "relu" if i < len(dense) - 1 else "none",
            })
    (out_dir / "model.json").write_text(json.dumps(meta, indent=2) + "\n")


def plot_history(history, test_acc, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="validation")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("cross-entropy loss")
    ax1.set_title("Loss")
    ax1.legend()

    ax2.plot(epochs, [100 * a for a in history["train_acc"]], label="train")
    ax2.plot(epochs, [100 * a for a in history["val_acc"]], label="validation")
    ax2.axhline(100 * test_acc, color="gray", linestyle="--", linewidth=1,
                label=f"test ({100 * test_acc:.2f}%)")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy (%)")
    ax2.set_title("Accuracy")
    ax2.legend(loc="lower right")

    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-decay", type=float, default=0.85,
                        help="multiply the learning rate by this after every epoch")
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--no-augment", action="store_true",
                        help="skip the random 2px shifts applied each epoch")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--export", action="store_true",
                        help="also write the weights for the browser demo into docs/")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    X_train, y_train, X_test, y_test = load_mnist(ROOT / "data")

    # hold out the last 5000 training images for validation
    X_val, y_val = X_train[-5000:], y_train[-5000:]
    X_train, y_train = X_train[:-5000], y_train[:-5000]

    model = build_model(args.hidden, args.dropout, rng)
    loss_fn = SoftmaxCrossEntropy()
    optimizer = Adam(model.params(), lr=args.lr)
    print(model)
    print(f"{model.num_params():,} parameters\n")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    start = time.perf_counter()

    # the loop is written out here instead of calling fit() because the
    # training images are re-augmented at the start of every epoch
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        X_epoch = X_train if args.no_augment else random_shift(X_train, 2, rng)

        model.train()
        loss_sum = correct = 0.0
        for xb, yb in iterate_minibatches(X_epoch, y_train, args.batch_size, rng=rng):
            out = model.forward(xb)
            loss = loss_fn.forward(out, yb)
            optimizer.zero_grad()
            model.backward(loss_fn.backward())
            optimizer.step()
            loss_sum += loss * len(xb)
            correct += accuracy(out, yb) * len(xb)

        val_loss, val_acc = evaluate(model, loss_fn, X_val, y_val)
        history["train_loss"].append(loss_sum / len(X_train))
        history["train_acc"].append(correct / len(X_train))
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"epoch {epoch:>2}/{args.epochs}  loss {history['train_loss'][-1]:.4f}  "
              f"acc {history['train_acc'][-1]:.4f}  val_loss {val_loss:.4f}  "
              f"val_acc {val_acc:.4f}  lr {optimizer.lr:.2e}  ({time.perf_counter() - t0:.1f}s)")
        optimizer.lr *= args.lr_decay

    elapsed = time.perf_counter() - start
    test_loss, test_acc = evaluate(model, loss_fn, X_test, y_test)
    print(f"\ntest loss {test_loss:.4f}  test accuracy {100 * test_acc:.2f}%  "
          f"(trained in {elapsed:.0f}s)")

    runs = ROOT / "runs"
    runs.mkdir(exist_ok=True)
    model.save(runs / "mnist_mlp")
    history.update(test_loss=test_loss, test_acc=test_acc, seconds=round(elapsed, 1),
                   args=vars(args))
    (runs / "history.json").write_text(json.dumps(history, indent=2))
    plot_history(history, test_acc, ROOT / "assets" / "training_curves.png")
    print("saved runs/mnist_mlp.npz, runs/history.json and assets/training_curves.png")

    if args.export:
        export_for_browser(model, test_acc, ROOT / "docs")
        print("exported docs/model.json and docs/weights.bin")


if __name__ == "__main__":
    main()
