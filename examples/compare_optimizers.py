"""Train the same network with SGD, SGD + momentum and Adam and plot the loss.

    python examples/compare_optimizers.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from numpynet import SGD, Adam, Dense, ReLU, Sequential, SoftmaxCrossEntropy, fit  # noqa: E402
from numpynet.data import load_mnist  # noqa: E402

EPOCHS = 8

OPTIMIZERS = {
    "SGD (lr 0.05)": lambda params: SGD(params, lr=0.05),
    "SGD + momentum 0.9 (lr 0.05)": lambda params: SGD(params, lr=0.05, momentum=0.9),
    "Adam (lr 0.001)": lambda params: Adam(params, lr=1e-3),
}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X_train, y_train, X_test, y_test = load_mnist(ROOT / "data")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    for name, make_optimizer in OPTIMIZERS.items():
        # same seed for every run so they all start from identical weights
        rng = np.random.default_rng(0)
        model = Sequential([
            Dense(784, 128, rng=rng), ReLU(),
            Dense(128, 64, rng=rng), ReLU(),
            Dense(64, 10, init="xavier", rng=rng),
        ]).astype(np.float32)

        print(f"\n{name}")
        history = fit(model, SoftmaxCrossEntropy(), make_optimizer(model.params()),
                      X_train, y_train, epochs=EPOCHS, batch_size=64,
                      val_data=(X_test, y_test), rng=rng)

        epochs = range(1, EPOCHS + 1)
        ax1.plot(epochs, history["train_loss"], marker="o", markersize=3, label=name)
        ax2.plot(epochs, [100 * a for a in history["val_acc"]], marker="o", markersize=3, label=name)

    ax1.set_title("Training loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("cross-entropy loss")
    ax1.set_yscale("log")
    ax2.set_title("Test accuracy")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy (%)")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()

    out = ROOT / "assets" / "optimizer_comparison.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\nsaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
