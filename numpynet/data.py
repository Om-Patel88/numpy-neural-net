import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np

MNIST_MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]

MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _download(filename, dest):
    last_error = None
    for mirror in MNIST_MIRRORS:
        try:
            print(f"downloading {filename} ...")
            tmp = dest.with_suffix(dest.suffix + ".part")
            urllib.request.urlretrieve(mirror + filename, tmp)
            tmp.replace(dest)
            return
        except OSError as err:
            last_error = err
    raise RuntimeError(f"could not download {filename}: {last_error}")


def _read_idx(path):
    # idx format: 2 zero bytes, a dtype byte, the number of dimensions,
    # then one big-endian uint32 per dimension, then the raw data
    with gzip.open(path, "rb") as f:
        _, dtype_code, ndim = struct.unpack(">HBB", f.read(4))
        if dtype_code != 0x08:
            raise ValueError(f"{path} is not a uint8 idx file")
        shape = struct.unpack(f">{ndim}I", f.read(4 * ndim))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def load_mnist(cache_dir="data", flatten=True, dtype=np.float32):
    """Returns (X_train, y_train, X_test, y_test) with pixels scaled to [0, 1].

    The four gzip files are downloaded into cache_dir the first time.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    arrays = {}
    for key, filename in MNIST_FILES.items():
        path = cache_dir / filename
        if not path.exists():
            _download(filename, path)
        arrays[key] = _read_idx(path)

    def images(a):
        a = a.astype(dtype) / 255.0
        return a.reshape(len(a), -1) if flatten else a

    return (images(arrays["train_images"]), arrays["train_labels"].astype(np.int64),
            images(arrays["test_images"]), arrays["test_labels"].astype(np.int64))


def random_shift(X, max_shift=2, rng=None):
    """Shifts each flattened 28x28 image by up to max_shift pixels in x and y."""
    rng = np.random.default_rng() if rng is None else rng
    n = len(X)
    side = 28
    padded = np.pad(X.reshape(n, side, side), ((0, 0), (max_shift, max_shift), (max_shift, max_shift)))
    dy = rng.integers(0, 2 * max_shift + 1, size=n)
    dx = rng.integers(0, 2 * max_shift + 1, size=n)
    rows = dy[:, None] + np.arange(side)
    cols = dx[:, None] + np.arange(side)
    shifted = padded[np.arange(n)[:, None, None], rows[:, :, None], cols[:, None, :]]
    return shifted.reshape(n, -1)
