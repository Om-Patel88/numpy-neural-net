import numpy as np
import pytest

from numpynet import (
    Dense, Dropout, BatchNorm, Flatten, ReLU, Tanh,
    SoftmaxCrossEntropy, Adam, Sequential,
    iterate_minibatches, accuracy, evaluate, fit,
)


def make_model(hidden=16, seed=0):
    rng = np.random.default_rng(seed)
    return Sequential([
        Dense(4, hidden, rng=rng), BatchNorm(hidden), ReLU(), Dropout(0.2, rng=rng),
        Dense(hidden, 3, rng=rng),
    ])


def blobs(n_per_class=100, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 3.0], [3.0, -2.0], [-3.0, -2.0]])
    X = np.concatenate([c + rng.standard_normal((n_per_class, 2)) for c in centers])
    y = np.repeat(np.arange(3), n_per_class)
    return X, y


def test_params_and_count():
    model = make_model(hidden=16)
    assert len(model.params()) == 6
    assert model.num_params() == 4 * 16 + 16 + 2 * 16 + 16 * 3 + 3


def test_train_eval_modes():
    model = make_model()
    assert model.eval() is model
    assert all(not layer.training for layer in model.layers)
    assert model.train() is model
    assert all(layer.training for layer in model.layers)


def test_save_load_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    x = rng.standard_normal((32, 4))
    model = make_model(seed=0)
    for _ in range(5):
        model.forward(x * 2 + 1)  # move the BatchNorm running stats off their defaults
    path = tmp_path / "weights.npz"
    model.save(path)

    other = make_model(seed=123)
    assert not np.allclose(other.predict(x), model.predict(x))
    other.load(path)
    np.testing.assert_array_equal(other.predict(x), model.predict(x))
    np.testing.assert_array_equal(other.layers[1].running_var, model.layers[1].running_var)


def test_save_without_extension(tmp_path):
    model = make_model()
    model.save(tmp_path / "ckpt")
    make_model(seed=5).load(tmp_path / "ckpt")


def test_load_shape_mismatch(tmp_path):
    make_model(hidden=16).save(tmp_path / "w.npz")
    with pytest.raises(ValueError, match="shape mismatch"):
        make_model(hidden=8).load(tmp_path / "w.npz")


def test_load_architecture_mismatch(tmp_path):
    make_model().save(tmp_path / "w.npz")
    rng = np.random.default_rng(0)
    smaller = Sequential([Dense(4, 16, rng=rng), ReLU(), Dense(16, 3, rng=rng)])
    with pytest.raises(ValueError, match="architecture"):
        smaller.load(tmp_path / "w.npz")


def test_load_keeps_arrays_in_place(tmp_path):
    model = make_model()
    model.save(tmp_path / "w.npz")
    other = make_model(seed=9)
    W = other.layers[0].W.data
    other.load(tmp_path / "w.npz")
    assert other.layers[0].W.data is W


def test_astype_float32_end_to_end():
    model = make_model().astype(np.float32)
    for p in model.params():
        assert p.data.dtype == np.float32 and p.grad.dtype == np.float32
    assert model.layers[1].running_mean.dtype == np.float32
    assert model.layers[1].running_var.dtype == np.float32

    rng = np.random.default_rng(0)
    x = rng.standard_normal((16, 4)).astype(np.float32)
    y = rng.integers(0, 3, size=16)
    loss_fn = SoftmaxCrossEntropy()
    opt = Adam(model.params(), lr=1e-2)

    out = model.forward(x)
    assert out.dtype == np.float32
    loss_fn.forward(out, y)
    grad = loss_fn.backward()
    assert grad.dtype == np.float32
    assert model.backward(grad).dtype == np.float32
    opt.step()
    for p in model.params():
        assert p.data.dtype == np.float32 and p.grad.dtype == np.float32
    assert model.layers[1].running_mean.dtype == np.float32
    assert model.predict(x).dtype == np.float32


def test_predict_restores_mode_and_batches():
    model = make_model()
    x = np.random.default_rng(0).standard_normal((50, 4))

    model.train()
    out = model.predict(x, batch_size=7)
    assert all(layer.training for layer in model.layers)

    model.eval()
    np.testing.assert_allclose(out, model.forward(x))
    model.predict(x)
    assert all(not layer.training for layer in model.layers)


def test_flatten_in_model():
    rng = np.random.default_rng(0)
    model = Sequential([Flatten(), Dense(12, 2, rng=rng)])
    x = rng.standard_normal((5, 3, 4))
    out = model.forward(x)
    assert out.shape == (5, 2)
    assert model.backward(np.ones_like(out)).shape == x.shape


def test_iterate_minibatches_covers_everything():
    X = np.arange(23).reshape(23, 1)
    y = np.arange(23)
    batches = list(iterate_minibatches(X, y, 5, rng=np.random.default_rng(0)))
    assert [len(b[1]) for b in batches] == [5, 5, 5, 5, 3]
    seen = np.concatenate([b[1] for b in batches])
    assert sorted(seen) == list(range(23))
    assert not np.array_equal(seen, y)
    for xb, yb in batches:
        np.testing.assert_array_equal(xb[:, 0], yb)

    ordered = np.concatenate([b[1] for b in iterate_minibatches(X, y, 5, shuffle=False)])
    np.testing.assert_array_equal(ordered, y)


def test_accuracy():
    logits = np.array([[2.0, 1.0], [0.0, 3.0], [1.0, 0.0]])
    assert accuracy(logits, np.array([0, 1, 1])) == pytest.approx(2 / 3)


def test_evaluate_weights_batches_correctly():
    model = make_model()
    X, y = np.random.default_rng(0).standard_normal((53, 4)), np.arange(53) % 3
    loss_fn = SoftmaxCrossEntropy()
    model.train()
    full = evaluate(model, loss_fn, X, y, batch_size=1000)
    small = evaluate(model, loss_fn, X, y, batch_size=10)
    assert small[0] == pytest.approx(full[0])
    assert small[1] == pytest.approx(full[1])
    assert model.layers[0].training  # mode restored


def test_fit_learns_blobs():
    X, y = blobs(seed=0)
    X_val, y_val = blobs(n_per_class=50, seed=1)
    rng = np.random.default_rng(0)
    model = Sequential([Dense(2, 32, rng=rng), ReLU(), Dense(32, 32, rng=rng), Tanh(),
                        Dense(32, 3, rng=rng)])
    calls = []
    history = fit(model, SoftmaxCrossEntropy(), Adam(model.params(), lr=1e-2), X, y,
                  epochs=15, batch_size=32, val_data=(X_val, y_val), rng=rng,
                  verbose=False, on_epoch_end=lambda e, h: calls.append(e))

    assert calls == list(range(1, 16))
    assert all(len(v) == 15 for v in history.values())
    assert history["train_loss"][-1] < history["train_loss"][0]
    assert history["train_acc"][-1] > 0.95
    assert history["val_acc"][-1] > 0.95
    assert evaluate(model, SoftmaxCrossEntropy(), X_val, y_val)[1] > 0.95


def test_fit_learns_xor_float32(capsys):
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([0, 1, 1, 0])
    X, y = np.tile(X, (16, 1)), np.tile(y, 16)
    rng = np.random.default_rng(0)
    model = Sequential([Dense(2, 16, rng=rng), Tanh(), Dense(16, 2, rng=rng)]).astype(np.float32)
    history = fit(model, SoftmaxCrossEntropy(), Adam(model.params(), lr=0.05), X, y,
                  epochs=40, batch_size=16, rng=rng, verbose=True)

    assert history["val_loss"] == [] and history["val_acc"] == []
    assert accuracy(model.predict(X), y) == 1.0
    assert model.params()[0].data.dtype == np.float32
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 40 and lines[-1].startswith("epoch 40/40")
