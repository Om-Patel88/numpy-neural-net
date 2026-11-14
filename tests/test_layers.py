import numpy as np
import pytest

from numpynet import (
    Dense, Dropout, BatchNorm, Flatten, ReLU, LeakyReLU, Sigmoid, Tanh,
    softmax, SoftmaxCrossEntropy,
)


def test_dense_shapes():
    layer = Dense(5, 3, rng=np.random.default_rng(0))
    x = np.ones((4, 5))
    out = layer.forward(x)
    assert out.shape == (4, 3)
    dx = layer.backward(np.ones_like(out))
    assert dx.shape == x.shape
    assert layer.W.grad.shape == (5, 3) and layer.b.grad.shape == (3,)


def test_backward_overwrites_grads():
    rng = np.random.default_rng(0)
    layer = Dense(3, 2, rng=rng)
    x = rng.standard_normal((4, 3))
    g = rng.standard_normal((4, 2))
    layer.forward(x)
    layer.backward(g)
    first = layer.W.grad.copy()
    layer.forward(x)
    layer.backward(g)
    np.testing.assert_allclose(layer.W.grad, first)


@pytest.mark.parametrize("init, expected", [
    ("he", np.sqrt(2 / 1000)),
    ("xavier", np.sqrt(2 / 1500)),
    ("normal", 0.01),
])
def test_init_scale(init, expected):
    layer = Dense(1000, 500, init=init, rng=np.random.default_rng(0))
    assert layer.W.data.std() == pytest.approx(expected, rel=0.02)
    assert abs(layer.W.data.mean()) < 0.1 * expected
    assert np.all(layer.b.data == 0)


def test_unknown_init():
    with pytest.raises(ValueError):
        Dense(2, 2, init="uniform")


def test_dropout_train_scaling():
    layer = Dropout(0.3, rng=np.random.default_rng(0))
    x = np.ones((1000, 100))
    out = layer.forward(x)
    kept = out != 0
    assert np.allclose(out[kept], 1 / 0.7)
    assert kept.mean() == pytest.approx(0.7, abs=0.01)
    assert out.mean() == pytest.approx(1.0, abs=0.02)
    # backward uses the same mask
    np.testing.assert_array_equal(layer.backward(np.ones_like(x)), out)


def test_dropout_eval_is_identity():
    layer = Dropout(0.5, rng=np.random.default_rng(0))
    layer.training = False
    x = np.random.default_rng(1).standard_normal((10, 5))
    np.testing.assert_array_equal(layer.forward(x), x)
    np.testing.assert_array_equal(layer.backward(x), x)


def test_dropout_keeps_float32():
    layer = Dropout(0.5, rng=np.random.default_rng(0))
    x = np.ones((8, 4), dtype=np.float32)
    assert layer.forward(x).dtype == np.float32
    assert layer.backward(x).dtype == np.float32


def test_batchnorm_normalizes_batch():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((256, 6)) * 4 + 7
    out = BatchNorm(6).forward(x)
    np.testing.assert_allclose(out.mean(axis=0), 0, atol=1e-10)
    np.testing.assert_allclose(out.std(axis=0), 1, atol=1e-4)


def test_batchnorm_running_stats_and_eval():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((128, 3)) * 2 - 5
    layer = BatchNorm(3, momentum=0.9)
    for _ in range(200):
        train_out = layer.forward(x)
    np.testing.assert_allclose(layer.running_mean, x.mean(axis=0), rtol=1e-6)
    np.testing.assert_allclose(layer.running_var, x.var(axis=0), rtol=1e-6)

    layer.training = False
    np.testing.assert_allclose(layer.forward(x), train_out, atol=1e-6)
    # eval mode must not touch the running stats
    before = layer.running_mean.copy()
    layer.forward(x + 100)
    np.testing.assert_array_equal(layer.running_mean, before)


def test_batchnorm_first_step_update():
    x = np.array([[1.0], [3.0]])
    layer = BatchNorm(1, momentum=0.9)
    layer.forward(x)
    assert layer.running_mean[0] == pytest.approx(0.1 * 2.0)
    assert layer.running_var[0] == pytest.approx(0.9 * 1.0 + 0.1 * 1.0)


def test_flatten_round_trip():
    x = np.arange(2 * 3 * 4 * 5, dtype=float).reshape(2, 3, 4, 5)
    layer = Flatten()
    out = layer.forward(x)
    assert out.shape == (2, 60)
    np.testing.assert_array_equal(layer.backward(out), x)


def test_activation_values():
    x = np.array([[-2.0, -0.5, 0.5, 2.0]])
    np.testing.assert_allclose(ReLU().forward(x), [[0, 0, 0.5, 2.0]])
    np.testing.assert_allclose(LeakyReLU(0.1).forward(x), [[-0.2, -0.05, 0.5, 2.0]])
    np.testing.assert_allclose(Sigmoid().forward(x), 1 / (1 + np.exp(-x)))
    np.testing.assert_allclose(Tanh().forward(x), np.tanh(x))


def test_sigmoid_is_stable():
    x = np.array([-1000.0, -50.0, 0.0, 50.0, 1000.0])
    with np.errstate(all="raise"):
        out = Sigmoid().forward(x)
    assert out[0] == 0.0 and out[-1] == 1.0 and out[2] == 0.5


def test_softmax_sums_to_one():
    z = np.random.default_rng(0).standard_normal((5, 7)) * 10
    p = softmax(z)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    assert np.all(p >= 0)
    np.testing.assert_allclose(softmax(z.T, axis=0).sum(axis=0), 1.0)


def test_softmax_huge_logits():
    z = np.array([[1000.0, 1001.0, 1002.0], [-1000.0, -1001.0, -1002.0]])
    with np.errstate(all="raise"):
        p = softmax(z)
    np.testing.assert_allclose(p[0], softmax(np.array([0.0, 1.0, 2.0])))
    np.testing.assert_allclose(p[1], softmax(np.array([0.0, -1.0, -2.0])))


def test_cross_entropy_formats_agree_and_stable():
    logits = np.array([[1000.0, 0.0, -1000.0], [2.0, 1.0, 0.1]])
    labels = np.array([0, 2])
    loss = SoftmaxCrossEntropy()
    a = loss(logits, labels)
    grad_a = loss.backward()
    b = loss(logits, np.eye(3)[labels])
    grad_b = loss.backward()
    assert np.isfinite(a) and a == pytest.approx(b)
    np.testing.assert_allclose(grad_a, grad_b)
    expected = -np.log(softmax(logits[1])[2]) / 2  # first row contributes ~0
    assert a == pytest.approx(expected)
