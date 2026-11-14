import numpy as np
import pytest

from numpynet import (
    Dense, BatchNorm, Flatten, ReLU, LeakyReLU, Sigmoid, Tanh,
    MSE, SoftmaxCrossEntropy, Sequential,
    numerical_gradient, relative_error, check_layer, check_model,
)

TOL = 1e-6


def away_from_zero(rng, shape, low=0.1):
    # keeps inputs off the kink of ReLU-style activations
    return rng.uniform(low, 2.0, size=shape) * rng.choice([-1.0, 1.0], size=shape)


def assert_all_small(errors, tol=TOL):
    for name, err in errors.items():
        assert err < tol, f"{name}: relative error {err:.2e}"


def test_numerical_gradient_restores_input():
    x = np.array([1.0, -2.0, 3.0])
    g = numerical_gradient(lambda: float(np.sum(x ** 3)), x)
    np.testing.assert_allclose(g, 3 * x ** 2, rtol=1e-8)
    np.testing.assert_array_equal(x, [1.0, -2.0, 3.0])


def test_relative_error_basic():
    assert relative_error(np.ones(3), np.ones(3)) == 0.0
    assert relative_error(np.zeros(3), np.zeros(3)) == 0.0
    assert relative_error(np.array([1.0]), np.array([-1.0])) == pytest.approx(1.0)


@pytest.mark.parametrize("init", ["he", "xavier", "normal"])
def test_dense(init):
    rng = np.random.default_rng(0)
    layer = Dense(5, 4, init=init, rng=rng)
    layer.b.data[:] = rng.standard_normal(4)
    errors = check_layer(layer, rng.standard_normal((6, 5)))
    assert set(errors) == {"input", "W", "b"}
    assert_all_small(errors)


@pytest.mark.parametrize("layer", [ReLU(), LeakyReLU(0.1), Sigmoid(), Tanh()],
                         ids=lambda l: type(l).__name__)
def test_activations(layer):
    rng = np.random.default_rng(1)
    assert_all_small(check_layer(layer, away_from_zero(rng, (4, 7))))


def test_batchnorm_training():
    rng = np.random.default_rng(2)
    layer = BatchNorm(5)
    layer.gamma.data[:] = rng.uniform(0.5, 1.5, 5)
    layer.beta.data[:] = rng.standard_normal(5)
    x = rng.standard_normal((8, 5)) * 3 + 1
    errors = check_layer(layer, x)
    assert set(errors) == {"input", "gamma", "beta"}
    assert_all_small(errors)


def test_batchnorm_eval():
    rng = np.random.default_rng(3)
    layer = BatchNorm(4)
    layer.running_mean = rng.standard_normal(4)
    layer.running_var = rng.uniform(0.5, 2.0, 4)
    layer.training = False
    assert_all_small(check_layer(layer, rng.standard_normal((6, 4))))


def test_flatten():
    rng = np.random.default_rng(4)
    assert_all_small(check_layer(Flatten(), rng.standard_normal((3, 2, 4, 2))))


def test_mse():
    rng = np.random.default_rng(5)
    pred = rng.standard_normal((6, 3))
    target = rng.standard_normal((6, 3))
    loss = MSE()
    loss.forward(pred, target)
    analytic = loss.backward()
    numeric = numerical_gradient(lambda: loss.forward(pred, target), pred)
    assert relative_error(analytic, numeric) < TOL


@pytest.mark.parametrize("one_hot", [False, True])
def test_softmax_cross_entropy(one_hot):
    rng = np.random.default_rng(6)
    # moderate logits: very peaked softmaxes give ~1e-7 gradient entries where
    # finite-difference roundoff alone dominates the relative error
    logits = rng.standard_normal((7, 4))
    labels = rng.integers(0, 4, size=7)
    target = np.eye(4)[labels] if one_hot else labels
    loss = SoftmaxCrossEntropy()
    loss.forward(logits, target)
    analytic = loss.backward()
    numeric = numerical_gradient(lambda: loss.forward(logits, target), logits)
    assert relative_error(analytic, numeric) < TOL


def test_mlp():
    rng = np.random.default_rng(7)
    model = Sequential([
        Dense(4, 8, rng=rng), Tanh(),
        Dense(8, 6, rng=rng), LeakyReLU(0.1),
        Dense(6, 3, rng=rng),
    ])
    x = rng.standard_normal((10, 4))
    y = rng.integers(0, 3, size=10)
    errors = check_model(model, SoftmaxCrossEntropy(), x, y)
    assert set(errors) == {f"layer{i}.{n}" for i in (0, 2, 4) for n in ("W", "b")}
    assert_all_small(errors)


def test_mlp_with_batchnorm():
    rng = np.random.default_rng(9)
    model = Sequential([
        Dense(4, 8, rng=rng), BatchNorm(8), ReLU(),
        Dense(8, 6, rng=rng), Sigmoid(),
        Dense(6, 3, rng=rng),
    ])
    x = rng.standard_normal((12, 4))
    y = rng.integers(0, 3, size=12)
    errors = check_model(model, SoftmaxCrossEntropy(), x, y)
    assert "layer1.gamma" in errors and "layer1.beta" in errors

    # BatchNorm subtracts the batch mean, so the bias right before it has an exact
    # zero gradient; relative error there only measures finite-difference noise
    assert np.abs(model.layers[0].b.grad).max() < 1e-12
    del errors["layer0.b"]
    assert_all_small(errors)


def test_mlp_regression_mse():
    rng = np.random.default_rng(9)
    model = Sequential([Dense(3, 5, rng=rng), Tanh(), Dense(5, 2, rng=rng)])
    x = rng.standard_normal((8, 3))
    y = rng.standard_normal((8, 2))
    assert_all_small(check_model(model, MSE(), x, y))
