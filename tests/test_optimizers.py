import numpy as np
import pytest

from numpynet import Parameter, SGD, Adam


def param(values, grad=None, dtype=np.float64):
    p = Parameter(np.array(values, dtype=dtype), "p")
    if grad is not None:
        p.grad[...] = grad
    return p


def test_sgd_single_step():
    p = param([1.0, 2.0], grad=[0.5, -1.0])
    SGD([p], lr=0.1).step()
    np.testing.assert_allclose(p.data, [0.95, 2.1])


def test_sgd_momentum_two_steps():
    # constant gradient 1: v1 = 1, v2 = 0.9 * 1 + 1 = 1.9
    p = param([0.0], grad=[1.0])
    opt = SGD([p], lr=0.1, momentum=0.9)
    opt.step()
    np.testing.assert_allclose(p.data, [-0.1])
    opt.step()
    np.testing.assert_allclose(p.data, [-0.1 - 0.19])


def test_sgd_nesterov_two_steps():
    # step uses g + mu * v: 1 + 0.9 * 1 = 1.9, then 1 + 0.9 * 1.9 = 2.71
    p = param([0.0], grad=[1.0])
    opt = SGD([p], lr=0.1, momentum=0.9, nesterov=True)
    opt.step()
    np.testing.assert_allclose(p.data, [-0.19])
    opt.step()
    np.testing.assert_allclose(p.data, [-0.19 - 0.271])


def test_sgd_weight_decay():
    p = param([2.0], grad=[0.0])
    SGD([p], lr=0.5, weight_decay=0.1).step()
    np.testing.assert_allclose(p.data, [2.0 - 0.5 * 0.2])
    # the decay term must not leak into the stored gradient
    np.testing.assert_array_equal(p.grad, [0.0])


def test_adam_two_steps():
    p = param([1.0], grad=[2.0])
    opt = Adam([p], lr=0.1)
    opt.step()
    # m = 0.2, v = 0.004, bias corrected: m_hat = 2, v_hat = 4
    np.testing.assert_allclose(p.data, [1.0 - 0.1 * 2.0 / (2.0 + 1e-8)])

    p.grad[...] = 1.0
    opt.step()
    # m = 0.9*0.2 + 0.1*1 = 0.28, v = 0.999*0.004 + 0.001*1 = 0.004996
    m_hat = 0.28 / (1 - 0.9 ** 2)
    v_hat = 0.004996 / (1 - 0.999 ** 2)
    expected = 1.0 - 0.1 * 2.0 / (2.0 + 1e-8) - 0.1 * m_hat / (np.sqrt(v_hat) + 1e-8)
    np.testing.assert_allclose(p.data, [expected], rtol=1e-12)
    assert p.data[0] == pytest.approx(0.9 - 0.0932183, abs=1e-6)


def test_adam_weight_decay():
    p = param([3.0], grad=[0.0])
    Adam([p], lr=0.01, weight_decay=0.5).step()
    # gradient becomes 1.5, first Adam step moves by roughly lr * sign(g)
    np.testing.assert_allclose(p.data, [3.0 - 0.01 * 1.5 / (1.5 + 1e-8)])


def test_zero_grad():
    p = param([1.0, 2.0], grad=[3.0, 4.0])
    SGD([p]).zero_grad()
    np.testing.assert_array_equal(p.grad, 0)


@pytest.mark.parametrize("make_opt", [
    lambda ps: SGD(ps, lr=0.1),
    lambda ps: SGD(ps, lr=0.05, momentum=0.9),
    lambda ps: SGD(ps, lr=0.05, momentum=0.9, nesterov=True),
    lambda ps: Adam(ps, lr=0.05),
], ids=["sgd", "momentum", "nesterov", "adam"])
def test_converges_on_quadratic(make_opt):
    # f(p) = 0.5 * sum(a * (p - c)^2), minimum at c
    a = np.array([1.0, 4.0, 0.5])
    c = np.array([3.0, -1.0, 2.0])
    p = param(np.zeros(3))
    opt = make_opt([p])
    for _ in range(500):
        p.grad[...] = a * (p.data - c)
        opt.step()
    np.testing.assert_allclose(p.data, c, atol=1e-3)


@pytest.mark.parametrize("make_opt", [
    lambda ps: SGD(ps, lr=0.1, weight_decay=1e-2),
    lambda ps: SGD(ps, lr=0.1, momentum=0.9, nesterov=True, weight_decay=1e-2),
    lambda ps: Adam(ps, lr=0.1, weight_decay=1e-2),
], ids=["sgd", "nesterov", "adam"])
def test_float32_preserved_in_place(make_opt):
    p = param(np.ones((3, 2)), grad=np.full((3, 2), 0.5), dtype=np.float32)
    data = p.data
    opt = make_opt([p])
    for _ in range(3):
        opt.step()
    assert p.data is data
    assert p.data.dtype == np.float32
    assert not np.allclose(p.data, 1.0)
