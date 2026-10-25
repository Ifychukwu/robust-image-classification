import ast
from pathlib import Path

import numpy as np
import pytest

from vision.evaluate import ece, load_data, saliency, shift
from vision.numpy_cnn import CNN, Conv2D, Dense, MaxPool2D, col2im, cross_entropy, im2col, softmax, train

ROOT = Path(__file__).resolve().parents[1]


def test_im2col_matches_direct_convolution():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 3, 5, 5))
    conv = Conv2D(3, 4, k=3, pad=1, rng=rng)
    out = conv.forward(x)
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    direct = np.zeros_like(out)
    for n in range(2):
        for f in range(4):
            for i in range(5):
                for j in range(5):
                    direct[n, f, i, j] = (xp[n, :, i:i + 3, j:j + 3] * conv.params["W"][f]).sum() + conv.params["b"][f]
    assert np.allclose(out, direct)


def test_col2im_is_adjoint_of_im2col():
    # <im2col(x), c> == <x, col2im(c)> for all x, c
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 2, 6, 6))
    cols = im2col(x, 3, 1, 1)
    c = rng.normal(size=cols.shape)
    assert np.isclose((cols * c).sum(), (x * col2im(c, x.shape, 3, 1, 1)).sum())


def test_gradients_match_finite_differences():
    rng = np.random.default_rng(2)
    m = CNN((1, 8, 8), 3, dropout=0.0)
    x, y = rng.normal(size=(3, 1, 8, 8)), np.array([0, 2, 1])
    loss, g = cross_entropy(m.forward(x, True), y)
    m.backward(g)
    for layer in m.param_layers():
        for k, p in layer.params.items():
            idx = tuple(rng.integers(0, s) for s in p.shape)
            old = p[idx]
            p[idx] = old + 1e-5; lp, _ = cross_entropy(m.forward(x, True), y)
            p[idx] = old - 1e-5; lm, _ = cross_entropy(m.forward(x, True), y)
            p[idx] = old
            assert abs((lp - lm) / 2e-5 - layer.grads[k][idx]) < 1e-6


def test_maxpool_routes_gradient_to_max():
    pool = MaxPool2D(2)
    x = np.array([[[[1, 2], [3, 4]]]], dtype=float)
    assert pool.forward(x)[0, 0, 0, 0] == 4
    assert np.array_equal(pool.backward(np.ones((1, 1, 1, 1)))[0, 0], [[0, 0], [0, 1]])


def test_softmax_and_cross_entropy():
    p = softmax(np.array([[1000.0, 1000.0]]))  # numerically stable
    assert np.allclose(p, 0.5)
    loss, _ = cross_entropy(np.log(np.array([[0.7, 0.3]])), np.array([0]))
    assert loss == pytest.approx(-np.log(0.7))


def test_small_cnn_learns_digits_quickly():
    x_tr, y_tr, x_val, y_val, x_te, y_te = load_data()
    m = CNN(seed=1)
    train(m, x_tr, y_tr, x_val, y_val, epochs=8)
    assert (m.predict(x_te) == y_te).mean() > 0.9


def test_shift_and_saliency_shapes():
    x = np.zeros((1, 1, 8, 8)); x[0, 0, 3, 3] = 1
    assert shift(x, 1, 0)[0, 0, 4, 3] == 1
    m = CNN()
    assert saliency(m, x[0], 3).shape == (8, 8)


def test_ece_bounds():
    y = np.array([0, 1]); p = np.array([[0.9, 0.1], [0.2, 0.8]])
    assert 0 <= ece(y, p) <= 1


def test_transfer_learning_module_is_valid_python():
    tree = ast.parse((ROOT / "src" / "vision" / "transfer_learning.py").read_text())
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert {"build_model", "load_datasets", "main", "grad_cam"} <= names
