import numpy as np

from bladebot.nn import MLP, Adam, bce_with_logits, sigmoid


def _loss(net, x, y):
    loss, _ = bce_with_logits(net.forward(x), y)
    return loss


def test_gradients_match_finite_differences():
    rng = np.random.default_rng(0)
    for act in ("tanh", "relu", "leaky_relu"):
        net = MLP([5, 7, 4], activation=act, seed=1, dtype=np.float64)
        x = rng.normal(size=(6, 5))
        y = (rng.random((6, 4)) < 0.5).astype(np.float64)
        logits = net.forward(x, train=True)
        _, grad = bce_with_logits(logits, y)
        gw, gb = net.backward(grad)
        eps = 1e-6
        for layer in range(net.n_layers):
            for (i, j) in [(0, 0), (1, 2), (net.weights[layer].shape[0] - 1, net.weights[layer].shape[1] - 1)]:
                w = net.weights[layer]
                old = w[i, j]
                w[i, j] = old + eps
                lp = _loss(net, x, y)
                w[i, j] = old - eps
                lm = _loss(net, x, y)
                w[i, j] = old
                num = (lp - lm) / (2 * eps)
                assert abs(num - gw[layer][i, j]) < 1e-6 + 1e-4 * abs(num), (act, layer, i, j)
            b = net.biases[layer]
            old = b[0]
            b[0] = old + eps
            lp = _loss(net, x, y)
            b[0] = old - eps
            lm = _loss(net, x, y)
            b[0] = old
            assert abs((lp - lm) / (2 * eps) - gb[layer][0]) < 1e-6


def test_sigmoid_is_stable():
    z = np.array([-1000.0, -10.0, 0.0, 10.0, 1000.0])
    s = sigmoid(z)
    assert np.all(np.isfinite(s))
    assert s[0] < 1e-6 and abs(s[2] - 0.5) < 1e-9 and s[-1] > 1 - 1e-6


def test_adam_learns_xor():
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)
    net = MLP([2, 16, 1], activation="tanh", seed=3)
    opt = Adam(net.parameters(), lr=0.05)
    for _ in range(600):
        loss, grad = bce_with_logits(net.forward(x, train=True), y)
        gw, gb = net.backward(grad)
        opt.step([*gw, *gb])
    pred = sigmoid(net.forward(x)) > 0.5
    assert np.array_equal(pred, y > 0.5)
    assert loss < 0.1


def test_save_load_roundtrip(tmp_path):
    net = MLP([3, 8, 2], activation="relu", seed=5)
    path = tmp_path / "net.npz"
    net.save(path, {"hello": "world"})
    net2, meta = MLP.load(path)
    x = np.random.default_rng(1).normal(size=(4, 3)).astype(np.float32)
    assert np.allclose(net.forward(x), net2.forward(x))
    assert meta["hello"] == "world"
