"""LAB 06 — Attention tự cài đặt bằng NumPy (không dùng torch.nn.MultiheadAttention / keras.layers.Attention).

    scores  S = Q K^T / sqrt(d_k)          (scale=False: S = Q K^T, dot-product attention)
    weights A = softmax(S)                 (theo từng hàng; softmax ổn định số: trừ max trước khi exp)
    context C = A V

Self-attention: Q = X W_Q, K = X W_K, V = X W_V (cùng một sequence X).
Quy ước: mỗi HÀNG của X, Q, K, V là một token.
"""
import numpy as np


def naive_softmax(x, axis=-1):
    """Softmax tính thẳng exp(x) — dễ overflow với score lớn (chỉ dùng để so sánh)."""
    e = np.exp(np.asarray(x, dtype=float))
    return e / e.sum(axis=axis, keepdims=True)


def softmax(x, axis=-1, temperature=1.0):
    """Softmax ổn định số: exp(x/T - max) / sum(exp(x/T - max)). Phần tử -inf (bị mask) nhận weight 0."""
    z = np.asarray(x, dtype=float) / temperature
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def entropy(p, axis=-1):
    """Entropy (nat) của phân phối p: −Σ p log p. Đều trên n phần tử → log n; dồn vào 1 phần tử → 0."""
    p = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.sum(np.where(p > 0, p * np.log(p), 0.0), axis=axis)


def scaled_dot_product_attention(Q, K, V, mask=None, scale=True, temperature=1.0):
    """Attention cho một query (vector d_k) hoặc nhiều query (ma trận n_q x d_k).

    K: (n_k, d_k), V: (n_k, d_v). mask: bool (n_k,) hoặc (n_q, n_k), True = được attend.
    Trả về (scores, attention_weights, context) với scores đã chia sqrt(d_k) nếu scale=True.
    """
    Q, K, V = (np.asarray(a, dtype=float) for a in (Q, K, V))
    single = Q.ndim == 1
    Q2 = Q[None, :] if single else Q
    scores = Q2 @ K.T
    if scale:
        scores = scores / np.sqrt(K.shape[-1])
    if mask is not None:
        scores = np.where(np.asarray(mask, dtype=bool), scores, -np.inf)
    weights = softmax(scores, axis=-1, temperature=temperature)
    context = weights @ V
    if single:
        return scores[0], weights[0], context[0]
    return scores, weights, context


def self_attention(X, W_Q, W_K, W_V, scale=True):
    """Self-attention trên sequence X (T x d_model). Trả về dict Q, K, V, scores, weights, output."""
    X = np.asarray(X, dtype=float)
    Q, K, V = X @ np.asarray(W_Q, float), X @ np.asarray(W_K, float), X @ np.asarray(W_V, float)
    scores, weights, output = scaled_dot_product_attention(Q, K, V, scale=scale)
    return {"Q": Q, "K": K, "V": V, "QK^T": Q @ K.T, "scores": scores, "weights": weights, "output": output}


# ----------------------------------------------------------------------------- dữ liệu các bài tập
BAI1 = {"q": [1, 2], "K": [[1, 0], [0, 1], [1, 1]]}
BAI3 = {"alpha": [0.1, 0.2, 0.7], "V": [[1, 0], [0, 1], [2, 2]]}
BAI4 = {"q": [1, 0], "K": [[1, 0], [0, 1], [1, 1]], "V": [[1, 0], [0, 2], [2, 1]]}
# Đề không cho số cụ thể cho X, W_Q, W_K, W_V → tự chọn (xem calculations.md)
SELF_ATT = {"tokens": ["I", "like", "cats"],
            "X": [[1, 0], [0, 1], [1, 1]],
            "W_Q": [[1, 0], [1, 1]], "W_K": [[0, 1], [1, 0]], "W_V": [[1, 0], [0, 2]]}


# ----------------------------------------------------------------------------- tests
def test_softmax_basic():
    p = softmax([2, 1, 0])
    assert np.allclose(p, [0.66524, 0.24473, 0.09003], atol=1e-5) and abs(p.sum() - 1) < 1e-12
    assert np.allclose(softmax([1, 1, 1, 1]), 0.25)
    assert softmax([1, 1, 10])[2] > 0.9997


def test_softmax_stability():
    x = [1000, 1001, 1002]
    with np.errstate(over="ignore", invalid="ignore"):
        assert np.isnan(naive_softmax(x)).all()                 # exp(1000) = inf → inf/inf = nan
    assert np.allclose(softmax(x), softmax([0, 1, 2]))          # trừ hằng số không đổi kết quả
    assert np.allclose(softmax([-1000, -1001]), softmax([0, -1]))


def test_temperature_and_entropy():
    s = [4, 2, 0]
    assert np.allclose(softmax(s, temperature=2.0), softmax([2, 1, 0]))
    ents = [entropy(softmax(s, temperature=T)) for T in (0.5, 1.0, 2.0)]
    assert ents[0] < ents[1] < ents[2]                          # T tăng → phân phối phẳng hơn
    assert abs(entropy([0.25] * 4) - np.log(4)) < 1e-12


def test_bai1_to_bai4():
    s, _, _ = scaled_dot_product_attention(BAI1["q"], BAI1["K"], np.eye(3), scale=False)
    assert np.allclose(s, [1, 2, 3])
    assert np.allclose(np.asarray(BAI3["alpha"]) @ np.asarray(BAI3["V"], float), [1.5, 1.6])
    s, a, c = scaled_dot_product_attention(BAI4["q"], BAI4["K"], BAI4["V"], scale=False)
    e = np.e
    assert np.allclose(s, [1, 0, 1]) and np.allclose(a, [e / (2 * e + 1), 1 / (2 * e + 1), e / (2 * e + 1)])
    assert np.allclose(c, [1.266956, 0.733044], atol=1e-6)


def test_scaling_exercise():
    s, a, _ = scaled_dot_product_attention([1.0], [[4.0], [2.0], [0.0]], np.eye(3), scale=False)
    assert np.allclose(np.asarray([4, 2, 0]) / np.sqrt(4), [2, 1, 0])
    assert np.allclose(softmax(np.asarray([4, 2, 0]) / 2), [0.66524, 0.24473, 0.09003], atol=1e-5)
    assert np.allclose(softmax([4, 2, 0]), [0.86681, 0.11731, 0.01588], atol=1e-5)


def test_self_attention_exercise():
    r = self_attention(SELF_ATT["X"], SELF_ATT["W_Q"], SELF_ATT["W_K"], SELF_ATT["W_V"])
    assert np.allclose(r["Q"], [[1, 0], [1, 1], [2, 1]]) and np.allclose(r["K"], [[0, 1], [1, 0], [1, 1]])
    assert np.allclose(r["V"], [[1, 0], [0, 2], [1, 2]]) and np.allclose(r["QK^T"], [[0, 1, 1], [1, 1, 2], [1, 2, 3]])
    assert np.allclose(r["weights"].sum(axis=1), 1)
    assert np.allclose(r["weights"][2], [0.14003, 0.28400, 0.57598], atol=1e-5)


def test_mask():
    _, a, _ = scaled_dot_product_attention([1, 0], [[1, 0], [0, 1], [5, 5]], np.eye(3), mask=[True, True, False])
    assert a[2] == 0 and abs(a.sum() - 1) < 1e-12


def test_against_torch():
    """So khớp với torch.nn.functional.scaled_dot_product_attention (chỉ để kiểm chứng)."""
    import torch
    rng = np.random.default_rng(0)
    Q, K, V = rng.normal(size=(5, 8)), rng.normal(size=(7, 8)), rng.normal(size=(7, 3))
    _, _, ours = scaled_dot_product_attention(Q, K, V)
    ref = torch.nn.functional.scaled_dot_product_attention(*(torch.from_numpy(a)[None] for a in (Q, K, V)))[0].numpy()
    assert np.abs(ours - ref).max() < 1e-12


def run_tests():
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"{len(tests)}/{len(tests)} tests passed")


if __name__ == "__main__":
    run_tests()
