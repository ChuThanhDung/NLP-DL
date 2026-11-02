"""Sinusoidal positional encoding (Vaswani et al., 2017) — NumPy và PyTorch.

PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

Chạy `python positional_encoding.py` để kiểm tra: giá trị bài tính tay (d = 4, pos = 0, 1, 2) và các tính chất cơ bản.
"""
import numpy as np
import torch
import torch.nn as nn


def positional_encoding(n_pos, d, base=10000.0):
    """Ma trận PE (n_pos, d), NumPy float64. d phải chẵn."""
    assert d % 2 == 0, "d phải chẵn"
    pos = np.arange(n_pos)[:, None]                       # (n_pos, 1)
    i = np.arange(d // 2)[None, :]                        # (1, d/2)
    angle = pos / base ** (2 * i / d)                     # (n_pos, d/2)
    pe = np.zeros((n_pos, d))
    pe[:, 0::2], pe[:, 1::2] = np.sin(angle), np.cos(angle)
    return pe


class SinusoidalPE(nn.Module):
    """Cộng PE cố định (không học) vào embedding: X' = X + PE. x: (B, T, d)."""
    def __init__(self, d, max_len=512):
        super().__init__()
        self.register_buffer("pe", torch.tensor(positional_encoding(max_len, d), dtype=torch.float32), persistent=False)

    def forward(self, x):
        return x + self.pe[: x.shape[1]][None]


def run_tests():
    pe = positional_encoding(3, 4)
    expected = np.array([[0, 1, 0, 1],
                         [np.sin(1), np.cos(1), np.sin(0.01), np.cos(0.01)],
                         [np.sin(2), np.cos(2), np.sin(0.02), np.cos(0.02)]])
    assert np.allclose(pe, expected), "PE d = 4, pos = 0, 1, 2"
    assert np.allclose(pe[0], [0, 1, 0, 1]), "pos = 0 → [0, 1, 0, 1, ...]"
    big = positional_encoding(50, 64)
    assert len(np.unique(big.round(8), axis=0)) == 50, "mỗi vị trí có một vector khác nhau"
    assert np.all(np.abs(big) <= 1), "|PE| ≤ 1"
    # PE(pos + k) là phép quay tuyến tính của PE(pos): tích vô hướng chỉ phụ thuộc k, không phụ thuộc pos
    d = lambda a, b: big[a] @ big[b]
    assert np.isclose(d(3, 8), d(20, 25)) and np.isclose(d(0, 5), d(30, 35)), "PE(p)·PE(p+k) chỉ phụ thuộc k"
    x = torch.zeros(2, 5, 64)
    assert torch.allclose(SinusoidalPE(64)(x)[1, :, :], torch.tensor(big[:5], dtype=torch.float32), atol=1e-6)
    print("positional_encoding.py: 5 tests passed")


if __name__ == "__main__":
    run_tests()
    np.set_printoptions(precision=4, suppress=True)
    print("PE (d = 4, pos = 0, 1, 2):\n", positional_encoding(3, 4))
