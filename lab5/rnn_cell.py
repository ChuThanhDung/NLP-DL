"""LAB 05 — Simple RNN cell tự cài đặt bằng NumPy:  h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h).

Quy ước giống phần lý thuyết: W_xh có shape (hidden, input), W_hh có shape (hidden, hidden), vector cột.
Kiểm chứng với torch.nn.RNNCell (công thức của PyTorch: tanh(W_ih x + b_ih + W_hh h + b_hh)).
"""
import numpy as np


class SimpleRNNCell:
    def __init__(self, input_size, hidden_size, seed=0):
        rng = np.random.default_rng(seed)
        k = 1 / np.sqrt(hidden_size)                       # cùng kiểu khởi tạo với PyTorch: U(-k, k)
        self.W_xh = rng.uniform(-k, k, (hidden_size, input_size))
        self.W_hh = rng.uniform(-k, k, (hidden_size, hidden_size))
        self.b_h = np.zeros(hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x_t, h_prev):
        """Một bước: trả về (h_t, a_t) với a_t = W_xh x_t + W_hh h_prev + b_h (pre-activation)."""
        a_t = self.W_xh @ np.asarray(x_t, dtype=float) + self.W_hh @ np.asarray(h_prev, dtype=float) + self.b_h
        return np.tanh(a_t), a_t

    def forward_sequence(self, xs, h0=None):
        """Chạy qua cả sequence; trả về list hidden states h_1..h_T."""
        h = np.zeros(self.hidden_size) if h0 is None else np.asarray(h0, dtype=float)
        hs = []
        for x_t in xs:
            h, _ = self.forward(x_t, h)
            hs.append(h)
        return hs


def check_against_torch(input_size=4, hidden_size=3, T=6, seed=0):
    """So sánh từng hidden state với torch.nn.RNNCell khi chép cùng trọng số; trả về max |diff|."""
    import torch

    cell = SimpleRNNCell(input_size, hidden_size, seed=seed)
    cell.b_h = np.random.default_rng(seed + 1).normal(size=hidden_size)
    ref = torch.nn.RNNCell(input_size, hidden_size, nonlinearity="tanh").double()
    with torch.no_grad():
        ref.weight_ih.copy_(torch.from_numpy(cell.W_xh))
        ref.weight_hh.copy_(torch.from_numpy(cell.W_hh))
        ref.bias_ih.copy_(torch.from_numpy(cell.b_h))
        ref.bias_hh.zero_()
    xs = np.random.default_rng(seed + 2).normal(size=(T, input_size))
    ours = cell.forward_sequence(xs)
    h = torch.zeros(1, hidden_size, dtype=torch.float64)
    diffs = []
    for t in range(T):
        h = ref(torch.from_numpy(xs[t:t + 1]), h)
        diffs.append(np.abs(h.detach().numpy()[0] - ours[t]).max())
    return max(diffs)


# ----------------------------------------------------------------------------- tests
def test_bai1_hand_calculation():
    cell = SimpleRNNCell(2, 2)
    cell.W_xh = np.array([[0.5, 0.2], [0.1, 0.4]])
    cell.W_hh = np.array([[0.3, 0.1], [0.2, 0.5]])
    cell.b_h = np.zeros(2)
    h, a = cell.forward([1, 0], [0.5, 0.2])
    assert np.allclose(a, [0.67, 0.30])
    assert np.allclose(h, np.tanh([0.67, 0.30])) and abs(h[0] - 0.5850) < 1e-4 and abs(h[1] - 0.2913) < 1e-4


def test_hidden_state_depends_on_order():
    cell = SimpleRNNCell(3, 4, seed=1)
    x = np.eye(3)
    h_abc = cell.forward_sequence([x[0], x[1], x[2]])[-1]
    h_cba = cell.forward_sequence([x[2], x[1], x[0]])[-1]
    assert not np.allclose(h_abc, h_cba)                   # khác Average Word2Vec: thứ tự làm thay đổi h_T
    assert np.allclose(np.mean([x[0], x[1], x[2]], 0), np.mean([x[2], x[1], x[0]], 0))


def test_against_torch():
    assert check_against_torch() < 1e-12


def run_tests():
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"{len(tests)}/{len(tests)} tests passed")


if __name__ == "__main__":
    run_tests()
