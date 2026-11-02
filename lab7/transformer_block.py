"""Transformer Encoder block tối giản (post-LN, như sơ đồ trong đề) — tự ghép từ Linear / softmax / LayerNorm / Dropout.

    x → Multi-Head Self-Attention → Add & Norm → Feed-Forward → Add & Norm → output

Không dùng nn.MultiheadAttention / nn.TransformerEncoderLayer trong phần core; hai lớp đó chỉ xuất hiện trong
`run_tests()` để đối chiếu kết quả.  Chạy `python transformer_block.py` để chạy 8 test và in shape của từng bước.
"""
import math
import numpy as np
import torch
import torch.nn as nn


# ---------- NumPy: một attention head (bài tính tay) ----------
def softmax_np(s, axis=-1):
    e = np.exp(s - s.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def attention_head_np(X, W_Q, W_K, W_V):
    """Q = XW_Q, K = XW_K, V = XW_V; S = QKᵀ/√d_k; A = softmax(S); Z = AV."""
    Q, K, V = X @ W_Q, X @ W_K, X @ W_V
    S = Q @ K.T / np.sqrt(K.shape[-1])
    A = softmax_np(S)
    return {"Q": Q, "K": K, "V": V, "QK^T": Q @ K.T, "S": S, "A": A, "Z": A @ V}


# bài tính tay (calculations.md, Bài 1): X ∈ R^{3×4}, d_k = 2, W ∈ R^{4×2} tự chọn vì đề không cho số
BAI1 = {
    "tokens": ["I", "love", "NLP"],
    "X": np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]], float),
    "W_Q": np.array([[1, 0], [0, 1], [1, 0], [0, 1]], float),
    "W_K": np.array([[1, 0], [0, 1], [1, 0], [0, 0]], float),
    "W_V": np.array([[1, 0], [0, 1], [1, 1], [0, 2]], float),
}


# ---------- PyTorch ----------
class MultiHeadSelfAttention(nn.Module):
    """h head chạy song song: W_Q, W_K, W_V ∈ R^{d_model×d_model} chính là h ma trận d_model×d_k xếp cạnh nhau."""
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model phải chia hết cho n_heads"
        self.h, self.d_k = n_heads, d_model // n_heads
        self.W_Q, self.W_K, self.W_V, self.W_O = (nn.Linear(d_model, d_model) for _ in range(4))
        self.drop = nn.Dropout(dropout)

    def split(self, t):                                   # (B, T, d_model) → (B, h, T, d_k)
        B, T, _ = t.shape
        return t.view(B, T, self.h, self.d_k).transpose(1, 2)

    def forward(self, x, key_mask=None, return_attn=False, trace=None):
        """x: (B, T, d_model); key_mask: (B, T) bool, True = token thật (padding không được attend)."""
        Q, K, V = self.split(self.W_Q(x)), self.split(self.W_K(x)), self.split(self.W_V(x))
        S = Q @ K.transpose(-1, -2) / math.sqrt(self.d_k)              # (B, h, T, T)
        if key_mask is not None:
            S = S.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        A = self.drop(torch.softmax(S, dim=-1))
        heads = A @ V                                                   # (B, h, T, d_k)
        concat = heads.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)   # (B, T, h·d_k = d_model)
        out = self.W_O(concat)                                          # (B, T, d_model)
        if trace is not None:
            trace.update({"Q": Q.shape, "K": K.shape, "V": V.shape, "scores QKᵀ/√d_k": S.shape,
                          "attention A": A.shape, "heads A·V": heads.shape, "concat": concat.shape, "MHA output": out.shape})
        return (out, A) if return_attn else out


class FeedForward(nn.Module):
    """FFN(x) = W₂ σ(W₁x + b₁) + b₂, áp dụng độc lập cho từng vị trí."""
    def __init__(self, d_model, d_ff, dropout=0.0, activation=nn.ReLU):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), activation(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_mask=None, return_attn=False, trace=None):
        a, A = self.attn(x, key_mask, return_attn=True, trace=trace)
        x = self.norm1(x + self.drop(a))                  # Residual + LayerNorm
        f = self.ffn(x)
        out = self.norm2(x + self.drop(f))                # Residual + LayerNorm
        if trace is not None:
            trace.update({"FFN hidden (d_ff)": (*x.shape[:2], self.ffn.net[0].out_features), "FFN output": f.shape, "block output": out.shape})
        return (out, A) if return_attn else out


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def run_tests():
    torch.manual_seed(0)
    # 1. shape (đề, mục 18): B = 8, T = 100, d_model = 128, 4 head → d_k = 32
    B, T, d, h, dff = 8, 100, 128, 4, 512
    blk = TransformerBlock(d, h, dff).eval()
    tr = {}
    out = blk(torch.randn(B, T, d), trace=tr)
    assert tr["Q"] == tr["K"] == tr["V"] == (8, 4, 100, 32) and tr["attention A"] == (8, 4, 100, 100) and out.shape == (B, T, d)
    # 2. số tham số: MHA 4·(d²+d) + FFN (d·dff+dff + dff·d+d) + 2 LayerNorm 2·2d
    assert count_params(blk) == 4 * (d * d + d) + (d * dff + dff + dff * d + d) + 2 * 2 * d
    # 3. mỗi hàng attention tổng bằng 1
    _, A = blk.attn(torch.randn(2, 7, d), return_attn=True)
    assert torch.allclose(A.sum(-1), torch.ones(2, h, 7), atol=1e-6)
    # 4. khớp nn.MultiheadAttention khi chép trọng số
    ref = nn.MultiheadAttention(d, h, batch_first=True).eval()
    mha = MultiHeadSelfAttention(d, h).eval()
    with torch.no_grad():
        Wq, Wk, Wv = ref.in_proj_weight.chunk(3); bq, bk, bv = ref.in_proj_bias.chunk(3)
        for lin, w, b in ((mha.W_Q, Wq, bq), (mha.W_K, Wk, bk), (mha.W_V, Wv, bv), (mha.W_O, ref.out_proj.weight, ref.out_proj.bias)):
            lin.weight.copy_(w); lin.bias.copy_(b)
        x = torch.randn(3, 11, d)
        assert torch.allclose(mha(x), ref(x, x, x, need_weights=False)[0], atol=1e-5), "MHA ≠ nn.MultiheadAttention"
    # 5. khớp nn.TransformerEncoderLayer (post-LN, ReLU) khi chép trọng số
    enc = nn.TransformerEncoderLayer(d, h, dff, dropout=0.0, batch_first=True).eval()
    mine = TransformerBlock(d, h, dff, dropout=0.0).eval()
    with torch.no_grad():
        Wq, Wk, Wv = enc.self_attn.in_proj_weight.chunk(3); bq, bk, bv = enc.self_attn.in_proj_bias.chunk(3)
        for lin, w, b in ((mine.attn.W_Q, Wq, bq), (mine.attn.W_K, Wk, bk), (mine.attn.W_V, Wv, bv),
                          (mine.attn.W_O, enc.self_attn.out_proj.weight, enc.self_attn.out_proj.bias),
                          (mine.ffn.net[0], enc.linear1.weight, enc.linear1.bias), (mine.ffn.net[3], enc.linear2.weight, enc.linear2.bias)):
            lin.weight.copy_(w); lin.bias.copy_(b)
        mine.norm1.load_state_dict(enc.norm1.state_dict()); mine.norm2.load_state_dict(enc.norm2.state_dict())
        x = torch.randn(3, 11, d)
        assert torch.allclose(mine(x), enc(x), atol=1e-5), "block ≠ nn.TransformerEncoderLayer"
    # 6. padding mask: đổi nội dung vị trí padding không làm đổi output của token thật
    blk = TransformerBlock(d, h, dff, dropout=0.0).eval()
    x = torch.randn(2, 9, d); m = torch.ones(2, 9, dtype=torch.bool); m[:, 6:] = False
    x2 = x.clone(); x2[:, 6:] = torch.randn(2, 3, d)
    assert torch.allclose(blk(x, m)[:, :6], blk(x2, m)[:, :6], atol=1e-5), "padding rò rỉ vào token thật"
    # 7. không có thông tin vị trí → hoán vị token = hoán vị output (equivariance)
    perm = torch.tensor([2, 0, 1, 4, 3])
    x = torch.randn(1, 5, d)
    assert torch.allclose(blk(x)[:, perm], blk(x[:, perm]), atol=1e-5), "block không equivariant với hoán vị"
    # 8. attention của lớp PyTorch (h = 1, W = I) khớp hàm NumPy trên đầu vào 3×2
    r = attention_head_np(**{k: BAI1[k] for k in ("X", "W_Q", "W_K", "W_V")})
    one = MultiHeadSelfAttention(2, 1).eval()
    with torch.no_grad():
        for lin in (one.W_Q, one.W_K, one.W_V, one.W_O):
            lin.weight.copy_(torch.eye(2)); lin.bias.zero_()
        _, A = one(torch.tensor(r["Q"], dtype=torch.float32)[None], return_attn=True)
    ref_np = attention_head_np(r["Q"], np.eye(2), np.eye(2), np.eye(2))
    assert np.allclose(A[0, 0].numpy(), ref_np["A"], atol=1e-6) and np.allclose(r["A"].sum(-1), 1)
    print("transformer_block.py: 8 tests passed")


if __name__ == "__main__":
    run_tests()
    np.set_printoptions(precision=4, suppress=True)
    r = attention_head_np(**{k: BAI1[k] for k in ("X", "W_Q", "W_K", "W_V")})
    for k in ("Q", "K", "V", "QK^T", "S", "A", "Z"):
        print(f"{k} =\n{r[k]}")
    tr = {}
    TransformerBlock(128, 4, 256).eval()(torch.randn(8, 100, 128), trace=tr)
    print("\nShape trace (B = 8, T = 100, d_model = 128, 4 heads, d_ff = 256):")
    for k, v in tr.items():
        print(f"  {k:20s} {tuple(v)}")
