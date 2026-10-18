"""LAB 03 — Word-context (co-occurrence) representation tự cài đặt (không dùng Word2Vec).

X[i, j] = số lần context word j xuất hiện trong cửa sổ ±window quanh target word i (không vượt qua ranh giới câu).
Token ngoài vocabulary vẫn giữ vị trí (không làm hai từ ở xa trở thành "kề nhau") nhưng không được đếm.
"""
from collections import Counter

import numpy as np
import scipy.sparse as sp


def build_vocabulary(sentences, min_count=1, max_size=None, exclude=()):
    """Trả về list từ, sắp xếp theo tần suất giảm dần (hoà thì theo alphabet)."""
    counts = Counter(w for s in sentences for w in s if w not in exclude)
    items = sorted((w for w, c in counts.items() if c >= min_count), key=lambda w: (-counts[w], w))
    return items[:max_size] if max_size else items


def build_cooccurrence_matrix(sentences, vocabulary, window=1):
    """Ma trận |V| x |V| (scipy CSR, đối xứng) đếm co-occurrence trong cửa sổ đối xứng."""
    index = {w: i for i, w in enumerate(vocabulary)}
    ids = np.array([index.get(w, -1) for s in sentences for w in s], dtype=np.int64)
    sent = np.repeat(np.arange(len(sentences)), [len(s) for s in sentences])
    V = len(vocabulary)
    X = sp.csr_matrix((V, V), dtype=np.float64)
    for d in range(1, window + 1):      # cộng dồn từng khoảng cách d để tiết kiệm bộ nhớ
        a, b = ids[:-d], ids[d:]
        ok = (a >= 0) & (b >= 0) & (sent[:-d] == sent[d:])
        a, b = a[ok], b[ok]
        half = sp.coo_matrix((np.ones(len(a)), (a, b)), shape=(V, V)).tocsr()
        X = X + half + half.T           # (target, context) theo cả hai chiều
    return X


def ppmi(X):
    """Positive PMI: max(0, log P(w,c) / (P(w) P(c))) — giảm trọng số các context quá phổ biến (the, of…)."""
    X = sp.csr_matrix(X, dtype=np.float64)
    total = X.sum()
    row = np.asarray(X.sum(axis=1)).ravel()
    col = np.asarray(X.sum(axis=0)).ravel()
    coo = X.tocoo()
    pmi = np.log(coo.data * total / (row[coo.row] * col[coo.col]))
    keep = pmi > 0
    return sp.csr_matrix((pmi[keep], (coo.row[keep], coo.col[keep])), shape=X.shape)


def cosine_similarity(u, v):
    """cos(u, v) cho vector numpy hoặc hàng sparse; 0 nếu một vector bằng 0."""
    u = u.toarray().ravel() if sp.issparse(u) else np.asarray(u, dtype=float).ravel()
    v = v.toarray().ravel() if sp.issparse(v) else np.asarray(v, dtype=float).ravel()
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    return 0.0 if nu == 0 or nv == 0 else float(u @ v / (nu * nv))


def most_similar(word, matrix, vocabulary, top_k=5):
    """Top-k từ có cosine cao nhất với word (bỏ chính nó). matrix: hàng = word vectors (sparse hoặc dense)."""
    index = {w: i for i, w in enumerate(vocabulary)}
    if word not in index:
        raise KeyError(f"'{word}' không có trong vocabulary")
    M = matrix.tocsr() if sp.issparse(matrix) else np.asarray(matrix, dtype=float)
    norms = np.sqrt(np.asarray(M.multiply(M).sum(axis=1)).ravel()) if sp.issparse(M) else np.linalg.norm(M, axis=1)
    q = M[index[word]]
    sims = (M @ q.T).toarray().ravel() if sp.issparse(M) else M @ q
    with np.errstate(divide="ignore", invalid="ignore"):
        sims = np.where(norms > 0, sims / (norms * norms[index[word]]), 0.0)
    sims[index[word]] = -np.inf
    order = np.argsort(-sims)[:top_k]
    return [(vocabulary[i], float(sims[i])) for i in order]


# ----------------------------------------------------------------------------- unit tests
SEC5 = [s.split() for s in ["the cat eats fish", "the cat likes milk", "the dog eats meat", "the dog likes fish"]]
BAI1 = [s.split() for s in ["the cat eats fish", "the dog eats fish", "the cat likes milk", "the dog likes meat"]]
VOCAB7 = ["cat", "dog", "eats", "likes", "fish", "milk", "meat"]


def test_build_vocabulary():
    v = build_vocabulary(BAI1, exclude={"the"})
    assert set(v) == set(VOCAB7) and v[0] in {"cat", "dog", "eats", "likes", "fish"}


def test_cooccurrence_section5():
    X = build_cooccurrence_matrix(SEC5, VOCAB7, window=1).toarray()
    assert X[0].tolist() == [0, 0, 1, 1, 0, 0, 0]          # cat
    assert X[2].tolist() == [1, 1, 0, 0, 1, 0, 1]          # eats
    assert X[3].tolist() == [1, 1, 0, 0, 1, 1, 0]          # likes
    assert (X == X.T).all()


def test_cooccurrence_bai1_and_window():
    X = build_cooccurrence_matrix(BAI1, VOCAB7, window=1).toarray()
    assert X[2].tolist() == [1, 1, 0, 0, 2, 0, 0]          # eats: cat, dog, fish×2
    assert X[3].tolist() == [1, 1, 0, 0, 0, 1, 1]          # likes: cat, dog, milk, meat
    X2 = build_cooccurrence_matrix(BAI1, VOCAB7, window=2).toarray()
    assert X2[0, 4] == 1                                   # cat–fish cách 2 vị trí trong "the cat eats fish"


def test_cosine_and_most_similar():
    X = build_cooccurrence_matrix(BAI1, VOCAB7, window=1)
    assert abs(cosine_similarity(X[0], X[1]) - 1.0) < 1e-12           # cat ≡ dog
    assert abs(cosine_similarity(X[2], X[3]) - 2 / np.sqrt(24)) < 1e-12
    assert abs(cosine_similarity([1, 2, 1], [2, 4, 2]) - 1.0) < 1e-12
    assert most_similar("cat", X, VOCAB7, top_k=1)[0][0] == "dog"
    assert most_similar("cat", X.toarray(), VOCAB7, top_k=1)[0][0] == "dog"


def test_ppmi():
    P = ppmi(build_cooccurrence_matrix(BAI1, VOCAB7, window=1))
    assert (P.data > 0).all() and P.shape == (7, 7)


def run_tests():
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"{len(tests)}/{len(tests)} tests passed")


if __name__ == "__main__":
    run_tests()
