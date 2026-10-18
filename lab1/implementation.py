"""LAB 01 — Core implementation: TF, IDF, TF-IDF và cosine similarity (tự cài đặt, không dùng TfidfVectorizer).

Quy ước (giống phần lý thuyết):
    tf(t, d)    = c(t, d) / sum_t' c(t', d)
    idf(t)      = ln(N / df(t))                      (smooth=False, mặc định)
    idf_s(t)    = ln((1 + N) / (1 + df(t))) + 1      (smooth=True, giống scikit-learn)
    tfidf(t, d) = tf(t, d) * idf(t)
    cos(x, y)   = x.y / (||x|| ||y||)
Dùng log tự nhiên (ln), giống scikit-learn.
"""
import math
from collections import Counter

import numpy as np


def tokenize(text):
    """Tokenizer tối giản: lowercase + tách theo khoảng trắng."""
    return text.lower().split()


def build_vocabulary(docs, sort=True):
    """Trả về list các term. sort=True: thứ tự alphabet; sort=False: thứ tự xuất hiện đầu tiên."""
    seen = {}
    for doc in docs:
        for tok in tokenize(doc):
            seen.setdefault(tok, len(seen))
    return sorted(seen) if sort else list(seen)


def compute_counts(docs, vocab):
    """Ma trận count C (N x V): C[i, j] = số lần vocab[j] xuất hiện trong docs[i]."""
    index = {t: j for j, t in enumerate(vocab)}
    C = np.zeros((len(docs), len(vocab)), dtype=float)
    for i, doc in enumerate(docs):
        for tok, c in Counter(tokenize(doc)).items():
            if tok in index:
                C[i, index[tok]] = c
    return C


def compute_tf(counts):
    """TF = count / tổng số term trong document (mỗi hàng có tổng = 1; hàng rỗng giữ 0)."""
    totals = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)


def compute_idf(counts, smooth=False):
    """IDF của mỗi term. df(t) = số document có count > 0."""
    N = counts.shape[0]
    df = (counts > 0).sum(axis=0)
    if smooth:
        return np.log((1 + N) / (1 + df)) + 1
    return np.log(N / df)


def compute_tfidf(tf, idf, normalize=False):
    """TF-IDF = TF * IDF (broadcast theo cột). normalize=True: chuẩn hóa L2 từng hàng."""
    X = tf * idf
    if normalize:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = np.divide(X, norms, out=np.zeros_like(X), where=norms > 0)
    return X


def cosine_similarity(x, y):
    """cos(x, y); trả về 0 nếu một trong hai vector bằng 0."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if nx == 0 or ny == 0:
        return 0.0
    return float(x @ y / (nx * ny))


# ----------------------------------------------------------------------------- unit tests
CORPUS = ["cat eats fish", "dog eats fish", "cat likes fish"]


def test_build_vocabulary():
    assert build_vocabulary(CORPUS) == ["cat", "dog", "eats", "fish", "likes"]
    assert build_vocabulary(CORPUS, sort=False) == ["cat", "eats", "fish", "dog", "likes"]


def test_compute_counts():
    C = compute_counts(CORPUS, build_vocabulary(CORPUS))
    assert C.tolist() == [[1, 0, 1, 1, 0], [0, 1, 1, 1, 0], [1, 0, 0, 1, 1]]


def test_compute_tf():
    tf = compute_tf(compute_counts(CORPUS, build_vocabulary(CORPUS)))
    assert abs(tf[0, 0] - 1 / 3) < 1e-9          # tf(cat, D1)
    assert np.allclose(tf.sum(axis=1), 1.0)       # tổng TF của mỗi document = 1


def test_compute_idf():
    idf = compute_idf(compute_counts(CORPUS, build_vocabulary(CORPUS)))
    expected = [math.log(3 / 2), math.log(3), math.log(3 / 2), 0.0, math.log(3)]
    assert np.allclose(idf, expected)
    assert idf.argmin() == 3                      # fish có IDF thấp nhất (= 0)
    idf_s = compute_idf(compute_counts(CORPUS, build_vocabulary(CORPUS)), smooth=True)
    assert abs(idf_s[3] - 1.0) < 1e-9             # smooth: ln(4/4) + 1 = 1


def test_compute_tfidf():
    C = compute_counts(CORPUS, build_vocabulary(CORPUS))
    X = compute_tfidf(compute_tf(C), compute_idf(C))
    assert abs(X[0, 0] - math.log(1.5) / 3) < 1e-9   # tfidf(cat, D1) ≈ 0.1352
    assert X[0, 3] == 0.0                             # tfidf(fish, D1) = 0
    Xn = compute_tfidf(compute_tf(C), compute_idf(C), normalize=True)
    assert np.allclose(np.linalg.norm(Xn, axis=1), 1.0)


def test_cosine_similarity():
    assert abs(cosine_similarity([1, 1, 1], [1, 1, 0]) - 2 / math.sqrt(6)) < 1e-9
    assert abs(cosine_similarity([1, 2], [2, 4]) - 1.0) < 1e-9
    assert cosine_similarity([1, 0], [0, 1]) == 0.0
    assert cosine_similarity([0, 0], [1, 1]) == 0.0


def run_tests():
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"{len(tests)}/{len(tests)} tests passed")


if __name__ == "__main__":
    run_tests()
