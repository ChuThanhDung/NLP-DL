"""LAB 02 — N-gram language model tự cài đặt (không dùng thư viện LM).

Mỗi câu được bọc bằng (n-1) token <s> ở đầu và một token </s> ở cuối.
Các token được dự đoán (và tính vào perplexity) là mọi từ của câu + </s>; <s> chỉ làm context.

Smoothing:
    'mle'      P(w|h) = C(h,w) / C(h)                          (0 nếu C(h,w) = 0)
    'laplace'  P(w|h) = (C(h,w) + 1) / (C(h) + V)
    'addk'     P(w|h) = (C(h,w) + k) / (C(h) + kV)
    'interp'   P(w|h) = λ_1 · P_Laplace(w) + Σ_{j≥2} λ_j · P_MLE(w | j-1 từ trước),  Σλ = 1
               (linear interpolation; unigram dùng add-one để không bao giờ bằng 0; nếu context bậc j
               chưa từng gặp thì λ_j được chia lại cho các bậc còn lại để Σ_w P(w|h) luôn bằng 1)
V = số loại token có thể được dự đoán (vocabulary + <unk> + </s>, không gồm <s>).
"""
import math
import re
from collections import Counter, defaultdict

BOS, EOS, UNK = "<s>", "</s>", "<unk>"

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")
_MATH = re.compile(r"\$[^$]*\$")
_NUM = re.compile(r"\b\d+(?:[.,]\d+)*\b")
_TOKEN = re.compile(r"<num>|[a-z]+(?:[-'][a-z]+)*")


def split_sentences(text):
    return [s for s in _SENT_SPLIT.split(text) if s.strip()]


def tokenize(sentence):
    """lowercase; công thức $...$ → <num>; số → <num>; giữ từ (kể cả từ có gạch nối), bỏ dấu câu."""
    s = _MATH.sub(" <num> ", sentence)
    s = _NUM.sub(" <num> ", s.lower())
    return _TOKEN.findall(s)


def build_vocabulary(sentences, min_count=2):
    """Từ xuất hiện ít nhất min_count lần trong train; các từ còn lại sẽ được thay bằng <unk>."""
    counts = Counter(w for s in sentences for w in s)
    return {w for w, c in counts.items() if c >= min_count}


def replace_oov(sentences, vocab):
    return [[w if w in vocab else UNK for w in s] for s in sentences]


def pad(sentence, n):
    return [BOS] * (n - 1) + list(sentence) + [EOS]


def count_ngrams(sentences, n):
    """Counter của các n-gram (tuple) trên các câu đã pad cho bậc n."""
    c = Counter()
    for s in sentences:
        p = pad(s, n)
        for i in range(len(p) - n + 1):
            c[tuple(p[i:i + n])] += 1
    return c


class NGramLanguageModel:
    def __init__(self, n, smoothing="mle", k=1.0, lambdas=None):
        assert smoothing in {"mle", "laplace", "addk", "interp"}
        self.n, self.smoothing = n, smoothing
        self.k = 1.0 if smoothing == "laplace" else k
        self.lambdas = lambdas

    # ------------------------------------------------------------------ training
    def fit(self, sentences):
        """sentences: list các list token (đã thay OOV bằng <unk>)."""
        self.vocab = {w for s in sentences for w in s} | {EOS}
        self.V = len(self.vocab)
        # counts[j][(h..., w)] cho mọi bậc j = 1..n, pad theo bậc n để mọi bậc dùng cùng vị trí dự đoán
        self.counts = {j: Counter() for j in range(1, self.n + 1)}
        self.context_counts = {j: Counter() for j in range(1, self.n + 1)}
        for s in sentences:
            p = pad(s, self.n)
            for i in range(self.n - 1, len(p)):
                for j in range(1, self.n + 1):
                    gram = tuple(p[i - j + 1:i + 1])
                    self.counts[j][gram] += 1
                    self.context_counts[j][gram[:-1]] += 1
        # followers[j][h] = các từ đã thấy theo sau context h (j-1 từ)
        self.followers = {j: defaultdict(list) for j in range(2, self.n + 1)}
        for j in range(2, self.n + 1):
            for gram in self.counts[j]:
                self.followers[j][gram[:-1]].append(gram[-1])
        self.total_tokens = sum(self.counts[1].values())
        if self.smoothing == "interp" and self.lambdas is None:
            self.lambdas = [1.0 / self.n] * self.n
        return self

    # --------------------------------------------------------------- probability
    def _mle(self, context, word, j):
        """P_MLE(word | j-1 từ cuối của context)."""
        h = tuple(context[len(context) - (j - 1):]) if j > 1 else ()
        ch = self.context_counts[j][h]
        return self.counts[j][h + (word,)] / ch if ch else 0.0

    def probability(self, context, word):
        """P(word | context); context là list/tuple token, chỉ dùng n-1 token cuối."""
        context = tuple(context)[-(self.n - 1):] if self.n > 1 else ()
        if self.smoothing == "mle":
            return self._mle(context, word, self.n)
        if self.smoothing in {"laplace", "addk"}:
            h = context
            return (self.counts[self.n][h + (word,)] + self.k) / (self.context_counts[self.n][h] + self.k * self.V)
        # interpolation: lambdas[0] cho unigram (add-one), lambdas[-1] cho bậc n
        p_uni = (self.counts[1][(word,)] + 1) / (self.total_tokens + self.V)
        total, weight = self.lambdas[0] * p_uni, self.lambdas[0]
        for j, lam in enumerate(self.lambdas[1:], start=2):
            if self.context_counts[j][context[len(context) - (j - 1):]]:   # chỉ dùng bậc có context đã gặp
                total += lam * self._mle(context, word, j)
                weight += lam
        return total / weight

    def _positions(self, sentence):
        p = pad(sentence, self.n)
        for i in range(self.n - 1, len(p)):
            yield p[i - self.n + 1:i], p[i]

    def sentence_probability(self, sentence):
        """Tích các xác suất có điều kiện (dễ underflow với câu dài)."""
        prob = 1.0
        for h, w in self._positions(sentence):
            prob *= self.probability(h, w)
        return prob

    def sentence_log_probability(self, sentence):
        """Σ log P(w_i | context); trả về -inf nếu có xác suất bằng 0."""
        total = 0.0
        for h, w in self._positions(sentence):
            p = self.probability(h, w)
            if p == 0.0:
                return float("-inf")
            total += math.log(p)
        return total

    def evaluate(self, sentences):
        """Trả về (perplexity, số token được dự đoán, số token có P = 0)."""
        log_sum, n_tok, n_zero = 0.0, 0, 0
        for s in sentences:
            for h, w in self._positions(s):
                p = self.probability(h, w)
                n_tok += 1
                if p == 0.0:
                    n_zero += 1
                else:
                    log_sum += math.log(p)
        ppl = float("inf") if n_zero else math.exp(-log_sum / n_tok)
        return ppl, n_tok, n_zero

    def perplexity(self, sentences):
        return self.evaluate(sentences)[0]

    def next_word_distribution(self, context, top_k=5):
        """Top-k từ kế tiếp và xác suất.

        Ứng viên = các từ đã theo sau context trong train (với interpolation: theo sau context ở mọi bậc,
        cộng 200 từ phổ biến nhất). Nếu context chưa gặp và không có backoff, mọi từ có cùng xác suất.
        """
        context = tuple(w if w in self.vocab or w == BOS else UNK for w in context)
        context = context[-(self.n - 1):] if self.n > 1 else ()
        h = (BOS,) * (self.n - 1 - len(context)) + context
        top_uni = [g[0] for g, _ in self.counts[1].most_common(200)]
        if self.n == 1:
            cands = top_uni
        elif self.smoothing == "interp":
            cands = set(top_uni)
            for j in range(2, self.n + 1):
                cands.update(self.followers[j].get(h[len(h) - (j - 1):], ()))
        else:
            cands = self.followers[self.n].get(h)
            if not cands:
                p = self.probability(h, EOS)
                return [(w, p) for w in sorted(self.vocab)[:top_k]]
        scored = sorted(((self.probability(h, w), w) for w in cands), key=lambda x: (-x[0], x[1]))[:top_k]
        return [(w, p) for p, w in scored]


# ----------------------------------------------------------------------------- unit tests
TOY = [s.split() for s in ["the cat eats fish", "the cat likes fish", "the dog eats meat"]]


def _toy_counts():
    """Đếm không có <s>, </s> — đúng như bài tính tay."""
    uni = Counter(w for s in TOY for w in s)
    bi = Counter((a, b) for s in TOY for a, b in zip(s, s[1:]))
    return uni, bi


def test_counts_and_unigram():
    uni, _ = _toy_counts()
    N = sum(uni.values())
    assert N == 12 and len(uni) == 7
    assert abs(uni["the"] / N - 0.25) < 1e-12 and abs(sum(c / N for c in uni.values()) - 1) < 1e-12
    assert count_ngrams(TOY, 2)[("<s>", "the")] == 3


def test_bigram_mle():
    lm = NGramLanguageModel(2, "mle").fit(TOY)
    assert abs(lm.probability(["the"], "cat") - 2 / 3) < 1e-12
    assert abs(lm.probability(["the"], "dog") - 1 / 3) < 1e-12
    assert abs(lm.probability(["cat"], "eats") - 0.5) < 1e-12
    assert lm.probability(["cat"], "meat") == 0.0
    # tổng xác suất các từ sau "the" bằng 1
    assert abs(sum(lm.probability(["the"], w) for w in lm.vocab) - 1) < 1e-12


def test_sentence_probability_and_log():
    lm = NGramLanguageModel(2, "mle").fit(TOY)
    s = "the cat eats fish".split()
    # P(the|<s>)=1, P(cat|the)=2/3, P(eats|cat)=1/2, P(fish|eats)=1/2, P(</s>|fish)=1
    assert abs(lm.sentence_probability(s) - 1 / 6) < 1e-12
    assert abs(lm.sentence_log_probability(s) - math.log(1 / 6)) < 1e-12
    assert lm.sentence_log_probability("the cat eats meat dog".split()) == float("-inf")


def test_laplace():
    lm = NGramLanguageModel(2, "laplace").fit(TOY)      # V = 7 từ + </s> = 8
    assert lm.V == 8
    assert abs(lm.probability(["cat"], "meat") - 1 / (2 + 8)) < 1e-12
    assert abs(sum(lm.probability(["cat"], w) for w in lm.vocab) - 1) < 1e-12


def test_interpolation_and_perplexity():
    lm = NGramLanguageModel(3, "interp", lambdas=[0.2, 0.3, 0.5]).fit(TOY)
    for ctx in (["the", "cat"], ["dog", "cat"], ["fish", "meat"]):     # context đã gặp / chưa gặp
        assert abs(sum(lm.probability(ctx, w) for w in lm.vocab) - 1) < 1e-12
    uni = NGramLanguageModel(1, "mle").fit(TOY)         # 15 token dự đoán (12 từ + 3 </s>)
    ppl, n, z = uni.evaluate(TOY)
    assert n == 15 and z == 0 and ppl > 1


def run_tests():
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"{len(tests)}/{len(tests)} tests passed")


if __name__ == "__main__":
    run_tests()
