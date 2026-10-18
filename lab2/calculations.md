# LAB 02 — Bài tập tính toán

Corpus dùng cho Bài 1–4:

```
the cat eats fish
the cat likes fish
the dog eats meat
```

(Bài 1–4 tính **không** có token `<s>`, `</s>`, đúng như đề.)

## Bài 1 — Unigram

1. **Vocabulary** = {the, cat, eats, fish, likes, dog, meat} → |V| = **7**.
2. **Tổng số token** N = 4 + 4 + 4 = **12**.
3. Đếm: C(the) = 3, C(cat) = 2, C(eats) = 2, C(fish) = 2, C(likes) = 1, C(dog) = 1, C(meat) = 1.
   P(w) = C(w) / N:
   - P(the) = 3/12 = **0.25**
   - P(cat) = 2/12 = 1/6 ≈ **0.1667**
   - P(fish) = 2/12 ≈ **0.1667**
   - P(dog) = 1/12 ≈ **0.0833**
   - (P(eats) = 2/12, P(likes) = 1/12, P(meat) = 1/12)
4. Kiểm tra: Σ P(w) = (3 + 2 + 2 + 2 + 1 + 1 + 1) / 12 = 12/12 = **1** ✓

## Bài 2 — Bigram

P(w<sub>i</sub> | w<sub>i−1</sub>) = C(w<sub>i−1</sub> w<sub>i</sub>) / C(w<sub>i−1</sub>)

Các bigram: the cat (2), cat eats (1), eats fish (1), cat likes (1), likes fish (1), the dog (1), dog eats (1), eats meat (1).

- P(cat | the) = C(the cat) / C(the) = 2/3 ≈ **0.6667**
- P(dog | the) = C(the dog) / C(the) = 1/3 ≈ **0.3333**
- P(eats | cat) = C(cat eats) / C(cat) = 1/2 = **0.5**
- P(likes | cat) = C(cat likes) / C(cat) = 1/2 = **0.5**

**Tại sao tổng xác suất các từ đứng sau *the* bằng 1?**
Σ<sub>w</sub> P(w | the) = Σ<sub>w</sub> C(the w) / C(the). Mỗi lần *the* xuất hiện (làm context) thì đúng một từ nào đó đứng sau, nên Σ<sub>w</sub> C(the w) = C(the) = 3 → tổng = 3/3 = 1 (ở đây 2/3 + 1/3 = 1). Điều này chỉ đúng khi vocabulary chứa mọi từ có thể đứng sau (kể cả `</s>` nếu *the* đứng cuối câu) và C(the) chỉ đếm những lần *the* đóng vai trò context.

## Bài 3 — Xác suất câu

P(fish | eats) = C(eats fish) / C(eats) = 1/2.

P(the cat eats fish) = P(the) · P(cat | the) · P(eats | cat) · P(fish | eats)
= 1/4 × 2/3 × 1/2 × 1/2 = 2/48 = **1/24 ≈ 0.0417**

**Thêm một từ vào câu, xác suất có thể tăng không?** **Không.** Mỗi thừa số mới là một xác suất ≤ 1 nên tích chỉ giảm hoặc giữ nguyên (khi thừa số mới bằng đúng 1). Ví dụ: P(the cat eats) = 1/4 × 2/3 × 1/2 = 1/12 > P(the cat eats fish) = 1/24.
→ Câu dài hơn gần như luôn có xác suất nhỏ hơn, nên **không thể dùng trực tiếp P(S) để so sánh các câu khác độ dài**; cần chuẩn hóa theo số token (log-prob trung bình / token, hoặc perplexity).
(Lưu ý: nếu model có `</s>`, so sánh hai câu *hoàn chỉnh* thì câu dài vẫn có thể có xác suất cao hơn khi câu ngắn kết thúc ở chỗ bất thường, ví dụ P(`</s>` | eats) = 0.)

## Bài 4 — Sentence ranking

S1 = the cat eats fish; S2 = the dog eats fish.

**Dự đoán trước:** S1 cao hơn, vì *cat* xuất hiện sau *the* nhiều hơn *dog* (2 lần so với 1 lần).

**Tính:**

- P(S1) = P(the) · P(cat | the) · P(eats | cat) · P(fish | eats) = 1/4 × 2/3 × 1/2 × 1/2 = **1/24**
- P(S2) = P(the) · P(dog | the) · P(eats | dog) · P(fish | eats) = 1/4 × 1/3 × 1/1 × 1/2 = **1/24**

→ **Hai câu có xác suất bằng nhau**, dự đoán ban đầu sai. *cat* được chọn sau *the* nhiều hơn (2/3 so với 1/3), nhưng sau *cat* xác suất bị chia đôi cho *eats* và *likes* (1/2), còn *dog* luôn được theo sau bởi *eats* (P = 1). Hai hiệu ứng triệt tiêu nhau: 2/3 × 1/2 = 1/3 × 1.

## Bài 9 — Suy luận trước khi smoothing

Corpus: *I like NLP*, *I like AI*, *I study NLP*.

1. C(study AI) = **0** (sau *study* chỉ có *NLP*).
2. P<sub>MLE</sub>(AI | study) = C(study AI) / C(study) = 0/1 = **0**.
3. Mọi câu chứa bigram *study AI* (vd. *I study AI*) có P(S) = 0 vì tích có một thừa số 0 → log P(S) = −∞, perplexity = ∞.
4. **Không.** Mô hình chỉ thấy 3 câu; *I study AI* là câu hoàn toàn tự nhiên. Xác suất 0 nghĩa là *chưa quan sát thấy* trong dữ liệu train, không phải *không thể xảy ra*. MLE đánh đồng hai điều này → cần smoothing.

## Bài 11 — Laplace smoothing

P<sub>Laplace</sub>(w | h) = (C(h, w) + 1) / (C(h) + V), với C(cat) = 10, V = 5.

- C(cat eats) = 0: P<sub>Laplace</sub>(eats | cat) = (0 + 1) / (10 + 5) = 1/15 ≈ **0.0667** (MLE = 0).
- C(cat eats) = 3: P<sub>Laplace</sub>(eats | cat) = (3 + 1) / (10 + 5) = 4/15 ≈ **0.2667** (MLE = 3/10 = 0.3).

**Smoothing thay đổi xác suất của các bigram khác thế nào?** Mẫu số tăng từ 10 lên 15 cho **mọi** bigram bắt đầu bằng *cat*. Bigram đã thấy bị **giảm** xác suất (vd. count 3: 0.3 → 0.267; count 7: 0.7 → 8/15 ≈ 0.533; count càng lớn bị trừ càng nhiều), còn bigram chưa thấy được **tăng** từ 0 lên 1/15. Tổng vẫn bằng 1: Σ(C + 1) / (C(cat) + V) = (10 + 5) / 15 = 1. Smoothing *lấy bớt* xác suất của các sự kiện đã thấy để *chia cho* các sự kiện chưa thấy, tức là thay đổi toàn bộ phân phối.

## Bài 18 — Perplexity

Cho P(w<sub>1</sub>) = 0.5, P(w<sub>2</sub> | w<sub>1</sub>) = 0.25, P(w<sub>3</sub> | w<sub>2</sub>) = 0.5; N = 3 token.

- P(S) = 0.5 × 0.25 × 0.5 = **0.0625**
- PP(S) = P(S)<sup>−1/N</sup> = 0.0625<sup>−1/3</sup> = (1/0.0625)<sup>1/3</sup> = 16<sup>1/3</sup> ≈ **2.52**

Khi P(w<sub>2</sub> | w<sub>1</sub>) = 0.1:

- P(S) = 0.5 × 0.1 × 0.5 = **0.025**
- PP(S) = 0.025<sup>−1/3</sup> = 40<sup>1/3</sup> ≈ **3.42**

**Vì sao một xác suất nhỏ làm perplexity thay đổi đáng kể?** PP là trung bình nhân của 1/P(w<sub>i</sub> | context): PP = exp(−(1/N) Σ log P). Chỉ một thừa số giảm 2.5 lần (0.25 → 0.1) mà PP tăng từ 2.52 lên 3.42 (×2.5<sup>1/3</sup> ≈ ×1.36). Trong log-space, −log p tăng không giới hạn khi p → 0 (−ln 0.25 = 1.39, −ln 0.1 = 2.30, −ln 0.001 = 6.91). Nếu một xác suất bằng 0 thì PP = ∞. Vì vậy một token bị model "bất ngờ" có thể chi phối perplexity, nhất là với chuỗi ngắn.
