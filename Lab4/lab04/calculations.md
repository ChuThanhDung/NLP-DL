# LAB 04 — Bài tập tính toán

*Bản gõ lại từ bài làm viết tay (`calculations.pdf`).*

## Bài 1 — Binary classification

Cho **x** = [1, 2, 0], **w** = [0.5, −0.2, 0.4], b = 0.1.

Tính z = **w**<sup>T</sup>**x** + b:

- **w**<sup>T</sup>**x** = 0.5 × 1 + (−0.2) × 2 + 0.4 × 0 = 0.5 + (−0.4) = 0.1
- z = 0.1 + 0.1 = **0.2**

Dùng sigmoid:

σ(z) = 1 / (1 + e<sup>−z</sup>) = 1 / (1 + e<sup>−0.2</sup>) ≈ 0.5498

Vậy P(y = 1 | x) = σ(z) ≈ **0.5498**.

## Bài 2 — Prediction threshold

P(y = 1 | x) = 0.72; τ<sub>1</sub> = 0.5; τ<sub>2</sub> = 0.8.

- τ<sub>1</sub> = 0.5: 0.72 > τ<sub>1</sub> nên dự đoán ŷ = **1**.
- τ<sub>2</sub> = 0.8: 0.72 < τ<sub>2</sub> nên dự đoán ŷ = **0**.

Ảnh hưởng của threshold:

- **Tăng τ:** dự đoán positive trở nên khó hơn. FP giảm (precision thường tăng), FN tăng (vì có nhiều mẫu có xác suất cao nhưng không vượt qua τ nên bị dự đoán là negative), recall giảm.
- **Giảm τ:** model dự đoán positive dễ hơn nên recall tăng, FP tăng, precision giảm.

⇒ τ tăng thì precision (thường) tăng, recall giảm; τ giảm thì recall tăng, precision giảm.

## Bài 3 — Confusion matrix

TP = 80, FN = 20, FP = 10, TN = 90.

- Precision = TP / (TP + FP) = 80 / (80 + 10) = 8/9 ≈ **0.889**
- Recall = TP / (TP + FN) = 80 / (80 + 20) = **0.8**
- F1 = 2PR / (P + R) = 2 × (8/9) × 0.8 / (0.8 + 8/9) = 1.4222 / 1.6889 ≈ **0.8421**

Trả lời (bài toán phát hiện spam, positive = spam):

- **FP:** email hợp lệ bị dự đoán là spam.
- **FN:** email spam bị dự đoán là hợp lệ.

## Bài 4 — Accuracy trap

Model dự đoán mọi mẫu là negative (950 negative, 50 positive).

Confusion matrix:

| | Dự đoán positive | Dự đoán negative |
|---|---|---|
| **Thực tế positive** | TP = 0 | FN = 50 |
| **Thực tế negative** | FP = 0 | TN = 950 |

- Accuracy = (TP + TN) / 1000 = 950 / 1000 = **95%**
- Recall = TP / (TP + FN) = 0 / (0 + 50) = **0**
- Precision = TP / (TP + FP) = 0 / (0 + 0) = 0/0, **không xác định** (quy ước bằng 0 vì model không dự đoán positive nào)
- F1 = 2PR / (P + R) = 2·0·0 / (0 + 0), cũng không xác định (quy ước bằng **0** vì recall = 0)

Với lớp negative: Precision = 950/1000 = **0.95**, Recall = **1**, F1 = 2 × 0.95 × 1 / (0.95 + 1) ≈ **0.974**.

| Metric | Giá trị | Nhận xét |
|---|---|---|
| Accuracy | 0.95 | Cao, nhưng chỉ vì lớp negative chiếm 95% dữ liệu |
| Recall (positive) | 0 | Không phát hiện được positive nào |
| Precision (positive) | 0/0 (quy ước 0) | Không xác định |
| F1 (positive) | 0 | Model vô dụng với lớp positive |

**Kết luận:** model **không tốt**. Accuracy cao chỉ vì dữ liệu mất cân bằng, còn recall và F1 của lớp positive đều bằng 0. Phải xem thêm recall, precision và F1 (theo từng lớp) chứ không chỉ accuracy.
