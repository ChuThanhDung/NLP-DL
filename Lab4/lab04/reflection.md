# Reflection — từ representation đến neural NLP

Số liệu lấy từ `experiment.ipynb` / `results.csv` (IMDb, test 25 000 review).

| Representation | Classifier | Accuracy | F1 |
|---|---|---|---|
| TF-IDF (unigram) | Logistic Regression | 0.8814 | 0.8810 |
| Average Word2Vec | Logistic Regression | 0.8426 | 0.8424 |
| TF-IDF (unigram) | Linear SVM | 0.8886 | 0.8889 |
| TF-IDF (uni + bigram) | Logistic Regression | 0.9022 | 0.9023 |

## 1. Vì sao TF-IDF (0.8814) khác Average Word2Vec (0.8426)?

- TF-IDF giữ nguyên từng từ nên Logistic Regression gán được trọng số riêng cho từ cảm xúc; Average Word2Vec lấy trung bình hàng trăm từ nên tín hiệu bị loãng.
- Word2Vec được tự train trên corpus nhỏ (khoảng 20 nghìn review) nên embedding chưa tốt.
- Word2Vec học theo ngữ cảnh nên *good* và *bad* có vector gần nhau (cosine 0.72 trong thí nghiệm), làm mất tính cực của từ.

## 2. Average embedding có hiểu "not" không? (Exp 4)

Với "this movie is good" → "this movie is not good": cosine giữa hai vector là 0.948; P(positive) của Word2Vec giảm từ 0.991 xuống 0.851, TF-IDF unigram từ 0.934 xuống 0.503, TF-IDF bigram từ 0.997 xuống 0.003.

→ **Hiểu một phần:** xác suất có giảm (0.991 → 0.851) nên "not" có tác động, nhưng không đảo được ý của câu.

## 3. Bigram giúp gì và tốn gì? (Exp 5)

| Features | Vocabulary | F1 | Training time |
|---|---|---|---|
| Unigram | 40 790 | 0.8810 | 3.1 s |
| Uni + Bigram | 364 040 | 0.9023 | 9.3 s |

Bigram bắt được cụm đảo cực tính như *not good, not bad*; nhưng số cặp từ khác nhau rất nhiều nên vocabulary phình to, nhiều bigram hiếm dễ gây overfit.

## 4. Top features có ý nghĩa không?

Top positive: *great, excellent, perfect, wonderful…*; top negative: *worst, waste, bad, awful, boring…*.
Phần lớn có nghĩa, nhưng vài từ (*today, save*) là tương quan tình cờ của tập train.

## 5. So prediction ban đầu với kết quả thật

Các dự đoán đều đúng hướng: TF-IDF tốt hơn Average Word2Vec (0.8814 so với 0.8426); tăng vocabulary thì F1 tăng nhưng bão hòa dần (0.861 → 0.877 → 0.889 → 0.892); bỏ stopwords làm F1 giảm (0.892 → 0.884); accuracy ≈ F1 trên IMDb vì dữ liệu cân bằng. Lý do là hiểu đúng đặc tính của hai representation (sparse/lexical và dense/mất thứ tự).

## 6. Giới hạn dẫn tới hướng nào ở LAB 05 trở đi?

- Cần mô hình đọc theo thứ tự (RNN/LSTM/GRU) để "not" ảnh hưởng được tới "good" ở sau.
- Cần dữ liệu lớn hơn hoặc embedding pretrained.
- Cần contextual embedding (một từ có nhiều vector tùy ngữ cảnh), rồi attention/Transformer.

## 7. Bài học rút ra

- Không chỉ nhìn accuracy; phải xem confusion matrix và precision/recall/F1.
- Representation quan trọng hơn lựa chọn classifier (Linear SVM 0.8886, Logistic Regression 0.8814, Average Word2Vec 0.8426).
- Phải phân tích lỗi mới biết representation hiện tại thiếu gì.
