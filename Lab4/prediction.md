# Prediction

Dataset: **IMDb Movie Reviews** (sentiment, binary: positive / negative; review dài, trung bình khoảng 230 từ, nhãn cân bằng 50/50).
Pipeline: raw text → TF-IDF hoặc Average Word2Vec → Logistic Regression.

Các dự đoán dưới đây dựa trên đặc tính của hai representation (TF-IDF: sparse, lexical, giữ thông tin từng term; Average Word2Vec: dense, semantic, mất thứ tự từ, mỗi từ một vector).

---

## Dự đoán chính: TF-IDF hay Average Word2Vec phù hợp hơn với IMDb?

**Prediction:** TF-IDF + Logistic Regression tốt hơn Average Word2Vec + Logistic Regression một cách rõ rệt.

**Reason:** Review dài chứa nhiều từ cảm xúc đặc trưng (*excellent, awful, waste…*). TF-IDF giữ nguyên từng từ nên Logistic Regression gán được trọng số riêng cho từng từ. Average Word2Vec lấy trung bình vector của hàng trăm từ, làm loãng tín hiệu của các từ cảm xúc giữa rất nhiều từ trung tính.

**Confidence:** Cao

---

## Prediction 1 — TF-IDF + Logistic Regression có khả năng hoạt động tốt không?

**Prediction:** Có, hoạt động rất tốt (accuracy khoảng 88–90%).

**Reason:** Cảm xúc phần lớn được quyết định bởi các từ có cực tính rõ, nên bài toán gần như tách được tuyến tính trong không gian từ vựng. Logistic Regression có đủ năng lực và ít overfit khi có regularization.

**Confidence:** Vừa

---

## Prediction 2 — Average Word2Vec có chắc chắn tốt hơn TF-IDF không?

**Prediction:** Không. Average Word2Vec chắc chắn kém hơn TF-IDF trên bài toán này.

**Reason:** Averaging làm mất thứ tự từ, mỗi từ chỉ có một vector cố định, và việc nén cả review thành 100 số làm mất quá nhiều thông tin so với TF-IDF giữ từng term.

**Confidence:** Cao

---

## Prediction 3 — Nếu tăng vocabulary size thì accuracy có chắc chắn tăng không?

**Prediction:** Không chắc chắn tăng mãi; accuracy tăng nhưng bão hòa dần.

**Reason:** Các từ thêm vào ngày càng hiếm, mang ít thông tin tổng quát nên lợi ích giảm dần. Không có chuyện tăng vocabulary mãi thì kết quả tốt mãi.

**Confidence:** Vừa

---

## Prediction 4 — Nếu bỏ stopwords thì performance có chắc chắn tăng không?

**Prediction:** Không chắc chắn tăng; có thể giảm.

**Reason:** Danh sách stopwords chuẩn chứa các từ như *not, no, never, very*, tức là chính các từ đảo hoặc nhấn mạnh cảm xúc. Ngoài ra ảnh hưởng của việc bỏ stopwords có thể phụ thuộc vào chủ đề/bài toán (ví dụ sentiment cần từ phủ định, còn topic classification có thể ít bị ảnh hưởng hơn).

**Confidence:** Thấp

---

## Prediction 5 — Accuracy và F1 có thể cho kết luận khác nhau không?

**Prediction:** Có thể. Trên IMDb thì accuracy và F1 gần như trùng nhau (vì dữ liệu cân bằng 50/50), nhưng khi dữ liệu mất cân bằng thì hai chỉ số có thể cho kết luận khác nhau.

**Reason:** IMDb cân bằng nên cả hai chỉ số dịch chuyển cùng nhau. Khi mất cân bằng, accuracy bị lớp đa số chi phối còn F1 thì không. Ví dụ ở Bài 4 trong `calculations.md`: model đoán mọi mẫu là negative (950 negative, 50 positive) có accuracy 95% nhưng F1 của lớp positive bằng 0.

**Confidence:** Cao
