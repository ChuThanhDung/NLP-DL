# Prediction (ghi trước khi train bất kỳ model nào)

Dataset: IMDb (review dài, median khoảng 174 token, cân bằng 50/50). Baseline LAB 04: TF-IDF + LR có accuracy khoảng 0.88.

---

## Prediction 1 — TF-IDF + LR hay RNN + Embedding xử lý word order trực tiếp hơn?

**Prediction:** RNN + Embedding.

**Reason:** TF-IDF unigram là bag-of-words: "dog bites man" và "man bites dog" có cùng vector, thứ tự bị bỏ hoàn toàn. RNN đọc tuần tự và cập nhật h<sub>t</sub> = f(x<sub>t</sub>, h<sub>t−1</sub>), nên cùng một từ nhưng đứng sau context khác sẽ cho hidden state khác. Tuy nhiên "xử lý được thứ tự" không có nghĩa là accuracy cao hơn: em dự đoán TF-IDF + LR vẫn có thể ngang hoặc tốt hơn RNN thuần trên IMDb.

**Confidence:** Cao (về khả năng), Thấp (về accuracy)

---

## Prediction 2 — RNN hay LSTM xử lý dependency dài tốt hơn?

**Prediction:** LSTM.

**Reason:** Trong RNN, thông tin và gradient phải đi qua tích của nhiều Jacobian W<sub>hh</sub> · diag(tanh′); khi các thừa số nhỏ hơn 1 thì tích → 0 (vanishing). LSTM có cell state cập nhật bằng **phép cộng** c<sub>t</sub> = f<sub>t</sub> c<sub>t−1</sub> + i<sub>t</sub> c̃<sub>t</sub>; khi forget gate ≈ 1 thì ∂c<sub>t</sub>/∂c<sub>t−1</sub> ≈ 1, nên thông tin và gradient đi xa được. Các gate học được *khi nào* giữ và khi nào xóa.

**Confidence:** Cao

---

## Prediction 3 — Sequence length 50 → 200, training RNN thay đổi thế nào?

**Prediction:** Thời gian train tăng gần tuyến tính (khoảng ×4) vì phải tính tuần tự 200 bước. RNN thuần khó học hơn: token đầu review gần như không nhận được gradient. Accuracy có thể không tăng, thậm chí giảm, vì h<sub>T</sub> chủ yếu nhớ phần cuối.

**Reason:** Chi phí O(T) và không song song được theo thời gian; vanishing gradient nặng hơn khi T lớn.

**Confidence:** Vừa

---

## Prediction 4 — Truncate về 30 tokens có ảnh hưởng không?

**Prediction:** Có, ảnh hưởng rõ. Với max_length 30 chỉ giữ khoảng 13% token (median review 174 token); phần đầu review thường là giới thiệu cốt truyện, còn kết luận ("overall…", "8/10", "don't waste your time") nằm ở cuối và bị cắt mất.

**Reason:** Thông tin sentiment phân bố khắp review và đậm đặc ở phần kết luận.

**Confidence:** Cao

---

## Prediction 5 — LSTM có chắc chắn accuracy cao hơn RNN trên mọi dataset không?

**Prediction:** Không. Trên câu ngắn, dữ liệu ít, hoặc bài toán chỉ cần từ khóa cục bộ, RNN (ít tham số hơn) có thể ngang hoặc tốt hơn; LSTM có gấp 4 lần tham số recurrent nên dễ overfit hơn khi dữ liệu nhỏ. Trên IMDb (review dài), em dự đoán LSTM/GRU tốt hơn RNN rõ rệt.

**Reason:** Lợi thế của LSTM đến từ dependency dài; khi bài toán không cần dependency dài thì lợi thế đó không còn.

**Confidence:** Vừa
