# Prediction (ghi trước khi code)

Corpus: 20 000 abstract arXiv ML (khoảng 2.7 triệu token train), chia train/validation/test theo document.

---

## Prediction 1 — Khi chuyển unigram → bigram → trigram, vocabulary có tăng không?

**Prediction:** Không. Vocabulary (tập từ) giữ nguyên vì cùng corpus và cùng tokenizer; chỉ có số loại n-gram thay đổi.

**Reason:** Vocabulary là tập từ đơn, không phụ thuộc vào n. n chỉ quyết định độ dài của đơn vị được đếm.

**Confidence:** Cao

---

## Prediction 2 — Số lượng n-gram thay đổi thế nào?

**Prediction:** Tăng rất mạnh: số bigram khác nhau gấp khoảng 10–20 lần số unigram, trigram gấp 2–3 lần bigram; phần lớn trigram chỉ xuất hiện 1 lần.

**Reason:** Số tổ hợp có thể tăng theo V<sup>n</sup>, nhưng số n-gram thực tế bị chặn bởi số token (khoảng 2.7 triệu) → trigram gần như "mỗi vị trí một trigram mới".

**Confidence:** Vừa

---

## Prediction 3 — Mô hình nào gặp zero probability nhiều hơn?

**Prediction:** Trigram (MLE) gặp nhiều nhất, bigram ít hơn, unigram gần như không gặp (vì từ lạ đã được thay bằng `<unk>`).

**Reason:** Context càng dài thì tổ hợp càng cụ thể, xác suất tổ hợp đó đã xuất hiện trong train càng thấp.

**Confidence:** Cao

---

## Prediction 4 — Mô hình nào có perplexity thấp hơn trên training set?

**Prediction:** Trigram MLE có train PPL thấp nhất, rồi bigram, unigram cao nhất.

**Reason:** Trên train, MLE "nhớ" chính các n-gram đã đếm; context càng dài thì từ tiếp theo càng gần như được xác định (nhiều trigram chỉ có một từ theo sau duy nhất).

**Confidence:** Cao

---

## Prediction 5 — Corpus rất nhỏ thì trigram có chắc chắn tốt hơn bigram không?

**Prediction:** Không. Với corpus nhỏ, trigram có thể có validation/test PPL **cao hơn** bigram.

**Reason:** Phần lớn trigram trong validation chưa từng xuất hiện trong train; smoothing phải chia xác suất cho một không gian rất lớn → trigram overfit train. Kết quả có thể phụ thuộc vào cách smoothing.

**Confidence:** Vừa
