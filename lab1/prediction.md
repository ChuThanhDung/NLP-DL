# Prediction (ghi trước khi chạy experiment)

Corpus: 30 000 abstract bài báo Machine Learning trên arXiv (title + abstract, trung bình khoảng 175 từ/document).
Các dự đoán dưới đây được ghi trước khi mở corpus; kết quả thực tế được đối chiếu trong `reflection.md`.

---

## Prediction 1 — Vocabulary

**Prediction:** Khoảng 40 000 – 60 000 unique terms (với tokenizer chuẩn, lowercase).

**Reason:** Mỗi abstract khoảng 175 từ → tổng khoảng 5 triệu token. Theo luật Zipf/Heaps, vocabulary tăng chậm hơn số token nhưng không bão hòa; abstract khoa học có nhiều thuật ngữ, tên model (BERT, ResNet…), ký hiệu toán và con số nên vocabulary lớn hơn văn bản thông thường. Khoảng một nửa vocabulary sẽ là term chỉ xuất hiện ở 1 document.

**Confidence:** Vừa

---

## Prediction 2 — Sparsity

**Prediction:** TF-IDF matrix rất sparse: hơn 99.5% phần tử bằng 0.

**Reason:** Một document chỉ dùng khoảng 100 term khác nhau trong khi vector có V ≈ 50 000 chiều → tỉ lệ khác 0 khoảng 100/50 000 = 0.2%, tức khoảng 99.8% là 0.

**Confidence:** Cao

---

## Prediction 3 — Search

**Prediction:** Không. Các document đứng đầu là những document **trùng từ** với query nhiều nhất (có trọng số IDF), không nhất thiết là gần nghĩa nhất.

**Reason:** TF-IDF + cosine chỉ so khớp lexical. Document dùng từ đồng nghĩa (vd. *autonomous vehicle* thay vì *self-driving car*) sẽ bị xếp thấp; document chứa từ của query nhưng với nghĩa khác (vd. *attack* trong *adversarial attack*) có thể bị xếp cao.

**Confidence:** Cao

---

## Prediction 4 — Preprocessing (Part F)

**Prediction:** Pipeline B (lowercase + bỏ dấu câu + bỏ stopwords) cho search tốt nhất; pipeline A (tối giản) kém nhất; vocabulary của B nhỏ hơn A rất nhiều.

**Reason:** A giữ dấu câu dính vào từ nên `model,` và `model` là hai term khác nhau → vocabulary phình to, khớp query kém. B gộp các biến thể và bỏ các từ không mang nội dung. C (subword) có vocabulary nhỏ nhất và OOV gần 0, nhưng các mảnh từ như `##ing` ít nghĩa nên có thể gây khớp nhiễu.

**Confidence:** Thấp

---

## Prediction 5 — Stopword removal

**Prediction:** Bỏ stopwords chỉ ảnh hưởng nhỏ đến search, vì stopwords vốn đã có IDF rất thấp (gần 0) trong TF-IDF.

**Reason:** IDF đã tự động giảm trọng số của *the, of, and* (xuất hiện ở gần như mọi document), nên bỏ chúng đi gần như chỉ làm giảm số token và kích thước index.

**Confidence:** Vừa
