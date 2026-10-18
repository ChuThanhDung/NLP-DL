# Prediction (ghi trước khi chạy model)

Corpus dự kiến: text8 (17 triệu token Wikipedia) cho Word2Vec; 2 triệu token đầu cho co-occurrence.

---

## Prediction 1 — Những từ nào gần nhau nhất? (doctor, physician, hospital, banana, car)

**Prediction:** doctor–physician gần nhau nhất, sau đó là doctor–hospital và physician–hospital. banana và car xa ba từ y tế; banana–car cũng xa nhau.

**Reason:** doctor và physician thay thế được cho nhau trong cùng câu (*the ___ treated the patient*) nên có context gần như giống nhau. hospital xuất hiện *cùng* doctor (quan hệ chủ đề) chứ không thay thế được doctor, nên gần nhưng kém hơn. banana và car thuộc chủ đề khác hẳn.

**Confidence:** Cao

---

## Prediction 2 — Context window tăng từ 2 lên 5, similarity có thay đổi không?

**Prediction:** Có. Window nhỏ ưu tiên từ **cùng chức năng** (thay thế được: doctor–physician, cat–dog); window lớn ưu tiên từ **cùng chủ đề** (doctor–hospital, doctor–patient). Vậy khi tăng window, doctor–hospital tăng còn doctor–physician và cat–dog có thể giảm.

**Reason:** Với window ±2, context chủ yếu là từ ngữ pháp sát bên (mạo từ, động từ). Với window ±5, context gồm cả các từ nội dung cùng đoạn văn, nên hai từ thường đi chung một chủ đề cũng có context giống nhau.

**Confidence:** Vừa

---

## Prediction 3 — Dimension 50 → 100 → 300, chất lượng có chắc chắn tăng không?

**Prediction:** Không chắc chắn. 50 → 100 có thể tăng; 100 → 300 tăng ít hoặc không tăng trên similarity/analogy với corpus 17M token, trong khi thời gian train và kích thước model tăng khoảng 3 lần.

**Reason:** Nhiều chiều hơn cho nhiều capacity hơn, nhưng cần nhiều dữ liệu hơn để học hết các tham số. Corpus không đủ lớn thì các chiều thêm vào chủ yếu học nhiễu. Bài toán downstream có thể vẫn hưởng lợi vì classifier có nhiều đặc trưng hơn.

**Confidence:** Vừa

---

## Prediction 4 — doctor và physician có chắc chắn gần nhau nếu corpus chỉ có 100 câu?

**Prediction:** Không. Với 100 câu, mỗi từ chỉ xuất hiện vài lần; vector gần như phụ thuộc vào khởi tạo ngẫu nhiên và vài context tình cờ, nên cosine có thể cao hay thấp tùy lần chạy và nearest neighbors không có ý nghĩa.

**Reason:** Distributional hypothesis là một phát biểu thống kê: cần nhiều quan sát context thì mới ước lượng được phân bố context đáng tin.

**Confidence:** Cao
