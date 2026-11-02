# Prediction (ghi trước khi chạy code)

Bối cảnh: Transformer Encoder tự ghép (LAB 07), classification IMDb giống LAB 05 (vocabulary 25 000, train/val/test như LAB 05). Các prediction này không được sửa sau khi xem kết quả.

---

## Prediction 1 — Transformer có xử lý các token song song hơn RNN không?

**Prediction:** Có. Mọi token của một câu được xử lý cùng lúc bằng các phép nhân ma trận (QKᵀ, AV, FFN), nên một epoch nhanh hơn LSTM khi sequence cỡ vài trăm token.

**Reason:** h<sub>t</sub> của RNN phụ thuộc h<sub>t−1</sub> nên T bước phải chạy nối tiếp; self-attention và FFN không có phụ thuộc giữa các vị trí trong cùng một lớp.

**Confidence:** Cao

---

## Prediction 2 — Nếu bỏ positional encoding, Transformer có còn biết thứ tự token không?

**Prediction:** Không. Mọi phép tính (Q, K, V, softmax, FFN, LayerNorm) đều không dùng chỉ số vị trí, nên đổi thứ tự token chỉ làm output đổi thứ tự theo, không đổi giá trị.

**Reason:** Self-attention chỉ so sánh các cặp vector; token "dog" cho cùng một query/key/value ở bất kỳ vị trí nào. Output của mỗi token là tổng có trọng số trên cùng một tập key.

**Confidence:** Cao

---

## Prediction 3 — Hai câu cùng tập token nhưng khác thứ tự, model không có positional information có phân biệt được không?

**Prediction:** Không. "dog bites man" và "man bites dog" cho cùng một tập representation; sau mean pooling hai vector câu bằng nhau (chênh lệch cỡ sai số số học ~10<sup>−7</sup>), nên classifier ra cùng một xác suất.

**Reason:** Hệ quả của Prediction 2: output của câu đảo = output câu gốc với các hàng bị hoán vị; mean pooling không phụ thuộc thứ tự hàng. Thêm PE thì mỗi token bị cộng một vector khác nhau theo vị trí, hai câu mới khác nhau.

**Confidence:** Cao

---

## Prediction 4 — Hai attention head có nhất thiết học cùng một attention pattern không?

**Prediction:** Không nhất thiết. Mỗi head có W<sub>Q</sub>, W<sub>K</sub> riêng nên có thể cho pattern khác nhau, nhưng em đoán sau train trên IMDb một số head vẫn khá giống nhau (redundant) vì task chỉ là classification sentiment và model rất nhỏ.

**Reason:** Khởi tạo ngẫu nhiên khác nhau cho từng head làm pattern khác nhau lúc đầu; không có loss nào ép các head phải khác nhau.

**Confidence:** Trung bình

---

## Prediction 5 — Tăng số attention head có chắc chắn làm accuracy tăng không?

**Prediction:** Không. Với d<sub>model</sub> cố định, tăng head chỉ làm d<sub>k</sub> nhỏ đi (128 → 16) và số tham số không đổi. Em đoán accuracy gần như không đổi giữa 1, 2, 4, 8 head (chênh lệch trong khoảng ±0.01, cùng cỡ nhiễu giữa các seed), có thể giảm nhẹ ở 8 head.

**Reason:** Tổng số tham số của W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub>, W<sub>O</sub> là 4 d<sub>model</sub><sup>2</sup> không phụ thuộc số head; mỗi head chỉ có không gian d<sub>k</sub> chiều để so khớp q và k.

**Confidence:** Trung bình

---

## Prediction 6 — Transformer có luôn nhanh hơn RNN trên mọi sequence length không?

**Prediction:** Không. Với sequence ngắn (32–64 token) LSTM và Transformer gần như ngang nhau về thời gian vì cả hai đều rất nhỏ. Với sequence rất dài, attention tốn O(T<sup>2</sup>) bộ nhớ và thời gian nên lợi thế song song bị mất dần, bộ nhớ của Transformer vượt LSTM.

**Reason:** Attention matrix T × T mỗi head: T gấp đôi thì số phần tử gấp 4. LSTM tốn O(T) nhưng tuần tự.

**Confidence:** Trung bình
