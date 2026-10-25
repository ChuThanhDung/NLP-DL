# Prediction (ghi trước khi chạy code)

Bối cảnh: attention tự cài bằng NumPy; thí nghiệm trên câu ngắn và trên IMDb (giống LAB 05: vocabulary 25 000, max_length 200).

---

## Prediction 1 — Một key có dot product lớn hơn các key khác thì attention weight của nó thế nào?

**Prediction:** Weight của key đó lớn nhất, và lớn hơn các key khác theo cấp số mũ chứ không theo tỉ lệ thuận: hai key chênh nhau Δs điểm thì tỉ lệ weight là e<sup>Δs</sup>.

**Reason:** softmax đơn điệu tăng và α<sub>i</sub> / α<sub>j</sub> = exp(s<sub>i</sub> − s<sub>j</sub>). Score lớn hơn 1 đơn vị đã cho weight gấp khoảng 2.7 lần.

**Confidence:** Cao

---

## Prediction 2 — Nếu mọi score bằng nhau (s = [1, 1, 1, 1])?

**Prediction:** Weight đều: 0.25 cho mỗi token. Context vector bằng trung bình cộng các value, tức attention trở thành average pooling (giống Average Word2Vec ở LAB 04).

**Reason:** exp(1) giống nhau ở mọi vị trí → mỗi phần tử bằng 1/4. Entropy đạt cực đại ln 4 ≈ 1.386.

**Confidence:** Cao

---

## Prediction 3 — Nếu một score lớn hơn rất nhiều (s = [1, 1, 10])?

**Prediction:** Gần như toàn bộ attention dồn vào token thứ ba: khoảng 0.9998; hai token còn lại mỗi token khoảng 0.0001. Context vector gần như chỉ là v<sub>3</sub> (attention gần "hard").

**Reason:** e<sup>10 − 1</sup> = e<sup>9</sup> ≈ 8 100 lần. Softmax bão hòa, gradient qua softmax gần 0 nên model khó điều chỉnh các weight nhỏ.

**Confidence:** Cao

---

## Prediction 4 — Sequence tăng từ 10 lên 1 000 token, chi phí attention thay đổi thế nào?

**Prediction:** Với self-attention đầy đủ, attention matrix tăng từ 10 × 10 = 100 lên 1 000 × 1 000 = 1 000 000 phần tử, tức gấp 10 000 lần (bậc hai). Thời gian và bộ nhớ tăng gần như vậy khi T đủ lớn. Nếu chỉ có một query (attention pooling) thì chi phí chỉ tăng tuyến tính (×100).

**Reason:** Mỗi token so với mọi token khác: T × T cặp, mỗi cặp một tích vô hướng d<sub>k</sub> chiều → O(T² d<sub>k</sub>).

**Confidence:** Cao

---

## Prediction 5 — Attention có luôn chọn đúng token mà con người cho là quan trọng nhất không?

**Prediction:** Không. Attention weights được học để giảm loss, không được huấn luyện để giống phán đoán của người. Model có thể dồn weight vào token tần suất cao, dấu hiệu tình cờ của dữ liệu, hoặc phân tán weight; và weight cao chưa chắc là token thực sự quyết định prediction.

**Reason:** Không có nhãn nào nói "token nào quan trọng"; chỉ có nhãn sentiment của cả review.

**Confidence:** Vừa

---

## Prediction 6 — Temperature T = 0.5 / 1 / 2 (cùng score vector)

**Prediction:** T = 0.5 cho phân phối tập trung nhất (max weight lớn nhất, entropy nhỏ nhất); T = 2 phẳng nhất. Chia cho √d<sub>k</sub> trong scaled dot-product chính là một temperature T = √d<sub>k</sub>.

**Reason:** softmax(s / T): T nhỏ phóng to khoảng cách giữa các score, T lớn thu nhỏ khoảng cách.

**Confidence:** Cao

---

## Prediction 7 — Experiment 4 trên IMDb: LSTM (final hidden state) vs Self-Attention (context vector)

**Prediction:** Self-attention classifier đạt accuracy ngang hoặc cao hơn LSTM một chút (LAB 05: LSTM 0.816), và train nhanh hơn trên GPU. Nhưng nó sẽ yếu với câu có phủ định hoặc phụ thuộc thứ tự, vì self-attention không có thông tin vị trí: đảo thứ tự từ không làm prediction thay đổi.

**Reason:** Sentiment trên IMDb phần lớn quyết định bởi các từ cảm xúc rải rác trong review. Attention nối trực tiếp mọi vị trí với context vector, không phải truyền qua 200 bước như LSTM. Các phép nhân ma trận chạy song song được. Tuy nhiên QK<sup>T</sup> chỉ phụ thuộc nội dung token, không phụ thuộc vị trí.

**Confidence:** Thấp (về accuracy), Cao (về việc không nhạy với thứ tự)
