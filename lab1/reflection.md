# Reflection

Số liệu lấy từ `experiments.ipynb` / `results.csv`.

## 1. Prediction nào của em sai?

- **Prediction 4 sai:** em đoán pipeline B (bỏ dấu câu + stopwords) cho search tốt nhất. Thực tế không pipeline nào thắng ở mọi metric: A (tối giản) có P@5 cao nhất (0.625 so với 0.525 của B), C (subword) có MRR cao nhất (0.710 so với 0.642).
- **Prediction 5 sai một phần:** bỏ stopwords không thay đổi P@5 (0.525) nhưng làm MRR giảm (0.705 → 0.642).
- **Đúng:** vocabulary 54 335 term (đoán 40–60K; 51.8% term có df = 1), sparsity 99.80% (đoán > 99.5%), và top kết quả không nhất thiết gần nghĩa nhất.

## 2. Kết quả nào bất ngờ nhất?

Query *heart attack prediction* có P@5 = 0 ở **cả 5 pipeline**: *attack* trong corpus ML gần như luôn là *adversarial attack*. Bất ngờ thứ hai: pipeline A có vocabulary gấp 3 lần B (162 634 so với 53 489) nhưng P@5 lại cao hơn → vocabulary nhỏ hơn không có nghĩa là search tốt hơn.

## 3. Experiment nào cung cấp evidence mạnh nhất?

Preprocessing ablation, khi kết hợp với phân tích đóng góp từng term. Cùng một query *self-driving car*, chỉ đổi cách tokenize mà RR thay đổi 0.25 (A) → 0.5 (B) → 1.0 (C). Điều này chứng minh preprocessing là một quyết định modeling. Lưu ý: bộ đánh giá chỉ có 8 query nên chênh lệch nhỏ giữa các pipeline chưa đủ để kết luận chắc chắn.

## 4. Failure case quan trọng nhất

*heart attack* ↔ *myocardial infarction*: bài D05586 dự đoán đúng nhồi máu cơ tim nhưng xếp hạng 238; D14669 có cosine = 0. Hai cách nói cùng nghĩa không có token chung nên TF-IDF coi chúng là trực giao.

## 5. Nếu xây lại search engine

- Giữ cụm từ (bigram/phrase) để *heart attack*, *fake news* là một đơn vị.
- Dùng BM25 (bão hòa TF, chuẩn hóa độ dài) thay cho TF-IDF thô.
- Mở rộng query bằng từ đồng nghĩa hoặc word embedding (*car* → *vehicle*), hoặc kết hợp lexical + dense retrieval (hybrid).
- Mở rộng bộ đánh giá: nhiều query hơn, gán nhãn sâu hơn để R@K có ý nghĩa.

## 6. AI đã được sử dụng ở đâu?

Xem mục *AI assistance statement* trong `README.md`: AI hỗ trợ viết code thí nghiệm, vẽ biểu đồ và trình bày báo cáo; mọi số liệu được trích lại từ output của notebook.

## Part J — Từ failure đến representation tiếp theo

**Hypothesis:** cần một representation trong đó các từ xuất hiện trong ngữ cảnh tương tự (*heart attack* và *myocardial infarction* đều đi cùng *patients, ECG, cardiac, risk*) có vector gần nhau. Khi đó similarity được đo theo nghĩa thay vì theo token trùng → distributional representation / word embedding (LAB 03), rồi contextual embedding và Transformer.

## Learning check (trả lời ngắn)

1. **Sparse:** mỗi document chỉ dùng khoảng 107 trong 54 335 term, các chiều còn lại bằng 0.
2. **IDF thấp:** df ≈ N thì N/df ≈ 1 nên log ≈ 0; term như vậy không phân biệt được document (*the*: df = 29 879, idf = 0.004).
3. **IDF cao chưa chắc TF-IDF cao:** TF-IDF = TF × IDF; document không chứa term thì TF = 0 → TF-IDF = 0. Ví dụ 28 120 term có IDF cao nhất (= 10.31) nhưng mỗi term chỉ khác 0 ở đúng 1 document.
4. **Cosine:** chuẩn hóa độ dài, chỉ đo hướng (tỉ lệ các term), nên document dài không tự động có điểm cao.
5. **Preprocessing thay đổi kết quả:** thay đổi tập term và trọng số, vd. `self-driving` (A) khác `self` + `driving` (B).
6. **Failure quan sát được:** *heart attack* không tìm được *myocardial infarction*, lại khớp nhầm *adversarial attack*.
7. **Gợi ý representation tiếp theo:** dense, distributional (word embedding), rồi contextual embedding để xử lý polysemy (*attack*).
