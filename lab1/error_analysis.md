# Error analysis — TF-IDF search (pipeline B) trên 30K arXiv abstracts

Số liệu lấy từ Part I của `experiments.ipynb`. "Đóng góp" của một term = q<sub>t</sub> · d<sub>t</sub> (vector đã chuẩn hóa L2 nên cosine = tổng các đóng góp). ✓ = relevant theo `qrels.json`.

| Query | P@5 | R@5 | RR | Đánh giá |
|---|---|---|---|---|
| speech emotion recognition | 1.0 | 0.263 | 1.0 | tốt |
| fake news detection | 1.0 | 0.263 | 1.0 | tốt |
| heart attack prediction | 0.0 | 0.000 | 0.056 (relevant đầu tiên ở rank 18) | kém |
| self-driving car | 0.4 | 0.087 | 0.5 | kém |

## 1. Query tốt: "speech emotion recognition"

**Expected:** 19 bài về nhận dạng cảm xúc từ giọng nói.
**Retrieved (top-5):** D23307 ✓, D01812 ✓, D16918 ✓, D16128 ✓, D15412 ✓.

1. *Vì sao đứng đầu?* D23307 dùng *emotion* rất nhiều lần, đóng góp emotion = 0.412, speech = 0.182.
2. *Từ đóng góp nhiều:* **emotion** (IDF 6.26, hiếm) là đóng góp lớn nhất ở cả 5 document; *speech* (IDF 4.56) đứng thứ hai; *recognition* (IDF 4.00, phổ biến hơn) đóng góp ít, D16128 thậm chí không chứa *recognition* vẫn ở top-5.
3. *Lexical overlap:* có, cao — cả 3 term của query đều là thuật ngữ chuẩn mà các bài cùng chủ đề đều dùng.
4. *Bỏ sót:* 14/19 relevant ngoài top-5, nhưng vẫn xếp gần (D21467 rank 6, D12221 rank 8, D14808 rank 14). Đây là giới hạn của R@5 khi có 19 relevant (R@5 tối đa = 5/19 = 0.263), không phải lỗi ranking.
5. *Nguyên nhân thành công:* chủ đề có thuật ngữ cố định và IDF cao → lexical matching đủ tốt.

## 2. Query tốt: "fake news detection"

**Retrieved (top-5):** D22060 ✓, D06367 ✓, D21211 ✓, D27034 ✓, D07008 ✓ (score 0.71 – 0.86, cao nhất trong 8 query).

1. Các document đầu đều có cụm *fake news* trong tiêu đề và lặp lại nhiều lần trong abstract.
2. **fake** (IDF 6.56) và **news** (IDF 6.22) chiếm gần hết điểm (vd. D22060: fake = 0.429, news = 0.385); *detection* (IDF 3.44, rất phổ biến trong ML) chỉ đóng góp khoảng 0.05.
3. Lexical overlap rất cao.
4. Bỏ sót: các bài dùng *misinformation / disinformation / rumor* thay cho *fake news* (vd. D05379 "Detection of Misinformation via Tensor Embeddings") bị xếp thấp vì không có từ chung với query ngoài *detection*.
5. Thành công nhờ cụm từ chuẩn; phần bỏ sót là do **lexical matching** (từ đồng nghĩa).

## 3. Query kém: "heart attack prediction"

**Expected:** 9 bài dự đoán/chẩn đoán nhồi máu cơ tim, bệnh tim mạch.
**Retrieved (top-5):** D01939 ✗ (âm thanh tim), D08336 ✗ (multi-model **attack**), D13202 ✗ (multi-stage **attack**), D29188 ✗ (GraphAttacker), D18256 ✗ (heart rate). P@5 = 0.

1. *Vì sao đứng đầu?* Mỗi document chỉ khớp **một** term của query: D01939 chỉ có heart = 0.427, D08336 chỉ có attack = 0.353.
2. *Từ đóng góp:* **attack** (df = 683, IDF 4.78) — trong corpus ML, *attack* gần như luôn mang nghĩa *adversarial attack* (462 document chứa cụm này). *heart* (df = 173) và *prediction* (df = 3 556, IDF thấp, đóng góp ≈ 0).
3. *Lexical overlap:* có overlap với từng từ riêng lẻ, nhưng sai nghĩa: TF-IDF không biết *heart attack* là một cụm.
4. *Bỏ sót:* cả 9 relevant. Relevant đầu tiên (D01363 "Ischemic Heart Disease") ở rank 18; không relevant nào chứa từ *attack*.
5. *Nguyên nhân:* **lexical matching + vocabulary** (unigram tách cụm từ, polysemy của *attack*) và **IDF** (*attack* có IDF cao hơn *prediction* nên chi phối điểm).

## 4. Query kém: "self-driving car"

**Retrieved (top-5):** D18459 ✗ (car-following của người lái), D11088 ✓, D00431 ✗ (game đua xe Gran Turismo), D04852 ✓, D06526 ✗ (car-following model). P@5 = 0.4.

1. D18459 đứng đầu nhờ *car* (0.234) + *driving* (0.121); D00431 và D06526 chỉ khớp **car** (0.351, 0.339).
2. **car** có IDF cao nhất trong query (6.45) nên một document lặp *car* nhiều lần (game đua xe, mô hình bám xe) được điểm cao dù không liên quan đến tự lái.
3. Có lexical overlap nhưng ở nghĩa rộng của *car*.
4. Bỏ sót 21/23 relevant: chỉ 6/23 relevant chứa từ *car(s)*, trong khi 16/23 dùng *vehicle* và 9/23 dùng *autonomous driving*.
5. *Nguyên nhân chính:* **lexical matching / vocabulary** — *car* ≠ *vehicle*, và *car* lại đa nghĩa (xe đua, xe trong giao thông). **Preprocessing** làm kết quả thay đổi rõ rệt:
   - Pipeline A giữ `self-driving` thành một token → các bài viết *autonomous driving* không khớp được với *driving*; top-5 bị chiếm bởi bài chỉ có *car* (RR = 0.25).
   - Pipeline B/C tách thành `self` + `driving` → *driving* khớp được các bài *autonomous driving* (B: RR = 0.5; C và B−sw: RR = 1.0).
   - Phần `self` sau khi tách gần như vô nghĩa: *self* có trong 1 404 document (phần lớn là *self-supervised*, *self-attention*), đóng góp chỉ khoảng 0.01–0.03.

## 5. Failure case quan trọng nhất: *heart attack* ↔ *myocardial infarction*

Corpus có 8 document chứa *myocardial infarction* nhưng chỉ 2 document chứa *heart attack*.

| doc | rank với query "heart attack prediction" | chứa từ *heart*? |
|---|---|---|
| D23733 (Diagnosis of Heart Disease) | 35 | có |
| D05586 (HRV as a predictor of myocardial infarction) | 238 | có |
| D15197 (ECG Heartbeat Classification) | 390 | có |
| D06268 (coronary arteries, stenosis) | 4 220 | không |
| D14669 (CNN pipeline … Myocardial Infarction) | 16 808 (score 0) | không |
| 3 document còn lại | > 17 000 (score 0) | không |

D05586 dự đoán chính xác *myocardial infarction* (= heart attack) nhưng ở rank 238; D14669 có score **bằng 0**. Cosine giữa query và câu "myocardial infarction prediction" chỉ là 0.09 (chỉ chung *prediction*). Hai biểu thức cùng nghĩa nhưng **không có token chung** nên trong không gian TF-IDF chúng gần như trực giao; IDF cao của *myocardial* (8.67) và *infarction* (9.11) không giúp được gì vì query không chứa chúng.

**Kết luận:** lỗi không đến từ cài đặt hay tham số mà từ bản chất representation: mỗi chiều là một từ, không có khái niệm hai chiều khác nhau "gần nghĩa". Cần representation trong đó *heart attack* và *myocardial infarction* có vector gần nhau (word/sentence embedding).
