# Reflection

Số liệu lấy từ `experiments.ipynb` / `results.csv`.

| Model | Train PPL | Validation PPL | Test PPL | % token P = 0 (test) |
|---|---|---|---|---|
| Unigram MLE | 909.8 | 830.8 | 827.7 | 0 |
| Bigram MLE | 86.3 | ∞ | ∞ | 15.5% |
| Bigram Laplace | 1 073.3 | 1 207.8 | 1 187.6 | 0 |
| Bigram add-k (k = 0.003) | 110.3 | 297.6 | 290.5 | 0 |
| Bigram interpolation | 99.4 | 226.0 | 221.1 | 0 |
| Trigram MLE | 10.9 | ∞ | ∞ | 50.6% |
| Trigram Laplace | 4 759.9 | 7 186.3 | 7 069.6 | 0 |
| Trigram add-k (k = 0.001) | 36.0 | 1 334.4 | 1 307.1 | 0 |
| Trigram interpolation | 23.1 | **191.4** | **187.3** | 0 |

## Đối chiếu prediction

- P1 đúng: vocabulary không đổi (29 314 từ).
- P2 đúng hướng nhưng em đánh giá thấp mức tăng: 29 316 → 651 846 (×22) → 1 665 210 (×2.6); 85.3% trigram types (68.0% bigram types) chỉ xuất hiện 1 lần.
- P3 đúng: 50.8% trigram token trong validation chưa thấy, so với 15.7% bigram.
- P4 đúng: train PPL giảm 910 → 86 → 10.9.
- P5 **đúng một phần**: với add-k, trigram tệ hơn bigram ở mọi kích thước corpus (10% train: 4 486 so với 724). Nhưng với interpolation, trigram tốt hơn bigram ngay cả ở 10% train (331 so với 365). Câu trả lời phụ thuộc vào smoothing nhiều hơn em nghĩ.

## Câu 1 — Tăng n, mô hình nhận thêm thông tin gì?

Thêm một từ context, nên phân biệt được các ngữ cảnh mà bigram gộp chung (vd. *in this* → *paper* với P = 0.31, trong khi chỉ *this* → *paper* thấp hơn nhiều). Train PPL giảm mạnh (86 → 10.9).

## Câu 2 — Tại sao tăng n làm sparsity tăng?

Số context có thể có tăng theo V<sup>n−1</sup> nhưng dữ liệu cố định (2.7 triệu token) → mỗi context có ít quan sát hơn. Unseen n-gram trên validation tăng 0% → 15.7% → 50.8%.

## Câu 3 — Tại sao smoothing cần thiết?

Không có smoothing, chỉ một n-gram chưa thấy cũng làm P(S) = 0 và PPL = ∞ (bigram/trigram MLE trên validation). Nhưng smoothing phải hợp lý: Laplace chia quá nhiều xác suất cho V = 29 316 từ chưa thấy, làm trigram tệ hơn cả unigram (7 186 so với 831). Interpolation, dựa vào bậc thấp hơn khi bậc cao thiếu dữ liệu, tốt nhất.

## Câu 4 — Perplexity đo điều gì?

Mức "bối rối" trung bình của model: PP = exp(−trung bình log P(w | context)), tương đương số lựa chọn "đều nhau" model phải đoán ở mỗi bước. Trigram interpolation có PP ≈ 187, tức tương đương chọn trong khoảng 187 từ, thay vì 828 như unigram. Chỉ so sánh được khi cùng dữ liệu, preprocessing, vocabulary.

## Câu 5 — PPL thấp hơn có luôn tạo văn bản tốt hơn với con người không?

Không. Trigram MLE có train PPL 10.9 vì thuộc lòng train, nhưng PPL trên dữ liệu mới là ∞. PPL đo khả năng dự đoán token trung bình, không đo tính mạch lạc toàn văn bản, tính đúng sự thật hay độ đa dạng. Một model ưu tiên từ phổ biến (*the, of*) có thể có PPL thấp nhưng sinh văn bản nhàm chán, lặp lại.

## Câu 6 — N-gram thất bại ở đâu so với con người?

Không có khái niệm từ tương tự (*method* và *approach* là hai từ hoàn toàn riêng); không dùng được context xa (*online retailers … such as → amazon*); không biết chủ đề hay ý nghĩa, chỉ đếm chuỗi bề mặt. Sentence ranking cho thấy điều này: *studies language models* (−21.0) được xếp trên *is useful for nlp* (−22.9), vì *studies language models* chứa các bigram phổ biến hơn trong corpus, chứ model không xét câu nào hợp nghĩa hơn.

## Câu 7 — Context 100 từ, trigram có dùng thông tin của 97 từ đầu không?

**Không.** Giả định Markov bậc 2: P(w<sub>100</sub> | w<sub>1</sub> … w<sub>99</sub>) ≈ P(w<sub>100</sub> | w<sub>98</sub>, w<sub>99</sub>). Mọi thông tin trước đó bị bỏ. Tăng n để nhìn xa hơn thì sparsity bùng nổ (mục Câu 2). Cần một cách nén toàn bộ lịch sử thành vector (neural LM, RNN) và chia sẻ thống kê giữa các từ tương tự (word embeddings).

## Context dài hơn có luôn tốt hơn không? (mục 23)

Không tự động. Train PPL luôn giảm khi tăng n, nhưng validation/test chỉ cải thiện nếu smoothing xử lý được sparsity: trigram add-k có val PPL 1 334, tệ hơn bigram add-k (298), trong khi trigram interpolation (191) tốt hơn bigram interpolation (226). Chênh lệch train–validation của trigram (23 → 191) lớn hơn bigram (99 → 226) là dấu hiệu overfitting. Tăng corpus từ 10% lên 100% làm val PPL của trigram interpolation giảm 331 → 191, tức context dài cần nhiều dữ liệu hơn.
