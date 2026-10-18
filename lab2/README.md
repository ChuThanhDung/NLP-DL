# LAB 02 — Language Models (N-gram, Smoothing, Perplexity)

Bài toán: tự xây dựng unigram/bigram/trigram language model trên **20 000 abstract arXiv ML** (cùng nguồn với LAB 01), so sánh smoothing, đánh giá bằng perplexity, ứng dụng vào next-word prediction và sentence ranking.
Pipeline: `text → sentence split → tokenize → <unk> → count n-gram → P(w | context) → log P → perplexity → next word / ranking`.

## Cấu trúc thư mục

| File | Nội dung |
|---|---|
| `report.pdf` | **Bản tổng hợp**: README, calculations (bản gõ + bản viết tay), prediction, experiments.ipynb, error_analysis, reflection |
| `calculations.md`, `calculations.pdf` | Bài 1–4, 9, 11, 18 (unigram, bigram, xác suất câu, ranking, zero probability, Laplace, perplexity). `calculations.pdf` là bản viết tay (scan) |
| `prediction.md` | 5 dự đoán trước experiment |
| `ngram_lm.py` | `NGramLanguageModel` tự cài đặt (`fit`, `probability`, `sentence_probability`, `sentence_log_probability`, `perplexity`, `next_word_distribution`), MLE / Laplace / add-k / interpolation + 5 unit tests |
| `experiments.ipynb` | Corpus statistics, MLE vs smoothing, perplexity uni/bi/tri, corpus size, next-word prediction, sentence ranking, error analysis (đã chạy, có output) |
| `results.csv` | Perplexity (train/val/test), % token P = 0, corpus-size PPL, next-word accuracy, các ví dụ next-word |
| `error_analysis.md` | 2 dự đoán đúng, 3 dự đoán sai, phân loại nguyên nhân |
| `reflection.md` | Đối chiếu prediction, trả lời 7 câu reflection, thí nghiệm context length |
| `ngram_frequency.png`, `perplexity.png` | Phân bố tần suất n-gram; PPL theo n và theo kích thước corpus |

## Thiết lập

- **Corpus:** 20 000 abstract (`CShorten/ML-ArXiv-Papers`, seed 42), chia **theo document** train 16 000 / validation 2 000 / test 2 000 → 114 562 / 14 253 / 14 373 câu, 2.68M / 333K / 334K token.
- **Preprocessing:** tách câu theo `. ! ?`, lowercase, công thức `$…$` và số → `<num>`, bỏ dấu câu; vocabulary = từ có count ≥ 2 trong train (29 314 từ), từ còn lại → `<unk>` (OOV validation 2.0%).
- **Boundary:** mỗi câu được thêm (n−1) token `<s>` và một `</s>`; token được dự đoán = từ + `</s>`, giống nhau cho mọi n.
- **Chọn siêu tham số trên validation:** k của add-k (bigram 0.003, trigram 0.001), λ của interpolation (bigram (0.2, 0.8), trigram (0.2, 0.5, 0.3)). Test chỉ dùng để báo cáo.

## Kết quả chính

| Model | Train PPL | Val PPL | Test PPL |
|---|---|---|---|
| Unigram MLE | 909.8 | 830.8 | 827.7 |
| Bigram MLE | 86.3 | ∞ (15.7% token P = 0) | ∞ |
| Bigram Laplace | 1 073.3 | 1 207.8 | 1 187.6 |
| Bigram interpolation | 99.4 | 226.0 | 221.1 |
| Trigram MLE | 10.9 | ∞ (50.8% token P = 0) | ∞ |
| Trigram Laplace | 4 759.9 | 7 186.3 | 7 069.6 |
| Trigram add-k | 36.0 | 1 334.4 | 1 307.1 |
| **Trigram interpolation** | 23.1 | **191.4** | **187.3** |

Các phát hiện chính:

1. **Sparsity:** số n-gram types 29K → 652K → 1.67M; 85% trigram types chỉ xuất hiện 1 lần; 51% trigram token trong validation chưa từng thấy.
2. **MLE** cho train PPL thấp nhất nhưng PPL = ∞ trên dữ liệu mới. **Laplace** quá thô (trigram Laplace còn tệ hơn unigram). **Interpolation** tốt nhất.
3. **Context dài hơn không tự động tốt hơn:** với add-k, trigram tệ hơn bigram; với interpolation, trigram tốt hơn. Corpus 10% → 100% làm val PPL của trigram interpolation giảm 331 → 191.
4. **Next-word prediction** (3 000 vị trí test): top-1 / top-5 accuracy = 0.050 / 0.137 (unigram), 0.164 / 0.347 (bigram), 0.195 / 0.374 (trigram interpolation).
5. **Log probability:** tích xác suất của chuỗi 291 token underflow thành 0.0; log P = −2 242.6 (P ≈ 10<sup>−974</sup>).

## Cách chạy lại

```bash
pip install datasets pandas matplotlib nbconvert ipykernel
cd lab2
python ngram_lm.py                                                      # unit tests
jupyter nbconvert --to notebook --execute --inplace experiments.ipynb  # khoảng 7 phút
```

## AI assistance statement

- **Tool:** Claude Code
- **Purpose:** viết và chạy code (`ngram_lm.py`, `experiments.ipynb`), tạo biểu đồ, `results.csv`; trình bày các file markdown và ghép `report.pdf`.
- **Generated:** code, biểu đồ, `results.csv`, `report.pdf`, bản nháp các file markdown.
- **Student:** bài tính tay (`calculations.pdf`); rà soát prediction, error analysis, reflection.
- **Verified:** 5 unit tests pass (xác suất tính tay, Σ P(w | h) = 1 cho Laplace và interpolation, kể cả context chưa gặp); các bài tính tay được kiểm chứng lại trong notebook; số liệu trong các file `.md` lấy từ output của notebook.
