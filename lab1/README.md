# LAB 01 — From Text Processing to Search

Bài toán: biểu diễn document bằng TF-IDF sparse vector và xây dựng document search engine trên **30 000 abstract bài báo Machine Learning (arXiv)**.
Pipeline: `raw text → tokenizer → counts → TF → IDF → TF-IDF → cosine → top-K → P@5 / R@5 / MRR → error analysis`.

## Cấu trúc thư mục

| File | Nội dung |
|---|---|
| `report.pdf` | **Bản tổng hợp**: ghép README, calculations (bản gõ + bản viết tay), prediction, experiments.ipynb, error_analysis, reflection |
| `calculations.md`, `calculations.pdf` | 6 bài tính tay (count, TF, IDF, TF-IDF, cosine, prediction). `calculations.pdf` là bản viết tay (scan), `calculations.md` là bản gõ lại |
| `prediction.md` | 5 dự đoán trước experiment (vocabulary, sparsity, search, preprocessing, stopwords) |
| `implementation.py` | Tự cài đặt `build_vocabulary`, `compute_counts`, `compute_tf`, `compute_idf`, `compute_tfidf`, `cosine_similarity` + 6 unit tests |
| `experiments.ipynb` | Toàn bộ thí nghiệm (Part B → I), đã chạy sẵn, có output |
| `qrels.json` | Nhãn relevance cho 8 query đánh giá (bán tự động, dựa chủ yếu trên title; xem ghi chú trong file) |
| `results.csv` | P@5, R@5, RR của từng query × 5 pipeline (+ dòng MEAN) |
| `error_analysis.md` | 2 query tốt, 2 query kém, failure case *heart attack ↔ myocardial infarction* |
| `reflection.md` | Đối chiếu prediction, Part J (hypothesis), trả lời learning check |
| `sparsity.png` | Phân bố document frequency (log-log) và số term khác 0 / document |

## Thiết lập

- **Corpus:** `CShorten/ML-ArXiv-Papers` (117 592 bài, sau khi bỏ trùng tiêu đề còn 113 277), lấy ngẫu nhiên 30 000 bài (seed 42). Document = title + abstract, trung bình 176 từ.
- **TF-IDF (Part D):** CountVectorizer chỉ dùng để tokenize + đếm; TF = c/Σc, IDF = ln(N/df) tính bằng code tự viết trên ma trận sparse. Đã kiểm chứng trùng với TfidfVectorizer khi cùng convention (max |diff| ≈ 2·10⁻¹⁶).
- **Evaluation:** 8 query, mỗi query có 9–24 relevant document. Nhãn tạo bán tự động: pool = top-10 của pipeline A, B, C + document có cụm từ liên quan; nhãn dựa chủ yếu trên title, một số document phân vân được đọc abstract. Không phải đánh giá đầy đủ toàn corpus, nên các con số là ước lượng.

## Kết quả chính

**Experiment 1 (sparse representation):** N = 30 000, V = 54 335, nnz = 3.2 triệu → **sparsity 99.80%**. Trung bình 107 term khác 0 / document. Dạng dense cần 13 GB, dạng CSR chỉ khoảng 39 MB.

**Experiment 2 (preprocessing ablation):**

| Pipeline | Vocabulary | Avg tokens/doc | Sparsity | OOV rate | P@5 | R@5 | MRR |
|---|---|---|---|---|---|---|---|
| A — lowercase + split khoảng trắng | 162 634 | 175.8 | 0.99931 | 0.0248 | **0.625** | **0.153** | 0.637 |
| B — normalize + bỏ dấu câu + bỏ stopwords | 53 489 | 112.1 | 0.99852 | 0.0148 | 0.525 | 0.130 | 0.642 |
| C — normalize + WordPiece subword | 18 695 | 208.6 | 0.99346 | **0.0006** | 0.550 | 0.140 | **0.710** |
| A0 — như A, không lowercase | 186 379 | 175.8 | **0.99936** | 0.0279 | 0.475 | 0.119 | 0.632 |
| B−sw — như B, giữ stopwords | 53 791 | 183.2 | 0.99797 | 0.0090 | 0.525 | 0.130 | 0.705 |

Các phát hiện chính:

1. **Lowercasing** giảm vocabulary 12.7% (186 379 → 162 634) và tăng P@5 từ 0.475 lên 0.625.
2. **Vocabulary nhỏ ≠ search tốt hơn:** A có vocabulary gấp 3 lần B nhưng P@5 cao hơn; C có OOV gần 0 và MRR cao nhất.
3. **Bỏ stopwords** không đổi P@5, làm MRR giảm (0.705 → 0.642). IDF vốn đã gần như triệt tiêu stopwords (idf(the) = 0.004).
4. **Lexical failure:** *heart attack prediction* có P@5 = 0 ở mọi pipeline (khớp *adversarial attack*); bài về *myocardial infarction* xếp hạng 238 hoặc có cosine = 0.

## Cách chạy lại

```bash
pip install scikit-learn datasets transformers pandas matplotlib nbconvert ipykernel
cd lab1
python implementation.py                                               # unit tests
jupyter nbconvert --to notebook --execute --inplace experiments.ipynb  # khoảng 2 phút, cần internet để tải dataset và tokenizer
```

## AI assistance statement

- **Tool:** Claude Code
- **Purpose:** viết và chạy code thí nghiệm (`implementation.py`, `experiments.ipynb`), tạo `results.csv` và biểu đồ; trình bày các file markdown và ghép `report.pdf`.
- **Generated:** code, biểu đồ, `results.csv`, `report.pdf`, bản nháp các file markdown.
- **Student:** bài tính tay (`calculations.pdf`); rà soát nhãn relevance (`qrels.json`, AI đề xuất từ pool), prediction, error analysis, reflection.
- **Verified:** 6 unit tests pass; cài đặt tự viết trùng với scikit-learn khi cùng convention; các số tính tay đã đối chiếu bằng code (Part B trong notebook); mọi số liệu trong các file `.md` lấy từ output của notebook.
