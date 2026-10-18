# LAB 03 — Word Representations and Embeddings

Bài toán: xây dựng biểu diễn từ dense từ thống kê đồng xuất hiện (co-occurrence) và Word2Vec; đánh giá bằng similarity, analogy và semantic search.
Pipeline: `context → co-occurrence matrix → PPMI → (SVD) → cosine → Word2Vec (CBOW / Skip-gram) → WordSim-353, analogy → semantic search`.

## Cấu trúc thư mục

| File                   | Nội dung                                                                                                                                                                                                      |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `report.pdf`           | Bản tổng hợp: README, calculations, prediction, word_embedding.ipynb, error_analysis, reflection                                                                                                              |
| `calculations.md`      | Bài 1 (co-occurrence), Bài 2–3 (cosine), Bài 4 (sparse vs dense), CBOW/Skip-gram training pairs, analogy                                                                                                      |
| `prediction.md`        | 4 dự đoán trước khi chạy model                                                                                                                                                                                |
| `cooccurrence.py`      | `build_vocabulary`, `build_cooccurrence_matrix`, `ppmi`, `cosine_similarity`, `most_similar` (tự cài đặt) + 5 unit tests                                                                                      |
| `word_embedding.ipynb` | Co-occurrence (window 1/2/5, raw vs PPMI), Word2Vec CBOW vs Skip-gram, nearest neighbors, siêu tham số (window, dimension), IMDb downstream, WordSim-353, analogy, semantic search, polysemy _bank_ (đã chạy) |
| `results.csv`          | Toàn bộ metric của các thí nghiệm                                                                                                                                                                             |
| `error_analysis.md`    | 3 similarity đúng, 3 sai/bất ngờ, phân loại nguyên nhân                                                                                                                                                       |
| `reflection.md`        | Đối chiếu prediction, bảng representation, bank cần contextual embedding, bài học                                                                                                                             |

## Thiết lập

- **Corpus:** text8 (17 005 207 token Wikipedia, lowercase, không dấu câu, 253 854 từ khác nhau).
- **Co-occurrence (tự cài):** 2 000 000 token đầu, vocabulary 10 000 từ phổ biến nhất; window đối xứng, không vượt qua ranh giới câu.
- **Word2Vec:** `gensim` 100 chiều, min_count 5 (71 290 từ), 5 epoch, negative sampling 5, subsampling 1e-3, seed 42. CBOW (`sg=0`) và Skip-gram (`sg=1`).
- **Đánh giá:** WordSim-353 (Spearman ρ) và Google analogy (19 544 câu, `questions-words.txt`).
- **Semantic search:** corpus 30K arXiv của LAB 01, Word2Vec-arXiv train trên 113K abstracts (không dùng nhãn), so sánh với TF-IDF bằng `qrels.json` của LAB 01.
- **Downstream (Exp 4):** 5 000 review IMDb train / 5 000 test, document = trung bình word vector → Logistic Regression.

## Kết quả chính

**Co-occurrence (Exp 1):** window 1 cho _doctor → physician_ là gần nhất (sau PPMI) trong khi window 5 cho _doctor → patient, medical_. Raw count cho cosine cao chung chung (_king–queen_ 0.93–0.97); PPMI làm giảm nhiễu từ phổ biến.

**Architecture (Exp 2, window 5, dim 100):**

| Model         | WordSim-353 ρ | Analogy acc | Train time |
| ------------- | ------------- | ----------- | ---------- |
| CBOW          | 0.622         | 0.312       | 26 s       |
| **Skip-gram** | **0.689**     | **0.386**   | 52 s       |

**Window (Exp 3):** window 2 / 5 / 10 → WordSim 0.637 / 0.689 / 0.691; analogy 0.302 / 0.386 / 0.410. Window lớn hơn tăng analogy nhưng làm _doctor_ gần nhân vật truyền hình hơn.

**Dimension (Exp 4):** 50 / 100 / 300 → WordSim 0.662 / 0.689 / 0.676; IMDb accuracy (avg embedding + LR) 0.723 / 0.754 / 0.785. Thời gian train 55 s / 52 s / 207 s.

**Semantic search (LAB 01 corpus, 8 query):**

| Phương pháp                        | P@5   | MRR   |
| ---------------------------------- | ----- | ----- |
| TF-IDF                             | 0.525 | 0.642 |
| Query expansion (Word2Vec)         | 0.250 | 0.321 |
| Avg embedding (Word2Vec-arXiv)     | 0.300 | 0.664 |
| Hybrid (0.5 TF-IDF + 0.5 avg emb.) | 0.475 | 0.664 |

Tóm lại: embedding cải thiện MRR nhưng làm giảm P@5 so với TF-IDF; Hybrid giữ được gần độ chính xác TF-IDF với MRR tốt hơn. Query expansion không có lợi trong thiết lập này.

## Cách chạy lại

```bash
pip install gensim scikit-learn datasets pandas matplotlib scipy nbconvert ipykernel
cd lab3
python cooccurrence.py                                                     # unit tests
jupyter nbconvert --to notebook --execute --inplace word_embedding.ipynb  # ~10 phút, cần text8 (gensim-data) và IMDb
```

## AI assistance statement

- **Tool:** Claude Code
- **Purpose:** viết code (`cooccurrence.py`, `word_embedding.ipynb`), tạo biểu đồ và `results.csv`; trình bày các file markdown và ghép `report.pdf`.
- **Generated:** code, biểu đồ, `results.csv`, `report.pdf`, bản nháp các file markdown.
- **Student:** bài tính tay, prediction, rà soát giải thích các similarity trong error analysis và reflection.
- **Verified:** 5 unit tests pass; bài tính tay khớp với `cooccurrence.py` (cos(cat, dog) = 1, cos(doctor, physician) = 0.987); số liệu trong các file `.md` lấy từ output notebook.
