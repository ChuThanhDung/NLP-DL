# LAB 04 — Text Classification (IMDb sentiment)

Bài toán: phân loại cảm xúc review phim (positive / negative) trên **IMDb** (25 000 train / 25 000 test, cân bằng 50/50).
Pipeline: `raw text → clean → TF-IDF | Average Word2Vec → Logistic Regression / Linear SVM → evaluation → error analysis`.

## Cấu trúc thư mục

| File                                                  | Nội dung                                                                                                                                                  |
| ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `report.pdf`                                          | **Bản tổng hợp**: ghép PDF của README, calculations, prediction, experiment.ipynb, error_analysis, reflection (theo thứ tự đọc)                           |
| `calculations.md`, `calculations.pdf`                 | 4 bài tính toán (sigmoid, threshold, confusion matrix, accuracy trap): `calculations.pdf` là bài làm viết tay (bản scan), `calculations.md` là bản gõ lại |
| `prediction.md`                                       | 5 dự đoán + dự đoán chính (TF-IDF vs Avg Word2Vec), dựa trên đặc tính của representation; sinh viên chọn phương án và mức tự tin                          |
| `experiment.ipynb`                                    | Toàn bộ thí nghiệm, đã chạy sẵn, có output                                                                                                                |
| `preprocessing.py`                                    | `clean_text`, `tokenize`, `make_splits`, `average_embedding`, `load_imdb`                                                                                 |
| `results.csv`                                         | Mọi kết quả (experiment, representation, classifier, split, accuracy, precision, recall, F1, thời gian)                                                   |
| `confusion_matrix.png`                                | Confusion matrix 3 model trên test                                                                                                                        |
| `error_analysis.md`                                   | 3 mẫu đúng + 4 mẫu sai, phân loại nguyên nhân                                                                                                             |
| `reflection.md`                                       | Đối chiếu prediction, giải thích khác biệt representation, hướng tới neural NLP                                                                           |
| `class_distribution.png`, `top_features.png` | Hình bổ trợ                                                                                                                                               |

## Thiết lập thí nghiệm

- **Split:** train gốc (25 000, loại 96 review trùng → 24 904) chia stratified 80/20 thành train 19 923 / validation 4 981 (seed 42). Test = tập test chính thức của IMDb (25 000), **không** dùng để fit hay chọn siêu tham số.
- **Chống leakage:** TfidfVectorizer, StandardScaler, Word2Vec và classifier chỉ được `fit` trên train; `C` chọn theo F1 trên validation.
- **TF-IDF:** `min_df=2`, `sublinear_tf=True`, vocab 40 790 (unigram) / 364 040 (uni+bigram).
- **Word2Vec:** gensim CBOW, 100 chiều, window 5, 10 epochs, train chỉ trên văn bản train (không dùng pretrained); document vector = trung bình, chuẩn hóa bằng StandardScaler.

## Kết quả chính (test set)

| Representation        | Classifier          | Accuracy   | Precision | Recall | F1         | Thời gian train |
| --------------------- | ------------------- | ---------- | --------- | ------ | ---------- | --------------- |
| TF-IDF (unigram)      | Logistic Regression | 0.8814     | 0.8842    | 0.8778 | 0.8810     | 3.1 s           |
| Avg Word2Vec          | Logistic Regression | 0.8426     | 0.8435    | 0.8413 | 0.8424     | 51.9 s          |
| TF-IDF (unigram)      | Linear SVM          | 0.8886     | 0.8862    | 0.8917 | 0.8889     | 2.2 s           |
| TF-IDF (uni + bigram) | Logistic Regression | **0.9022** | 0.9019    | 0.9027 | **0.9023** | 9.3 s           |

Các phát hiện chính:

1. **TF-IDF > Avg Word2Vec** (+3.9 điểm accuracy) trên IMDb: review dài, tín hiệu cảm xúc nằm ở từ cụ thể; averaging làm loãng tín hiệu và mất thứ tự.
2. **Bigram** tăng F1 +2.1 điểm (0.8810 → 0.9023) nhờ bắt được _not good / not bad_ nhưng vocab tăng 8.9×, thời gian train gấp 3.0×.
3. **Bỏ stopwords làm giảm** val F1 (0.8920 → 0.8841) vì xóa mất _not, no, never, very_.
4. **Word order (Exp 4):** cos(avg("this movie is good"), avg("…is not good")) = 0.948 → average embedding gần như không phân biệt hai câu.
5. **Accuracy trap:** trên tập 950 neg / 50 pos, model "luôn đoán negative" có accuracy 0.95 nhưng F1 = 0.

## Cách chạy lại

```bash
pip install scikit-learn gensim datasets pandas matplotlib seaborn nbformat nbconvert ipykernel
cd lab4
jupyter nbconvert --to notebook --execute --inplace experiment.ipynb   # ~2-3 phút, cần internet để tải IMDb
```

## AI assistance statement

- **Tool:** Claude Code (Claude Sonnet 5.5)
- **Purpose:**
  - Viết và chạy toàn bộ code thí nghiệm (`experiment.ipynb`, `preprocessing.py`), tạo biểu đồ, tạo `results.csv`.
  - Gõ lại bài tính toán viết tay của người làm vào `calculations.md` v
  - Trình bày và ghép các file thành `report.pdf`.
- **Generated (do AI):** code, biểu đồ, `results.csv`, `report.pdf`,
- **Student (do sinh viên):** bài tính toán viết tay (`calculations.pdf`); chọn phương án và mức tự tin cho từng dự đoán trong `prediction.md`; chọn nguyên nhân cho từng mẫu và kết luận về confusion matrix trong `error_analysis.md`; chọn các ý trong `reflection.md`. AI chỉ ghi lại các lựa chọn đó.
- **Verified:** code đã chạy thành công đầu-cuối (kết quả trong `experiment.ipynb`/`results.csv`); các kết quả tính toán (z, σ(z), precision, recall, F1) đã được đối chiếu bằng code với tính tay; các số liệu trích trong `error_analysis.md` và `reflection.md` lấy từ output của notebook.
