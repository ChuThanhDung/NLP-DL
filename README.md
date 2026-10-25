# Xử lý ngôn ngữ tự nhiên và ứng dụng — Bài thực hành

Repo này chứa bài làm các lab của môn **Xử lý ngôn ngữ tự nhiên và ứng dụng**, Học kỳ I – 2026, Trường ĐH Khoa học Tự nhiên – ĐHQGHN.

- Sinh viên: Chu Thanh Dung — MSSV 23001506
- Giảng viên / TA: Phạm Ngọc Hải

Mỗi lab đi theo cùng một chu trình: **tính tay → dự đoán → thí nghiệm → cài đặt → ứng dụng → đánh giá → phân tích lỗi → reflection**. Các lab nối tiếp nhau, đi từ biểu diễn thưa dựa trên đếm từ tới mô hình tuần tự và attention:

```
TF-IDF  →  N-gram LM  →  Word Embedding  →  Text Classification  →  RNN / LSTM / GRU  →  Attention  →  (Transformer)
 Lab 1       Lab 2          Lab 3                Lab 4                   Lab 5             Lab 6
```

## Các lab

| Lab | Chủ đề | Dữ liệu | Kết quả nổi bật |
|---|---|---|---|
| [Lab 1](lab1/) | Text processing, TF-IDF và document search | 30 000 abstract arXiv (ML) | Ma trận TF-IDF thưa 99.80%; search: P@5 tốt nhất 0.625, MRR tốt nhất 0.710; phân tích failure *heart attack* ↔ *myocardial infarction* |
| [Lab 2](lab2/) | N-gram language model, smoothing, perplexity | 20 000 abstract arXiv (ML) | Trigram interpolation: test perplexity 187.3 (unigram 827.7); MLE cho PPL = ∞ trên dữ liệu mới |
| [Lab 3](lab3/) | Co-occurrence, Word2Vec, semantic search | text8 (17M token Wikipedia) | Skip-gram: WordSim-353 ρ = 0.689, analogy 38.6%; phân tích polysemy (*bank*, *doctor*) |
| [Lab 4](Lab4/) | Text classification: TF-IDF vs Average Word2Vec | IMDb (50 000 review) | TF-IDF uni + bigram + Logistic Regression: accuracy 0.902 |
| [Lab 5](lab5/) | Sequence models: RNN, LSTM, GRU | IMDb | GRU 0.832, LSTM 0.816, RNN 0.741 (baseline TF-IDF 0.888); gradient tới token đầu của RNN gần như bằng 0 |
| [Lab 6](lab6/) | Attention: scaled dot-product, self-attention (NumPy), attention classifier | text8, IMDb | Self-attention classifier 0.835 so với LSTM 0.826, train nhanh hơn 3.4 lần; không nhạy thứ tự từ; chi phí O(T²); attention chỉ khớp một phần với token quyết định (ρ = 0.36) |

## Cấu trúc mỗi thư mục lab

| File | Nội dung |
|---|---|
| `README.md` | Thiết lập thí nghiệm, kết quả chính, cách chạy lại, AI assistance statement |
| `report.pdf` | Bản tổng hợp toàn bộ bài làm của lab (README, bài tính tay, prediction, notebook, error analysis, reflection) |
| `calculations*.pdf` | Bài tính tay (bản scan) |
| `prediction.md` | Dự đoán ghi trước khi chạy thí nghiệm |
| `*.py` | Phần cài đặt cốt lõi tự viết, có unit test |
| `*.ipynb` | Notebook thí nghiệm, đã chạy sẵn và có output |
| `results.csv` | Toàn bộ số liệu thí nghiệm |
| `error_analysis.md`, `reflection.md` | Phân tích lỗi và tổng kết |

## Chạy lại

Môi trường: Python 3.12. Các thư viện chính: `scikit-learn`, `gensim`, `datasets`, `transformers`, `pandas`, `matplotlib`, `torch` (Lab 5, 6; có GPU thì nhanh hơn).

```bash
pip install scikit-learn gensim datasets transformers pandas matplotlib torch nbconvert ipykernel
cd lab1
jupyter nbconvert --to notebook --execute --inplace experiments.ipynb
```

Dữ liệu được tải tự động từ Hugging Face (`CShorten/ML-ArXiv-Papers`, `stanfordnlp/imdb`) và `gensim-data` (text8), nên lần chạy đầu cần internet. Lệnh chạy riêng cho từng lab nằm trong `README.md` của lab đó.
