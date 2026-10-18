# LAB 05 — Sequence Models: RNN, LSTM, GRU

Bài toán: sentiment IMDb (positive / negative) bằng sequence model. Pipeline: `raw text → tokenize → vocabulary → token IDs → padding/truncation → Embedding → RNN/LSTM/GRU → Linear → sentiment`.

## Cấu trúc thư mục

| File | Nội dung |
|---|---|
| `report.pdf` | Bản tổng hợp: README, calculations (bản gõ + bản viết tay), prediction, sequence_classification.ipynb, error_analysis, reflection |
| `calculations.md`, `calculations.pdf` | Bài 1 (một bước RNN), Bài 2 (information flow), Bài 3 (thứ tự câu), LSTM gates, vanishing gradient bằng số. `calculations.pdf` là bản viết tay (scan) |
| `prediction.md` | 5 dự đoán trước khi train |
| `rnn_cell.py` | `SimpleRNNCell` tự cài bằng NumPy, kiểm chứng với `torch.nn.RNNCell` (sai số ≈ 10⁻¹⁶) |
| `sequence_classification.ipynb` | Dataset inspection, baseline, RNN/LSTM/GRU, sequence length, long-range dependency, gradient theo vị trí, error analysis, application (đã chạy, có output) |
| `results.csv` | Metrics, thời gian train, số tham số, bộ nhớ GPU của mọi model |
| `error_analysis.md` | 3 mẫu đúng, 3 mẫu sai, 3 mẫu sai do truncation |
| `reflection.md` | Đối chiếu prediction, trade-off sequence length, giới hạn recurrent model |
| `learning_curves.png`, `confusion_matrix.png`, `max_length.png`, `gradient_position.png`, `sequence_length.png` | Biểu đồ |

## Thiết lập

- **Dữ liệu:** IMDb (`stanfordnlp/imdb`). Train gốc bỏ 96 review trùng, chia 80/20 stratified → train 19 923 / validation 4 981 (seed 42). Test = 25 000 review chính thức, không dùng để chọn siêu tham số. 123 review xuất hiện ở cả train và test (đã ghi nhận).
- **Tokenizer:** lowercase, bỏ `<br />` và ký tự đặc biệt, tách theo khoảng trắng. Vocabulary 25 000 từ phổ biến nhất + `<pad>` + `<unk>`.
- **Model:** Embedding 100 chiều (train từ đầu) → RNN/LSTM/GRU 1 layer, hidden 128 → Dropout 0.3 → Linear. Dùng packed sequence để hidden state cuối là token thật cuối cùng.
- **Huấn luyện:** Adam lr 1e-3, batch 64, 6 epoch, clip gradient 1.0, BCE loss. Chọn epoch có validation loss thấp nhất.
- **Phần cứng:** GPU NVIDIA RTX 3050 Laptop, PyTorch 2.6 + CUDA 12.4.

## Kết quả chính (test)

| Model | Max len | Accuracy | F1 | Train time |
|---|---|---|---|---|
| TF-IDF + LR | full | 0.8882 | 0.8883 | 7.1 s |
| RNN | 200 | 0.7408 | 0.7482 | 37.3 s |
| LSTM | 200 | 0.8156 | 0.8117 | 42.3 s |
| GRU | 200 | **0.8322** | **0.8219** | 40.4 s |

Các phát hiện chính:

1. **Thứ tự và dependency xa:** gradient đến token đầu của review 200 token: RNN ≈ 10⁻⁷, LSTM ≈ 0.06, GRU ≈ 0.29 (so với token cuối).
2. **Sequence length:** max_len 50 / 100 / 200 / 400 → F1 0.752 / 0.785 / 0.812 / 0.847; thời gian train 17.5 / 25.6 / 42.3 / 69.9 s.
3. **Cắt đầu hay cắt cuối:** với max_len 100, giữ 100 token cuối cho F1 0.828 so với 0.785 khi giữ 100 token đầu.
4. **Baseline vẫn mạnh nhất:** TF-IDF + LR (0.888) thắng cả ba recurrent model train từ đầu trên 25K review.

## Cách chạy lại

```bash
pip install torch datasets scikit-learn pandas matplotlib nbconvert ipykernel
cd lab5
python rnn_cell.py                                                       # kiểm chứng RNN cell
jupyter nbconvert --to notebook --execute --inplace sequence_classification.ipynb   # ~20–30 phút trên GPU
```

## AI assistance statement

- **Tool:** Claude Code
- **Purpose:** viết code (`rnn_cell.py`, `sequence_classification.ipynb`), gỡ lỗi (NaN ở validation loss; đổi sang tính BCE trên logits, sau đó notebook chạy hết không lỗi), tạo biểu đồ và `results.csv`; trình bày các file markdown và ghép `report.pdf`.
- **Generated:** code, biểu đồ, `results.csv`, `report.pdf`, bản nháp các file markdown.
- **Student:** bài tính tay (`calculations.pdf`); rà soát prediction, giải thích gradient, lựa chọn nguyên nhân lỗi, reflection.
- **Verified:** `rnn_cell.py` trùng PyTorch (sai số ≈ 10⁻¹⁶) và khớp bài tính tay (h<sub>t</sub> = [0.5850, 0.2913]); số liệu trong các file `.md` lấy từ output notebook.
