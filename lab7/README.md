# LAB 07 — Transformer Architecture

Bài toán: tự ghép một Transformer Encoder block từ các thành phần cơ bản (multi-head self-attention, positional encoding, FFN, residual + LayerNorm), kiểm chứng vai trò của positional information, rồi dùng nó để phân loại sentiment IMDb và so sánh với LSTM của LAB 05.

Pipeline: `raw text → tokenizer → token IDs → Embedding → + Positional Encoding → [Multi-Head Self-Attention → Add & Norm → FFN → Add & Norm] → mean pooling → Linear → P(positive)`.

## Cấu trúc thư mục

| File | Nội dung |
|---|---|
| `report.pdf` | Bản tổng hợp: README, calculations, prediction, notebook, error_analysis, reflection |
| `calculations.pdf` | Bài tính tay (bản scan), bài 1–6: một attention head (X ∈ ℝ<sup>3×4</sup>, d<sub>k</sub> = 2), multi-head và concat + W<sub>O</sub>, positional encoding (d = 4, pos = 0, 1, 2), FFN, residual + LayerNorm, shape của từng bước (B = 8, T = 100, d<sub>model</sub> = 128, 4 head) |
| `prediction.md` | 6 dự đoán, ghi trước khi chạy code |
| `transformer_block.py` | `MultiHeadSelfAttention`, `FeedForward`, `TransformerBlock` (post-LN) tự ghép từ Linear / softmax / LayerNorm / Dropout, có ghi shape từng bước. Kèm 8 unit test, trong đó có so khớp với `nn.MultiheadAttention` và `nn.TransformerEncoderLayer` khi chép trọng số |
| `positional_encoding.py` | Sinusoidal PE (NumPy và module PyTorch) + 5 test |
| `transformer_classification.ipynb` | Toàn bộ thí nghiệm (đã chạy, có output) |
| `results.csv` | Mọi số liệu (dạng dài: experiment, setting, metric, value) |
| `images/positional_encoding.png` | PE matrix, PE(p)·PE(q), khác biệt representation "dog bites man" / "man bites dog" khi không có và có PE (cột đỏ "không PE" = 0 nên không thấy) |
| `images/attention_heads.png` | Attention của 4 head trên một câu, khoảng cách giữa các head (đã train so với lúc khởi tạo, 4 và 8 head) |
| `images/learning_curves.png` | Train / validation loss của LSTM, Transformer, Transformer không PE, Transformer CLS |
| `images/confusion_matrix.png` | Confusion matrix của LSTM, Transformer, Transformer không PE |
| `images/cost.png` | Thời gian và bộ nhớ một bước train theo T (LSTM so với Transformer) |
| `error_analysis.md` | 3 mẫu đúng, 5 mẫu sai, 9 câu application; attention đối chiếu với occlusion |
| `reflection.md` | Đối chiếu prediction, trả lời câu hỏi từng experiment, so sánh LAB 04 → 07, learning check |

## Thiết lập

- **Dữ liệu (giống LAB 05/06):** IMDb, train 19 923 / validation 4 981 / test 25 000; vocabulary 25 000 từ; max_length 200 (Exp 5: 32–256).
- **Transformer:** embedding 128 → sinusoidal PE → 1 `TransformerBlock` (d<sub>model</sub> 128, 4 head, d<sub>ff</sub> 256, dropout 0.1) → mean pooling trên token thật (Exp 6: [CLS]) → dropout 0.3 → Linear. Padding được mask khỏi key và khỏi pooling.
- **LSTM:** embedding 128 → LSTM hidden 128 → h<sub>T</sub> → dropout 0.3 → Linear.
- **Chung:** Adam lr 1e-3, batch 64, 6 epoch, clip 1.0, chọn epoch có validation loss thấp nhất; 3 seed (42, 43, 44) cho Exp 2, 4, 6. Hai model gần như cùng số tham số (3.33 triệu).
- **Phần cứng:** GPU NVIDIA RTX 3050 Laptop (4 GB), PyTorch + CUDA.
- **Lưu ý về GPU:**
  - Trên máy này, thỉnh thoảng một forward pass trả về NaN. Chạy lại với cùng input, tham số và random state thì không còn NaN, nên đây là lỗi không tất định của GPU.
  - Vòng train bỏ qua bước có gradient không hữu hạn (như `GradScaler` của AMP): 9 / khoảng 48 000 bước, xảy ra cả với `nn.TransformerEncoderLayer` của PyTorch.
  - Khi dự đoán, batch nào ra NaN thì được tính lại.

## Kết quả chính

| Model (IMDb test, trung bình 3 seed) | Params | Train time | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| LSTM | 3 332 481 | 60.7 s | 0.825 ± 0.009 | 0.845 | 0.798 | 0.820 ± 0.013 |
| **Transformer** (4 head, mean, PE) | 3 332 865 | **41.8 s** | **0.835 ± 0.004** | 0.841 | 0.828 | **0.834 ± 0.008** |
| Transformer không PE | 3 332 865 | 41.4 s | 0.838 ± 0.007 | 0.849 | 0.824 | 0.835 ± 0.012 |
| Transformer CLS pooling | 3 332 993 | 42.6 s | 0.837 ± 0.004 | — | — | 0.834 ± 0.007 |
| `nn.TransformerEncoderLayer` (1 seed, đối chiếu) | 3 332 865 | 46.6 s | 0.833 | 0.854 | 0.804 | 0.828 |

| Heads | d<sub>k</sub> | Params | Train time | Accuracy | F1 |
|---|---|---|---|---|---|
| 1 | 128 | 3 332 865 | 27.3 s | 0.831 ± 0.004 | 0.828 ± 0.007 |
| 2 | 64 | 3 332 865 | 34.1 s | 0.835 ± 0.003 | 0.834 ± 0.005 |
| 4 | 32 | 3 332 865 | 41.8 s | 0.835 ± 0.004 | 0.834 ± 0.008 |
| 8 | 16 | 3 332 865 | 60.0 s | 0.837 ± 0.004 | 0.838 ± 0.008 |

Các phát hiện chính:

1. **Positional information là bắt buộc để biết thứ tự:**
   - Không có PE, đảo thứ tự từ làm P đổi tối đa 9·10<sup>−8</sup>, và "dog bites man" / "man bites dog" cho representation giống hệt (khác biệt 0).
   - Có PE: khác biệt 0.28 (vector câu), ΔP tới 0.115.
   - Nhưng trên IMDb, bỏ PE **không** làm giảm accuracy, vì tín hiệu chủ yếu nằm ở việc có những từ nào.
2. **Transformer nhỉnh hơn LSTM** (+0.01 accuracy, ổn định hơn giữa các seed) và **train nhanh hơn 1.45 lần** ở T = 200. Nhưng vẫn kém TF-IDF + LR của LAB 04 (0.881).
3. **Số head:**
   - Tham số giữ nguyên, d<sub>k</sub> giảm; accuracy không đổi đáng kể (chênh ≤ 0.006); thời gian tăng theo số head.
   - Sau khi train, các head khác nhau rõ (TV 0.61–0.66, so với 0.25 lúc khởi tạo); không có head redundant.
4. **Chi phí O(T<sup>2</sup>):**
   - Tăng max_length 128 → 256 cải thiện accuracy 0.805 → 0.846, nhưng attention matrix ×4 và bộ nhớ ×2.
   - Từ T ≈ 512 Transformer chậm hơn LSTM. T = 1 024 cần 4.5 GB (vượt VRAM); T = 2 048 không chạy được.
5. **Attention ≠ giải thích:** ρ(attention nhận được, ảnh hưởng thật) = 0.46. Che 5 token được nhìn nhiều nhất làm P đổi gấp 10 lần che ngẫu nhiên, nhưng chỉ 44% review có token được nhìn nhiều nhất trùng với token ảnh hưởng nhất.

## Cách chạy lại

```bash
pip install numpy torch datasets scikit-learn scipy pandas matplotlib nbconvert ipykernel
cd lab7
python transformer_block.py        # 8 unit tests + shape trace
python positional_encoding.py      # 5 unit tests + PE d = 4
python -m jupyter nbconvert --to notebook --execute --inplace transformer_classification.ipynb   # khoảng 35 phút trên GPU
```

## AI assistance statement

- **Tool:** Claude Code
- **Purpose:** viết code (`transformer_block.py`, `positional_encoding.py`, notebook: training loop, thí nghiệm, biểu đồ), debug lỗi NaN; trình bày các file markdown và ghép `report.pdf`.
- **Generated:** code, biểu đồ, `results.csv`, `report.pdf`, bản nháp các file markdown.
- **Modified:**
  - Debug lỗi NaN: ban đầu tưởng do mask / softmax. Kiểm tra cho thấy cùng input, tham số và random state lúc NaN lúc không, nên chuyển sang bỏ qua bước lỗi và ghi lại số bước bị bỏ.
  - Chạy 3 seed thay vì 1 vì chênh lệch giữa các model chỉ khoảng 0.01.
  - Thêm Transformer không PE và đo khoảng cách giữa các head so với lúc khởi tạo.
- **Student:** bài tính tay `calculations.pdf`, shape calculation, positional encoding calculation, prediction (ghi trước khi chạy), diễn giải attention head, error analysis, reflection, learning check.
- **Verified:**
  - 13 unit test pass;
  - block tự ghép khớp `nn.TransformerEncoderLayer` khi chép trọng số (sai số < 10<sup>−5</sup>);
  - bài tính tay khớp code;
  - số liệu trong các file `.md` lấy từ output notebook.
