# LAB 08 — Pretrained Transformers and BERT Fine-Tuning

Bài toán: dùng một Transformer đã pretrain (DistilBERT) cho phân loại sentiment IMDb, so sánh với Transformer train from scratch của LAB 07. Thí nghiệm gồm: frozen so với fine-tuning, lượng training data, max sequence length, và domain shift (IMDb → Yelp / Amazon).

Pipeline: `raw text → WordPiece tokenizer → [CLS] … [SEP] + attention mask → DistilBERT (6 layer, pretrained bằng MLM) → h[CLS] → Dropout → Linear → softmax → positive / negative`.

## Cấu trúc thư mục

| File | Nội dung |
|---|---|
| `report.pdf` | Bản tổng hợp: README, calculations, prediction, 2 notebook, error_analysis, reflection |
| `calculations.md` | Bài 1: MLM loss (−ln 0.70, 0.95, 0.01). Bài 2: tokenize "The cats are playing." (token, ID, special token, attention mask, đếm word / token / subword) |
| `prediction.md` | Prediction 1–3 (và dự đoán phụ cho Exp 4, 5), ghi trước khi chạy |
| `tokenization.ipynb` | `tokens.keys()`, `input_ids` / `attention_mask`, padding trong batch, kiểm tra tác dụng của attention mask, subword (từ hiếm, tên riêng, lỗi chính tả, tiếng Việt, emoji), DistilBERT điền `[MASK]` và kiểm tra Bài 1 |
| `pretrained_classification.py` | Dữ liệu (IMDb chia như LAB 05–07, Yelp / Amazon), `PretrainedClassifier` (encoder → [CLS] → Linear), `ScratchTransformer` (cấu hình LAB 07, cùng tokenizer), `train_model`, frozen mode (`extract_features` + `train_head`), `predict_proba`, `metrics` |
| `fine_tuning.ipynb` | Exp 1–5, application, confidence analysis, error analysis (đã chạy, có output) |
| `results.csv` | Mọi số liệu (dạng dài: experiment, setting, metric, value) |
| `images/data_efficiency.png` | Accuracy / F1 theo 10–100% data: pretrained, frozen ([CLS], mean), scratch |
| `images/confusion_matrix.png` | Confusion matrix của scratch, frozen, fine-tuned |
| `images/confidence_analysis.png` | Reliability diagram (3 model), phân bố confidence của dự đoán đúng / sai |
| `images/learning_curves.png` | Train / validation loss, validation accuracy của scratch và fine-tuned |
| `images/sequence_length.png` | Accuracy, thời gian train, bộ nhớ GPU theo T = 64 / 128 / 256 |
| `error_analysis.md` | Metrics, confusion matrix, câu tự viết, 4 FP + 4 FN + 4 lỗi tự tin nhất, phân loại lỗi, confidence analysis |
| `reflection.md` | Đối chiếu prediction, trả lời câu hỏi từng experiment, Q1–Q6, learning check, reflection |

## Reproducibility

| | |
|---|---|
| Model | `distilbert-base-uncased` (66.4 triệu tham số) + head Linear(768, 2). Baseline: Transformer from scratch (cấu hình LAB 07: d_model 128, 4 head, d_ff 256, 1 layer, mean pooling; 4.0 triệu tham số) |
| Tokenizer | `distilbert-base-uncased` (WordPiece, vocab 30 522, chữ thường), dùng cho cả hai model |
| Dataset | IMDb (`stanfordnlp/imdb`); domain shift: 2 000 mẫu cân bằng nhãn từ test của `fancyzhx/yelp_polarity` và `fancyzhx/amazon_polarity` |
| Train / validation / test split | 19 923 / 4 981 (bỏ review trùng, chia stratified 80/20, seed 42) / 25 000. Giống LAB 05–07 |
| Max sequence length | 256 (Exp 4: 64 / 128 / 256) |
| Batch size | 16 (pretrained), 64 (scratch) |
| Learning rate | 2e-5 (pretrained), 1e-3 (scratch); warmup 6% rồi giảm tuyến tính; head của frozen: Adam 1e-3 |
| Epochs | 2 (pretrained), 6 (scratch), 30 (head của frozen); chọn epoch có validation loss thấp nhất |
| Optimizer | AdamW, weight decay 0.01, clip gradient 1.0, bf16 autocast |
| Random seed | 42, 43, 44 cho Exp 1; 42 cho các thí nghiệm còn lại; subset 10/25/50% lồng nhau (seed 42) |
| Hardware | NVIDIA RTX 3050 Laptop (4 GB), PyTorch 2.6.0 + CUDA 12.4, transformers 5.17.0 |
| Training time | Fine-tuning 484 s / 2 epoch (T = 256); scratch 33 s / 6 epoch; frozen 290 s (chủ yếu để trích feature). Cả notebook: 114 phút |

## Kết quả chính

| Model (IMDb test) | Params train được | Train time | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Transformer from scratch (3 seed) | 4 039 554 | 33 s | 0.835 ± 0.004 | 0.835 | 0.836 | 0.835 ± 0.003 |
| Frozen DistilBERT, [CLS] | 1 538 | 290 s | 0.853 | 0.867 | 0.833 | 0.850 |
| Frozen DistilBERT, mean | 1 538 | 289 s | 0.859 | 0.873 | 0.839 | 0.856 |
| **DistilBERT fine-tuned (3 seed)** | 66 364 418 | 484 s | **0.911 ± 0.002** | 0.909 | 0.915 | **0.911 ± 0.002** |

| Training data | 10% | 25% | 50% | 100% |
|---|---|---|---|---|
| Fine-tuned | 0.876 | 0.894 | 0.897 | 0.911 |
| Frozen (mean) | 0.837 | 0.849 | 0.854 | 0.859 |
| Scratch | 0.712 | 0.778 | 0.813 | 0.835 |

| Max length | 64 | 128 | 256 |
|---|---|---|---|
| Accuracy | 0.822 | 0.875 | 0.910 |
| Train time / peak GPU memory | 196 s / 1 345 MB | 282 s / 1 548 MB | 484 s / 1 951 MB |

| Accuracy | IMDb | Amazon | Yelp |
|---|---|---|---|
| Scratch | 0.832 | 0.785 | 0.796 |
| Fine-tuned | 0.910 | 0.927 | 0.924 |

Các phát hiện chính:

1. **Pretrained + fine-tuning hơn from scratch 7.6 điểm** (0.911 so với 0.835, ổn định qua 3 seed). Đổi lại, model lớn hơn 16 lần và train lâu hơn khoảng 15 lần.
2. **Frozen encoder đã vượt from scratch:** chỉ train 1 538 tham số mà đạt 0.853–0.859. Fine-tuning vẫn hơn frozen 5–6 điểm.
3. **Data efficiency:**
   - Chỉ với 10% data, pretrained (0.876) đã hơn from scratch dùng 100% data (0.835).
   - Đi từ 100% xuống 10%, pretrained giảm 3.5 điểm, from scratch giảm 12.4 điểm.
4. **Max length:** T = 64 → 256 tăng 8.8 điểm accuracy vì review IMDb dài (median 221 token). Chi phí tăng chậm hơn T² ở mức T này (thời gian × 2.5, bộ nhớ × 1.45).
5. **Domain shift:**
   - Model fine-tuned không giảm trên Yelp và Amazon (review ngắn, cảm xúc nói thẳng). From scratch giảm 3.6–4.7 điểm.
   - Đây là domain gần; kết quả không chứng minh model "hiểu mọi domain".
6. **Lỗi:**
   - Review bị cắt ở 256 token có tỉ lệ lỗi 12.4%, gấp đôi review không bị cắt (6.7%).
   - 3 trên 4 lỗi tự tin nhất là review khen rõ ràng nhưng mang nhãn negative (label noise).
   - Nhóm confidence ≥ 0.95 đúng 98.1% nhưng vẫn chiếm 14% tổng số lỗi.

## Lưu ý

- **Best epoch:** model fine-tuned đạt validation loss thấp nhất ở epoch 1 trong 7 trên 8 lần chạy. Epoch 2 làm val loss tăng (0.238 → 0.279) dù val accuracy vẫn tăng nhẹ.
- **Bước bị bỏ qua (NaN):**
  - Như LAB 07, GPU này thỉnh thoảng cho ra NaN; bước có gradient không hữu hạn bị bỏ qua.
  - Tổng cộng 85 bước: 63 / 2 490 bước ở T = 128 (2.5%), 21 ở T = 64, 1 ở seed 43. Các lần chạy T = 256 còn lại không có.
  - Kết quả T = 64 / 128 vì vậy có thể thấp hơn một chút so với khi train đủ bước.
- **Thời gian:** wall-clock của seed 43 epoch 1 (1 678 s) và thời gian dự đoán test ở T = 256 (232 s) bị kéo dài do máy chững lại khi đang chạy. Cột train time chỉ tính thời gian các bước train.
- **Số seed:** các điểm 10–50% data, T = 64 / 128 và domain shift chỉ chạy 1 seed (42).

## Cách chạy lại

```bash
pip install torch transformers datasets scikit-learn pandas matplotlib nbconvert ipykernel
cd lab8
python pretrained_classification.py      # kiểm tra nhanh: tokenizer, số tham số, forward
python -m jupyter nbconvert --to notebook --execute --inplace tokenization.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 fine_tuning.ipynb   # khoảng 2 giờ trên RTX 3050 Laptop
```

Đặt biến môi trường `LAB8_QUICK=1` để chạy thử notebook trên dữ liệu nhỏ (khoảng 6 phút). Đặt `LAB8_LOG=<file>` để ghi tiến độ ra file.

## AI assistance statement

- **AI-assisted code:** Claude Code.
- **What was generated:**
  - `pretrained_classification.py` (data loader, model, training loop, frozen mode);
  - code của 2 notebook (thí nghiệm, bảng, biểu đồ);
  - bản nháp các file markdown;
  - ghép `report.pdf`.
- **What was modified:**
  - Baseline from scratch ban đầu dùng pooling [CLS] và nhân embedding với √d, chỉ đạt khoảng 0.68. Khi kiểm tra thấy model bị underfit; chuyển sang mean pooling, bỏ hệ số √d như LAB 07 thì đạt 0.835, ngang LAB 07.
  - Đo bộ nhớ GPU ban đầu tính cả model khác đang nằm trên GPU; sửa lại để chỉ đo phần của lần train đang chạy.
  - Thêm frozen mean pooling và thêm from scratch vào Exp 3 và Exp 5 để so sánh.
  - Domain shift chỉ tải file test của Yelp và Amazon thay vì cả dataset.
- **What was verified:**
  - Bài tính tay MLM khớp `nll_loss` của PyTorch;
  - tokenization "The cats are playing." khớp output tokenizer;
  - attention mask đúng thì padding không làm đổi vector [CLS] (sai số 7e-6);
  - số liệu trong các file `.md` lấy từ output notebook.
- **Student:** MLM calculation, prediction (ghi trước khi chạy), giải thích pretraining / fine-tuning, phân tích kết quả, error analysis, reflection, learning check.
