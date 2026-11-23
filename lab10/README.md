# LAB 10 — Extractive Question Answering

Bài toán: cho câu hỏi và một đoạn context, tìm đoạn (span) trong context trả lời câu hỏi: (question, context) → (start, end). Dữ liệu là SQuAD v1.1. So sánh baseline keyword overlap với DistilBERT fine-tuned, đánh giá bằng Exact Match và F1, kiểm tra ảnh hưởng của độ dài context, mức trùng từ, độ dài đáp án và distractor, rồi dùng model cho Document QA.

Pipeline: `question + context → tokenize riêng → sliding window [CLS] question [SEP] context[i : i + L] [SEP] (stride 128) → DistilBERT → H (T × 768) → Linear(768, 2) → start_logits, end_logits → chọn (i, j) có start_logit + end_logit lớn nhất, i ≤ j, j − i < 30, trong context, trên mọi window → cắt text gốc theo offset_mapping`.

## Cấu trúc thư mục

| File                           | Nội dung                                                                                                                                                                                                                                                                                                                                                 |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `report.pdf`                 | Bản tổng hợp: README, calculations, prediction, 2 notebook, error_analysis, reflection                                                                                                                                                                                                                                                                 |
| `calculations.pdf`           | Bản chép tay. Bài 1: start / end cho "Paris" và "capital of France". Bài 2: L<sub>start</sub> = −ln 0.6, QA loss, vì sao cần 2 prediction. Bài 3: span ký tự [25, 30) → token 14. Bài 4: tính tay F1 / EM                                                                                                                                                   |
| `prediction.md`              | Prediction 1–4, ghi trước khi chạy                                                                                                                                                                                                                                                                                                                    |
| `tokenizer_alignment.ipynb`  | `input_ids` / `attention_mask` / `token_type_ids` của DistilBERT, BERT, RoBERTa; kiểm tra Exercise 1 và 3; character / word / token index; các trường hợp alignment khó (bắt đầu / kết thúc giữa token, bị cắt, ngoài window); so sánh sliding window của tokenizer với tự cắt trên 30 000 câu hỏi                         |
| `qa_model.py`                | `load_squad`, `char_to_token_span`, `encode_windows` (sliding window), `prepare_train/eval_features`, `train_qa` (có checkpoint), `postprocess`, EM / F1 chuẩn SQuAD, `baseline_answer`                                                                                                                                                   |
| `qa_finetuning.ipynb`        | Một batch qua model; baseline; **Exp 1** (max_length 128 / 256 / 384) + **Exp 2** (learning curves); **Exp 3** EM vs F1; **Exp 4** lexical overlap + 8 cặp câu diễn đạt lại; **Exp 5** độ dài đáp án; thêm câu không liên quan / distractor / đổi thứ tự câu; error analysis wrong start / end / span |
| `document_qa.py`             | Mini QA trên 3 tài liệu (FAQ của môn học, Transformer / BERT, SQuAD): 25 câu hỏi có gold + 3 câu không có đáp án; hỏi trên nhiều tài liệu                                                                                                                                                                                             |
| `results.csv`                | Mọi số liệu (dạng dài: experiment, setting, metric, value)                                                                                                                                                                                                                                                                                           |
| `learning_curves.png`        | Train / validation loss 4 lần mỗi epoch, kèm val EM / F1, cho 3 max_length                                                                                                                                                                                                                                                                             |
| `em_f1_analysis.png`         | EM và F1 của baseline / DistilBERT; phân bố F1 của các câu sai EM                                                                                                                                                                                                                                                                                  |
| `answer_length_analysis.png` | EM và F1 theo độ dài đáp án (1 / 2–3 / 4–6 / 7+ word)                                                                                                                                                                                                                                                                                            |
| `error_analysis.md`          | Thống kê wrong start / end / boundary / span; 7 + 6 + 7 ví dụ; EM = 0 nhưng F1 > 0; nguyên nhân; Document QA                                                                                                                                                                                                                                       |
| `reflection.md`              | Đối chiếu prediction, trả lời các experiment, learning check, reflection LAB 08–10                                                                                                                                                                                                                                                                 |

## Reproducibility

|                     |                                                                                                            |
| ------------------- | ---------------------------------------------------------------------------------------------------------- |
| Model               | `distilbert-base-uncased` + `Linear(768, 2)` (`AutoModelForQuestionAnswering`), 66.4 triệu tham số |
| Tokenizer           | `distilbert-base-uncased` (WordPiece, chữ thường). DistilBERT không dùng `token_type_ids`         |
| Dataset             | SQuAD v1.1 (`rajpurkar/squad`)                                                                           |
| Train split         | 30 000 câu hỏi lấy ngẫu nhiên từ SQuAD train (417 bài, sau khi bỏ 25 bài dùng cho validation)    |
| Validation split    | 2 000 câu hỏi từ 25 bài Wikipedia giữ riêng, không trùng context với train                        |
| Test split          | SQuAD v1.1 dev chính thức: 10 570 câu hỏi, 48 bài, tới 3 đáp án gold mỗi câu                    |
| Max sequence length | 384 (model chính); Exp 1: 128 / 256 / 384                                                                 |
| Stride              | 128 (64 khi max_length = 128); câu hỏi cắt tối đa 64 token                                            |
| Batch size          | 16                                                                                                         |
| Learning rate       | 3e-5, warmup 10% rồi giảm tuyến tính, weight decay 0.01                                                |
| Epochs              | 2; chọn epoch có validation F1 cao nhất (epoch 2 cho cả 3 độ dài)                                   |
| Optimizer           | AdamW, clip gradient 1.0, bf16 autocast                                                                    |
| Random seed         | 42                                                                                                         |
| Hardware            | NVIDIA RTX 3050 Laptop (4 GB), PyTorch 2.6.0 + CUDA 12.4, transformers 5.17.0                              |
| Training time       | 950 s (max_length 384); 843 s (256); 1 132 s (128). Cả notebook khoảng 1 giờ                            |
| Post-processing     | top-20 start × top-20 end, độ dài đáp án ≤ 30 token                                                |

## Kết quả chính (SQuAD dev)

| Model                                | EM              | F1              |
| ------------------------------------ | --------------- | --------------- |
| Baseline (keyword overlap)           | 15.42           | 28.27           |
| DistilBERT, max_length 128           | 70.46           | 79.42           |
| DistilBERT, max_length 256           | 70.77           | 80.17           |
| **DistilBERT, max_length 384** | **71.07** | **80.49** |

| Thí nghiệm (DistilBERT 384)                                                | Kết quả                                                  |
| ---------------------------------------------------------------------------- | ---------------------------------------------------------- |
| Mức trùng từ câu hỏi ↔ câu chứa đáp án: thấp / trung bình / cao | F1 72.5 / 81.5 / 86.7 (baseline 15.3 / 29.6 / 38.5)        |
| Độ dài đáp án 1 / 2–3 / 4–6 / 7+ word                                | EM 75.8 / 74.6 / 67.1 / 43.1, F1 81.8 / 82.6 / 80.1 / 65.8 |
| 1 000 câu hỏi: gốc / + không liên quan / + distractor / đổi thứ tự  | F1 80.6 / 79.1 / **22.4** / 79.1                     |
| Lỗi: wrong start / wrong end / wrong boundary / wrong span                  | 759 / 669 / 139 / 1 603                                    |
| Document QA (25 câu hỏi)                                                   | EM 88.0, F1 93.4                                           |

Các phát hiện chính:

1. **Pretrained Transformer hơn baseline 52 điểm F1.** Baseline tìm đúng câu chứa đáp án 77.9% nhưng chọn span rất kém (EM 15.4): phần khó của QA là chọn biên, không phải tìm câu.
2. **Context length gần như không quan trọng nếu có sliding window:** 128 → 384 chỉ tăng 1.1 F1. T = 128 lại train lâu nhất vì có 2.4 lần số window.
3. **Model vẫn dựa vào trùng từ:** F1 giảm 14 điểm khi câu hỏi ít trùng từ với context. Một câu distractor chép lại câu hỏi làm F1 rơi từ 80.6 xuống 22.4, và 70.5% câu trả lời là đáp án giả.
4. **Đáp án dài khó hơn:** EM nhóm 7+ word chỉ 43%, vì model trả đáp án ngắn hơn gold khoảng 4 word. Wrong start (759) nhiều hơn wrong end (669).
5. **Extractive QA luôn trả về một span,** kể cả khi không có đáp án. Nhưng score của các câu đó (−0.5 đến 3) thấp hơn hẳn câu có đáp án (median 14), nên có thể dùng ngưỡng score để trả lời "không tìm thấy".

## Lưu ý

- **Sliding window tự viết:** với `transformers` 5.17, `return_overflowing_tokens=True` cho cặp (question, context) chỉ sinh tối đa 2 window, phần context phía sau bị bỏ mà không báo lỗi. Ở max_length 128, chỉ 86.5% câu hỏi có window chứa đáp án. `qm.encode_windows` tự cắt window, phủ 100% context; window đầu trùng khớp với tokenizer (bảng so sánh ở `tokenizer_alignment.ipynb`, mục 5).
- **Bước train bị bỏ qua (NaN):** như LAB 07–09, bước có gradient không hữu hạn bị bỏ: 504 / khoảng 9 100 bước ở max_length 128, 10 ở 256, 7 ở 384. Kết quả T = 128 có thể bị kéo thấp một chút.
- **Chỉ dùng 30 000 / 87 599 câu hỏi train** và 2 epoch, để chạy được 3 lần fine-tune trên GPU laptop. DistilBERT train trên toàn bộ SQuAD đạt khoảng 77 EM / 85 F1, nên kết quả ở đây thấp hơn khoảng 5 điểm.
- **Distractor tạo tự động** bằng cách thay từ hỏi trong câu hỏi bằng một đáp án giả cùng loại câu hỏi, nên nhiều câu sai ngữ pháp.
- **Tài liệu và gold của Document QA do mình tự viết;** FAQ môn học lấy thông tin từ các đề lab.
- **`ckpt/` và `model_qa/` không đưa lên git.** Chạy lại `qa_finetuning.ipynb` để tạo lại.

## Cách chạy lại

```bash
pip install torch transformers datasets pandas matplotlib nbconvert ipykernel
cd lab10
python qa_model.py                        # test char_to_token_span (6 trường hợp) + EM / F1
python -m jupyter nbconvert --to notebook --execute --inplace tokenizer_alignment.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 qa_finetuning.ipynb   # khoảng 1 giờ; tạo model_qa/
python document_qa.py                     # 25 câu hỏi có gold + 3 câu không có đáp án
python document_qa.py "When is LAB 10 due?"
```

Đặt `LAB10_QUICK=1` để chạy thử notebook trên dữ liệu nhỏ, `LAB10_LOG=<file>` để ghi tiến độ ra file.

## Sử dụng AI

- **Công cụ:** Claude.
- **AI tạo:** code (`qa_model.py`, 2 notebook, `document_qa.py`), biểu đồ, bản nháp các file markdown, ghép `report.pdf`.
- **Đã sửa:** tự viết sliding window vì tokenizer chỉ sinh 2 window; thêm checkpoint theo epoch vì GPU lỗi; chặn đáp án rỗng trong `char_to_token_span`.
- **Đã kiểm tra:** test alignment và EM / F1 pass; bài tính tay khớp tokenizer; số liệu trong file `.md` lấy từ output notebook.
- **Sinh viên làm:** bài tính tay, prediction, error analysis, reflection.
