# LAB 09 — Named Entity Recognition and Information Extraction

Bài toán: chuyển từ sequence classification (LAB 08, một nhãn cho cả câu) sang token classification. Mỗi word nhận một nhãn BIO trong 9 nhãn (`O`, `B-/I-` của PER, ORG, LOC, MISC). Dữ liệu là CoNLL-2003. So sánh baseline Logistic Regression trên word features với `bert-base-cased` fine-tuned, đánh giá ở mức entity, rồi dùng model cho Information Extraction.

Pipeline: `words → tokenizer (is_split_into_words) → subword tokens + word_ids → alignment nhãn (Strategy A: subword đầu, còn lại IGNORE) → BERT → H (T × 768) → Linear → Z (T × 9) → nhãn ở subword đầu của mỗi word → BIO → entity span → JSON`.

## Cấu trúc thư mục

| File | Nội dung |
|---|---|
| `report.pdf` | Bản tổng hợp: README, calculations (bản viết tay), prediction, 2 notebook, error_analysis, reflection |
| `calculations.md`, `calculations9.pdf` | Bài 1: BIO cho "Elon Musk founded SpaceX in California.". Bài 2: biên entity "University of California Berkeley". Bài 3: shape H, Z, Y với B = 8, T = 128, d = 768, C = 9 |
| `prediction.md` | Prediction 1–3, ghi trước khi chạy |
| `baseline_ner.py` | Baseline: word features (dạng từ, chữ hoa, tiền tố / hậu tố, 2 từ mỗi bên) → Logistic Regression. Kèm các hàm dùng chung: `load_conll`, `bio_to_spans`, `entity_scores` / `entity_report` (entity-level P / R / F1, khớp với `seqeval`), `align_labels` (Strategy A / B), `word_predictions` |
| `tokenizer_alignment.ipynb` | `tokens`, `input_ids`, `attention_mask`, `word_ids`, `labels` cho các câu trong đề; Strategy A so với B; padding; IGNORE trong loss; **Exp 1**: thống kê tokenization trên toàn bộ test |
| `ner_finetuning.ipynb` | Một batch qua model (shape H, Z, labels, predictions); fine-tune BERT (A và B); baseline; **Exp 2–5**; entity hiếm và subword; confusion matrix; error analysis; context analysis |
| `information_extraction.py` | Text → NER → JSON `{PERSON, ORGANIZATION, LOCATION, MISC}`; đánh giá entity-level trên 30 câu tin tức tự viết có gold |
| `results.csv` | Mọi số liệu (dạng dài: experiment, setting, metric, value) |
| `entity_metrics.csv` | P / R / F1 / support / TP / FP / FN theo loại entity, cho baseline, BERT A và BERT B |
| `entity_length_analysis.png` | F1 và recall theo độ dài entity (1 / 2 / 3 / 4+ word) |
| `confusion_matrix.png` | Entity-level (baseline, BERT): loại gold → loại dự đoán / sai biên / bỏ sót; word-level 9 × 9 nhãn BIO |
| `error_analysis.md` | Phân loại lỗi, 6 FP + 6 FN + 6 lỗi biên + 3 lỗi sai loại, phân tích theo nguyên nhân, context analysis |
| `reflection.md` | Đối chiếu prediction, trả lời câu hỏi các experiment, IE, learning check, reflection |

## Reproducibility

| | |
|---|---|
| Dataset | CoNLL-2003 (`eriktks/conll2003`, bản parquet `refs/convert/parquet`). Train 14 041 câu / 23 499 entity, validation 3 250 / 5 942, test 3 453 / 5 648. 9 nhãn BIO |
| Model | `bert-base-cased` + `Linear(768, 9)` (`AutoModelForTokenClassification`), 107.7 triệu tham số. Dùng bản `cased` vì chữ hoa là tín hiệu entity mạnh |
| Tokenizer | `bert-base-cased` (WordPiece, vocab 28 996), `is_split_into_words=True`, `max_length=128` (chỉ 2 câu test bị cắt) |
| Alignment | Strategy A (chính): subword đầu nhận nhãn word; subword sau, `[CLS]`, `[SEP]`, `[PAD]` → −100. Strategy B (so sánh): subword sau nhận I-X. Đánh giá ở mức word (dự đoán ở subword đầu) cho cả hai |
| Training | AdamW lr 3e-5, weight decay 0.01, warmup 10% rồi giảm tuyến tính, batch 16, 3 epoch, bf16 autocast, clip 1.0; chọn epoch có validation entity F1 cao nhất (epoch 3 cho cả A và B) |
| Baseline | `DictVectorizer` (133 786 feature) + `LogisticRegression(C=1, max_iter=300)`; dự đoán từng token độc lập |
| Metric | Entity-level P / R / F1 (khớp đúng biên + loại), tự viết bằng `bio_to_spans`; kiểm tra trùng với `seqeval` |
| Seed | 42 (1 lần chạy cho mỗi model) |
| Hardware | NVIDIA RTX 3050 Laptop (4 GB), PyTorch 2.6.0 + CUDA 12.4, transformers 5.17.0 |
| Thời gian | Baseline 35 s; BERT khoảng 115 s / epoch (A: 331 s cho 3 epoch) |

## Kết quả chính (test)

| Model | Entity P | Entity R | Entity F1 | Token accuracy |
|---|---|---|---|---|
| Baseline (Logistic Regression) | 0.742 | 0.778 | 0.760 | 0.958 |
| **BERT, Strategy A** | **0.905** | **0.916** | **0.911** | 0.983 |
| BERT, Strategy B | 0.907 | 0.922 | 0.914 | 0.983 |

| F1 (BERT A / baseline) | PER | ORG | LOC | MISC |
|---|---|---|---|---|
| | 0.959 / 0.809 | 0.893 / 0.644 | 0.929 / 0.834 | 0.803 / 0.739 |

| F1 theo độ dài entity (BERT A / baseline) | 1 word | 2 word | 3 word | 4+ word |
|---|---|---|---|---|
| | 0.908 / 0.737 | 0.924 / 0.835 | 0.869 / 0.630 | 0.851 / 0.473 |

Các phát hiện chính:

1. **Pretrained Transformer tăng entity F1 thêm 15 điểm** (0.760 → 0.911). Token accuracy chỉ tăng 2.5 điểm, vì 83% word là O.
2. **Entity chưa gặp trong train:**
   - BERT: F1 0.855 so với 0.962 cho entity đã gặp.
   - Baseline: 0.598 so với 0.914. Baseline chủ yếu nhớ tên.
3. **Entity từ 3 word trở lên khó hơn,** nhưng entity 1 word cũng không dễ nhất. Nhóm 2 word (phần lớn là tên người) có F1 cao nhất.
4. **MISC và ORG khó nhất:** MISC vì ít dữ liệu và không có mẫu chung; ORG vì trùng tên với LOC (đội thể thao mang tên thành phố).
5. **Lỗi nhiều nhất của BERT là sai loại (224), không phải bỏ sót (69).** 78% lỗi rơi vào entity chưa gặp trong train. Với từ mơ hồ, BERT đúng 86.1%, baseline 65.4%.
6. **Alignment:** Strategy B cao hơn A 0.4 điểm, chủ yếu ở entity dài và entity chưa gặp. Mức chênh này chưa đủ để kết luận vì chỉ chạy 1 seed.
7. **Information Extraction:** entity F1 0.941 trên 30 câu tin tức tự viết.

## Lưu ý

- **Bước train bị bỏ qua (NaN):**
  - Như LAB 07–08, GPU của máy này thỉnh thoảng cho ra NaN; bước có gradient không hữu hạn bị bỏ qua.
  - A: 67 / 2 634 bước (2.5%). B: 43 bước.
  - Một lần chạy còn bị dừng vì lỗi `CUDA error: an illegal memory access`. Vì vậy `train()` lưu checkpoint mỗi epoch trong `ckpt/` để chạy lại thì tiếp tục từ epoch đã xong.
- **Thời gian train của B (940 s) không so được với A:** epoch 1 của B chậm gấp 6 lần (749 s), vì model A vẫn nằm trên GPU 4 GB làm VRAM tràn sang RAM. Sau khi chuyển model không dùng về CPU, epoch 2 và 3 của B mất khoảng 110 s như A.
- **`ckpt/` và `model_ner/` không đưa lên git** (khoảng 400 MB mỗi model). Chạy lại `ner_finetuning.ipynb` để tạo lại.
- **Gold của 30 câu IE do mình tự gán nhãn** theo quy ước CoNLL (ví dụ chức danh không thuộc entity). Các câu này đầy đủ ngữ pháp, dễ hơn tiêu đề và bảng điểm trong CoNLL.

## Cách chạy lại

```bash
pip install torch transformers datasets scikit-learn seqeval pandas matplotlib nbconvert ipykernel
cd lab9
python baseline_ner.py                    # 6 test cho bio_to_spans + baseline (khoảng 1 phút)
python -m jupyter nbconvert --to notebook --execute --inplace tokenizer_alignment.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 ner_finetuning.ipynb   # khoảng 15 phút trên RTX 3050 Laptop; tạo model_ner/
python information_extraction.py          # đánh giá 30 câu
python information_extraction.py "Barack Obama visited Paris."   # trích xuất một đoạn văn bản
```

## AI assistance statement

- **AI-assisted code:** Claude Code.
- **What was generated:**
  - `baseline_ner.py` (features, `bio_to_spans`, metric, alignment);
  - code của 2 notebook (training loop, các bảng phân tích, biểu đồ);
  - `information_extraction.py` và 30 câu đánh giá;
  - bản nháp các file markdown;
  - ghép `report.pdf`.
- **What was modified:**
  - Thêm checkpoint theo epoch sau khi GPU bị lỗi `illegal memory access`.
  - Chuyển model không dùng về CPU sau khi phát hiện VRAM bị tràn làm train chậm 6 lần.
  - Train thêm Strategy B để có số liệu phân tích ảnh hưởng của alignment.
  - Thêm phân tích seen / unseen và subword để kiểm tra Prediction 3.
- **What was verified:**
  - 6 test cho `bio_to_spans` (span 1 token, nhiều token, I sau O, đổi loại, B liền B, ví dụ New York của đề);
  - entity F1 tự tính trùng với `seqeval` trên cả validation và test;
  - `cross_entropy(ignore_index=-100)` trùng với tính tay;
  - bài tính tay (BIO, shape) khớp output tokenizer và model;
  - số liệu trong các file `.md` lấy từ output notebook.
- **Student:** bài tính tay (`calculations9.pdf`), prediction, lựa chọn và lập luận alignment strategy, diễn giải metric, error analysis, reflection.
