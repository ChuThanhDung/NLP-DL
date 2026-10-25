# LAB 06 — Attention Mechanism

Bài toán: tự cài đặt attention (score → softmax → context vector), scaled dot-product và self-attention bằng NumPy; quan sát attention theo query, temperature và scaling; dùng self-attention làm classifier sentiment trên IMDb và so sánh với LSTM của LAB 05.
Pipeline: `Q, K, V → QKᵀ / √d_k → softmax → weights → weights · V → context vector → classifier`.

## Cấu trúc thư mục

| File                                                                                                                                                                           | Nội dung                                                                                                                                                                                                                               |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `report.pdf`                                                                                                                                                                   | Bản tổng hợp: README, calculations, prediction, self_attention.ipynb, error_analysis, reflection                                                                                                                                       |
| `calculations.md`                                                                                                                                                              | Bài 0–9: score, softmax, context vector, toàn bộ attention, scaling, temperature, numerical stability, self-attention (ma trận tự chọn vì đề không cho số), chi phí O(T²)                                                              |
| `prediction.md`                                                                                                                                                                | 7 dự đoán trước khi chạy code                                                                                                                                                                                                          |
| `attention_numpy.py`                                                                                                                                                           | `softmax` ổn định số, `scaled_dot_product_attention` (trả về scores, attention_weights, context; có mask, temperature), `self_attention`, `entropy` + 8 unit tests (có so khớp với `torch.nn.functional.scaled_dot_product_attention`) |
| `self_attention.ipynb`                                                                                                                                                         | Toàn bộ thí nghiệm (đã chạy, có output)                                                                                                                                                                                                |
| `results.csv`                                                                                                                                                                  | Mọi số liệu (dạng dài: experiment, setting, metric, value)                                                                                                                                                                             |
| `images/attention_heatmap.png`                                                                                                                                                 | Exp 3: attention matrix của 2 câu, Word2Vec (W = I) so với model đã train                                                                                                                                                              |
| `images/confusion_matrix.png`                                                                                                                                                  | Confusion matrix của 4 model trên IMDb test                                                                                                                                                                                            |
| `error_analysis.md`                                                                                                                                                            | 3 mẫu đúng, 4 mẫu sai, 6 câu application, kiểm chứng attention bằng occlusion                                                                                                                                                          |
| `reflection.md`                                                                                                                                                                | Đối chiếu prediction, bảng so sánh LAB 05 và LAB 06, learning check                                                                                                                                                                    |
| `images/` (các ảnh còn lại: `self_attention_exercise.png`, `attention_queries.png`, `scaling_temperature.png`, `learning_curves.png`, `cost.png`, `application_attention.png`) | Hình bổ trợ                                                                                                                                                                                                                            |

## Thiết lập

- **Exp 1–2 (query giả lập):** key/value = word vector Word2Vec CBOW 100 chiều train trên text8 (như LAB 03, 1 luồng + hàm hash cố định nên chạy lại ra cùng kết quả), mỗi vector chuẩn hóa về độ dài √d<sub>k</sub>. Query = trung bình vector của các từ **không có trong câu** (sentiment: _excellent, terrible, …_; chủ đề phim: _film, cinema, …_; thời gian: _end, start, …_).
- **IMDb (Exp 4, application, error analysis):** giống LAB 05. Train 19 923 / validation 4 981 / test 25 000; vocabulary 25 000 từ; embedding 100 chiều train từ đầu; max_length 200; batch 64; Adam 1e-3; 6 epochs; clip 1.0; dropout 0.3; chọn epoch có validation loss thấp nhất.
- **Model Self-Attention (tự viết, không dùng `nn.MultiheadAttention`):** Embedding → Q, K, V (d = 128) → H = softmax(QKᵀ/√d)V (mask padding) → context c = Attention(u, K, H) với query học được u → Linear. Lớp attention trong PyTorch khớp với hàm NumPy (sai số 8·10<sup>−17</sup>).
- **Phần cứng:** GPU NVIDIA RTX 3050 Laptop, PyTorch 2.6 + CUDA 12.4.

## Kết quả chính

| Model (IMDb test)                         | Accuracy   | Precision | Recall | F1         | Train time | Params    |
| ----------------------------------------- | ---------- | --------- | ------ | ---------- | ---------- | --------- |
| LSTM (h<sub>T</sub>)                      | 0.8264     | 0.8612    | 0.7783 | 0.8177     | 42.1 s     | 2 618 089 |
| Self-Attention, pooling trên H (ablation) | 0.8318     | 0.8718    | 0.7779 | 0.8222     | 12.7 s     | 2 538 857 |
| **Self-Attention**                        | **0.8351** | 0.8746    | 0.7824 | 0.8259     | **12.5 s** | 2 538 857 |
| Self-Attention + positional emb.          | 0.8304     | 0.8409    | 0.8150 | **0.8278** | 12.9 s     | 2 558 857 |

LSTM không tất định trên GPU (4 lần chạy: accuracy 0.817–0.827); các model attention cho kết quả giống hệt nhau mỗi lần chạy. Tham khảo: TF-IDF + LR đạt 0.887.

Các phát hiện chính:

1. **Attention phụ thuộc query:** cùng câu "The movie was surprisingly good despite a very slow beginning", query sentiment chú ý vào _good_ (0.90), query phim ảnh vào _movie_ (0.99), query thời gian vào _beginning_ (0.95).
2. **Scaling = temperature √d<sub>k</sub>:** không chia √d<sub>k</sub> thì ở d<sub>k</sub> = 1 024 softmax gần one-hot (max weight 0.97) và gradient qua softmax nhỏ (‖J‖ = 0.047); có chia thì max weight khoảng 0.32, ‖J‖ khoảng 0.35 ở mọi d<sub>k</sub>.
3. **Self-attention không có thứ tự:** đảo thứ tự từ làm P(pos) đổi khoảng 2·10<sup>−10</sup> (LSTM: tới 0.38), nên "Not good." bị đoán positive (0.984).
4. **Chi phí O(T²):** T gấp đôi thì thời gian NumPy ×3.8–3.9. Ở T = 1 600 (batch 32), bộ nhớ GPU của attention là 1 547 MB so với 451 MB của LSTM.
5. **Attention ≠ giải thích:** ρ(β, ảnh hưởng thật khi bỏ token) = 0.36. Bỏ 5 token β cao nhất làm P đổi gấp 4.7 lần bỏ 5 token ngẫu nhiên, nhưng chỉ 30% review có token β cao nhất trùng token ảnh hưởng nhất. Pooling trên H cho β gần như đều mà accuracy vẫn 0.832.

## Cách chạy lại

```bash
pip install numpy torch gensim datasets scikit-learn scipy pandas matplotlib nbconvert ipykernel
cd lab6
python attention_numpy.py                                                   # 8 unit tests
jupyter nbconvert --to notebook --execute --inplace self_attention.ipynb    # khoảng 6–8 phút trên GPU; cần text8 (gensim-data) và IMDb
```

## AI assistance statement

- **Tool:** Claude Code
- **Purpose:** viết code, tạo biểu đồ và `results.csv`; trình bày các file markdown và ghép `report.pdf`.
- **Generated:** code, biểu đồ, `results.csv`, `report.pdf`, bản nháp các file markdown.
- **Modified:** phiên bản đầu của classifier tính điểm pooling trên H và cho β gần như đều; đã đổi sang tính trên K (giữ bản cũ làm ablation). Thêm occlusion để kiểm chứng attention thay vì chỉ đọc heatmap.
- **Student:** bài tính tay `calculations.pdf`; rà soát prediction, cách diễn giải heatmap, nguyên nhân lỗi và reflection.
- **Verified:** 8 unit tests pass; bài tính tay khớp với `attention_numpy.py`; hàm NumPy khớp `torch.nn.functional.scaled_dot_product_attention` (sai số < 10<sup>−12</sup>); số liệu trong các file `.md` lấy từ output notebook.
