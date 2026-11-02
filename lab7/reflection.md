# Reflection — từ Attention đến Transformer

Số liệu lấy từ `transformer_classification.ipynb` và `results.csv`: IMDb test 25 000 review, max_length 200. Kết quả là trung bình ± độ lệch chuẩn của 3 seed (42, 43, 44) nếu không ghi khác.

| Model | Params | Train time (6 epoch) | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| LSTM | 3 332 481 | 60.7 s (10.1 s/epoch) | 0.825 ± 0.009 | 0.845 | 0.798 | 0.820 ± 0.013 |
| **Transformer** (4 head, mean, PE) | 3 332 865 | **41.8 s** (7.0 s/epoch) | **0.835 ± 0.004** | 0.841 | 0.828 | **0.834 ± 0.008** |
| Transformer không PE | 3 332 865 | 41.4 s | 0.838 ± 0.007 | 0.849 | 0.824 | 0.835 ± 0.012 |
| Transformer, `nn.TransformerEncoderLayer` (1 seed) | 3 332 865 | 46.6 s | 0.833 | 0.854 | 0.804 | 0.828 |

Hai model gần như cùng số tham số, phần lớn là embedding 25 002 × 128. Block tự ghép cho kết quả ngang lớp có sẵn của PyTorch, và trùng output khi chép trọng số (test 5 trong `transformer_block.py`).

## 1. Prediction verification

| # | Dự đoán | Kết quả | Đúng/Sai | Giải thích |
|---|---|---|---|---|
| 1 | Transformer song song hơn, một epoch nhanh hơn LSTM | 7.0 so với 10.1 s/epoch (nhanh hơn 1.45 lần) ở T = 200. Một bước train nhanh hơn ở mọi T ≤ 256 | **Đúng** | Mọi vị trí tính cùng lúc bằng phép nhân ma trận; LSTM phải đi 200 bước nối tiếp |
| 2 | Bỏ PE thì không biết thứ tự | Đảo thứ tự từ làm P của model không PE đổi tối đa 8.9·10<sup>−8</sup> (sai số float), so với 0.115 khi có PE và 0.40 với LSTM | **Đúng** | Mọi phép tính trong block đều equivariant với hoán vị (test 7) |
| 3 | "dog bites man" = "man bites dog" khi không PE | Block ngẫu nhiên: ‖mean(Z<sub>1</sub>) − mean(Z<sub>2</sub>)‖ = 0, ‖z(dog)<sub>S1</sub> − z(dog)<sub>S2</sub>‖ = 0. Có PE: lần lượt 0.278 và 1.97. Model đã train không PE: P = 0.3169 cho cả hai câu | **Đúng** | Không PE thì *dog* ở vị trí 0 hay 2 đều có cùng q, k, v và cùng tập key, nên cùng output |
| 4 | Các head không nhất thiết giống nhau, nhưng **một số head sẽ gần như redundant** | TV giữa các cặp head: 0.61–0.66 (4 head), 0.59–0.79 (8 head), so với 0.24–0.26 lúc khởi tạo. Không có cặp nào gần nhau | **Sai một phần** | Phần "không giống nhau" đúng. Phần "redundant" sai: sau khi train các head **khác nhau nhiều hơn** lúc khởi tạo. Có thể vì mỗi head chỉ có d<sub>k</sub> = 32 chiều, và gradient đẩy mỗi head về những từ cảm xúc khác nhau (ví dụ trong hình: head 4 dồn vào *boring*, head 1 vào *not / plot*). Em đã nghĩ task đơn giản thì các head sẽ học trùng, nhưng "khác nhau" không có nghĩa là "cần thiết": 1 head vẫn đạt 0.831 |
| 5 | Tăng số head không chắc tăng accuracy; chênh lệch ±0.01, có thể giảm ở 8 head | 1 / 2 / 4 / 8 head: 0.831 / 0.835 / 0.835 / 0.837 (std khoảng 0.004) | **Đúng** (phần "giảm ở 8 head" sai) | Chênh lệch ≤ 0.006, cỡ 1–1.5 std, nên không kết luận được. 8 head còn nhỉnh nhất. Điều em **không** lường trước: thời gian train tăng theo số head (27 → 60 s), vì có h ma trận T × T nhỏ, và softmax / dropout trên (B, h, T, T) tốn bộ nhớ nhiều gấp h lần |
| 6 | Transformer không luôn nhanh hơn; ngang nhau ở T ngắn, chậm hơn và tốn bộ nhớ hơn ở T dài | ms/bước, Transformer so với LSTM: T = 32: 8.2 / 9.5; T = 256: 26.7 / 29.1; **T = 512: 79.8 / 56.4**; **T = 1 024: 1 123 / 98** (bộ nhớ 4 476 MB vượt 4 GB VRAM); T = 2 048: hết bộ nhớ | **Đúng** | Điểm hòa vốn khoảng T ≈ 300. Sau đó O(T<sup>2</sup>) thắng lợi thế song song |

## 2. Trả lời các câu hỏi trong experiment

**Exp 1 — Không có positional information, tại sao thứ tự thành vấn đề?**

- Self-attention tính score q<sub>i</sub>·k<sub>j</sub> chỉ dựa trên **nội dung** của hai token. Output z<sub>i</sub> = Σ<sub>j</sub> α<sub>ij</sub>v<sub>j</sub> là tổng trên một **tập** key, không có khái niệm "trước / sau".
- Đảo câu chỉ làm hoán vị hàng và cột của A (đã kiểm tra: A(S<sub>2</sub>) = P·A(S<sub>1</sub>)·P<sup>T</sup>). Output của mỗi từ giữ nguyên, chỉ đổi chỗ.
- FFN và LayerNorm áp dụng từng vị trí riêng, nên cũng không thêm thông tin vị trí. Sau mean pooling, hai câu là một vector giống hệt.
- Kết quả: "dog bites man" và "man bites dog", hay "not good… bad" và "bad… not good", là cùng một input với model.

**Exp 2 — Tăng head có tăng representation dimension không? d<sub>k</sub> thay đổi thế nào?**

Không. d<sub>model</sub> = 128 giữ nguyên, số tham số giữ nguyên (3 332 865 cho cả 4 cấu hình). Chỉ có **d<sub>k</sub> = 128/h giảm**: 128, 64, 32, 16. Nhiều head hơn nghĩa là nhiều phân phối attention độc lập hơn, nhưng mỗi phân phối so khớp q và k trong không gian ít chiều hơn.

**Exp 3 — Multi-head có thực sự tạo ra pattern khác nhau không?**

**Có**, sau khi train:

- **Các head khác nhau rõ rệt:** TV trung bình giữa hai head là 0.63, so với 0.25 lúc khởi tạo.
- **Head nào tập trung, head nào phân tán:**
  - lúc khởi tạo mọi head gần như đều (entropy / log L ≈ 0.98, max weight ≈ 0.02);
  - sau khi train: head 3 tập trung nhất (max weight 0.21, entropy 0.73), head 4 phân tán nhất trên 500 review (0.82);
  - trong câu ví dụ, head 4 lại dồn gần như mọi query vào *boring* (entropy 0.51), còn head 3 gần đều (0.90).
- **Không có head nào redundant:** cặp gần nhau nhất có TV = 0.61 (4 head) và 0.59 (8 head).
- **Không head nào học pattern vị trí rõ ràng:**
  - khối lượng attention vào chính token và vào hai token kề bên chỉ khoảng 0.01 mỗi loại, gần mức đều;
  - khoảng cách trung bình |i − j| ≈ 55, bằng lúc khởi tạo.
  - Các head phân biệt nhau theo **nội dung** (từ nào được nhìn), không theo vị trí.
- Em **không** đặt tên kiểu "head 4 = sentiment head": chỉ một câu ví dụ thì chưa đủ bằng chứng.

**Exp 4 — Khác biệt performance đến từ thành phần nào?**

Transformer hơn LSTM khoảng 0.01 accuracy và ổn định hơn giữa các seed (std 0.004 so với 0.009). Các thành phần có thể giải thích:

- **Mean pooling trên mọi token**, thay vì chỉ dùng h<sub>T</sub>: mọi từ cảm xúc ở bất kỳ đâu trong 200 token đều đóng góp trực tiếp. LSTM phải giữ thông tin qua 200 bước, nên dễ thiên về cuối câu.
- **Self-attention**: token tương tác trực tiếp trong 1 bước.
- **Residual**: embedding của từ đi thẳng tới pooling. Model có thể hoạt động gần như một "bag of embeddings" có trọng số.

Ablation quan trọng là **bỏ PE không làm giảm accuracy** (0.838 so với 0.835). Trên IMDb với một lớp, phần lớn tín hiệu là **có những từ nào**, không phải thứ tự. Đây cũng là lý do TF-IDF + LR (LAB 04) đạt 0.881, cao hơn mọi model ở đây. Thứ tự chỉ quan trọng ở một số ít câu: có PE thì "This movie is not good." đúng (P = 0.38), không PE thì sai (0.90). Số câu như vậy quá ít để thấy trên accuracy trung bình.

**Exp 5 — Vì sao không đơn giản tăng sequence length?**

| max_length | Tokens retained | Attention matrix / lớp (64 × 4 × T<sup>2</sup>) | Peak GPU (train) | Train time | Accuracy | F1 |
|---|---|---|---|---|---|---|
| 32 | 14% | 1 MB | 280 MB | 14.9 s | 0.699 | 0.704 |
| 64 | 28% | 4 MB | 332 MB | 19.7 s | 0.753 | 0.776 |
| 128 | 52% | 16 MB | 484 MB | 24.5 s | 0.805 | 0.796 |
| 256 | 78% | 64 MB | 981 MB | 54.1 s | 0.846 | 0.847 |

Tăng T giúp rõ rệt, vì giữ được nhiều review hơn và 49% lỗi đến từ review bị cắt. Nhưng chi phí tăng theo T<sup>2</sup>:

- T 128 → 256: attention matrix ×4, peak memory ×2.0, thời gian ×2.2.
- Benchmark một bước train: T = 512 đã chậm hơn LSTM; T = 1 024 tốn 4.5 GB (vượt VRAM 4 GB, chậm hơn LSTM 11 lần); T = 2 048 không chạy được.
- Phần lớn review IMDb dưới 400 token, nhưng văn bản thật (tài liệu, sách, hội thoại dài) dài hơn nhiều.
- Muốn tăng T phải trả theo bình phương, hoặc dùng các biến thể attention hiệu quả (sparse, sliding window, linear attention), hoặc chia văn bản thành đoạn.

**Exp 6 — Cách lấy sequence representation có ảnh hưởng không?**

Gần như không ở thí nghiệm này. Mean: 0.835 ± 0.004, F1 0.834; CLS: 0.837 ± 0.004, F1 0.834. Khác biệt nằm ở **quá trình học**:

- CLS học chậm hơn: validation accuracy ở epoch 1 là 0.768 so với 0.800; best epoch 2–4 (trung bình 3) so với 2–3 (trung bình 2.3).
- Lý do: vector [CLS] ban đầu không chứa nội dung, phải học cách "hỏi" các token qua attention. Mean pooling thì dùng được ngay mọi token.
- Với một lớp, [CLS] chỉ nhìn các token **một lần**. Ở BERT (12 lớp, pretrain) [CLS] có nhiều lớp để tổng hợp.

## 3. Learning curves

- **Transformer học nhanh và overfit sớm:**
  - validation loss thấp nhất ở epoch 2 (0.357);
  - sau đó train loss giảm tới 0.094 nhưng validation loss tăng lên 0.562 ở epoch 6, trong khi validation accuracy gần như đứng yên (0.84–0.85);
  - model trở nên **quá tự tin** hơn là sai nhiều hơn.
  - Chọn epoch theo validation loss tránh được phần overfit này. Bản không PE giống hệt (best epoch 2, val loss lên 0.628).
- **LSTM hội tụ chậm hơn:** best epoch 4–5, val loss 0.399. Ít overfit hơn trong 6 epoch.
- **CLS:** học chậm hơn mean pooling, đường val loss không đơn điệu (0.350 ở epoch 4, 0.432 ở epoch 5).
- **Instability:** 9 bước trong khoảng 48 000 bước train (mọi model) bị bỏ qua vì GPU trả về NaN. Lỗi này xảy ra cả với `nn.TransformerEncoderLayer` của PyTorch (2 bước), nên không phải do code của block. Không có loss spike nào do model.

## 4. So sánh LAB 04 → LAB 07

| Property | TF-IDF | Avg Word2Vec | LSTM | Attention (LAB 06) | Transformer |
|---|---|---|---|---|---|
| Sparse | ✓ | ✗ | ✗ | ✗ | ✗ |
| Dense | ✗ | ✓ | ✓ | ✓ | ✓ |
| Word order | Limited (chỉ n-gram) | ✗ | ✓ | ✗ (không PE) | ✓* |
| Context-dependent | ✗ | ✗ | ✓ | ✓ | ✓ |
| Direct token interaction | ✗ | ✗ | Limited | ✓ | ✓ |
| Sequential computation | ✗ | ✗ | ✓ | ✗ | ✗ |
| Parallelizable | ✓ | ✓ | Limited | ✓ | ✓ |
| Long-range interaction | Limited | ✗ | Difficult | Improved | Improved |
| Attention | ✗ | ✗ | ✗ | ✓ | ✓ |
| Positional information | — | — | Implicit (thứ tự đọc) | Required | Required |
| Cost with sequence length | ~linear | ~linear | ~linear (tuần tự) | O(T<sup>2</sup>) | O(T<sup>2</sup>) |
| IMDb accuracy (các lab) | 0.881 (LR) | 0.843 | 0.816 (LAB 05) / 0.825 | 0.835 | 0.835 |

\* Transformer cần positional information để biểu diễn thứ tự (Exp 1, permutation).

**Mỗi bước giải quyết hạn chế nào của bước trước:**

```
TF-IDF        → Limitation: sparse, mỗi từ là một chiều độc lập; "good" và "great" không liên quan;
                không có ngữ cảnh; thứ tự chỉ có qua n-gram (vocabulary bùng nổ).
Word2Vec      → giải quyết: vector dense, từ gần nghĩa có vector gần nhau.
                Limitation: một từ một vector bất kể ngữ cảnh; trung bình các vector làm mất thứ tự và loãng tín hiệu.
LSTM          → giải quyết: đọc theo thứ tự, h_t phụ thuộc ngữ cảnh phía trước.
                Limitation: tuần tự (không song song); nén cả câu vào một h_T; thông tin xa phải đi qua nhiều bước.
Attention     → giải quyết: mọi token tương tác trực tiếp, context vector lấy từ mọi vị trí, tính song song.
                Limitation: không có thứ tự; một lớp attention đơn lẻ dễ làm token giống nhau; O(T²).
Transformer   → giải quyết: positional encoding (thứ tự), multi-head (nhiều kiểu quan hệ),
                residual + LayerNorm (giữ thông tin, train ổn định), FFN (phi tuyến từng token), xếp chồng được.
                Remaining limitation: chi phí O(T²) nên phải cắt văn bản; học từ đầu trên 20k review thì
                không hơn TF-IDF (0.835 so với 0.881) và vẫn sai negation / contrast; cần rất nhiều dữ liệu
                để học được ngôn ngữ → động lực cho pretraining (LAB 08).
```

## 5. Individual learning check (trả lời ngắn)

- **Tại sao cần positional encoding?** Self-attention, FFN và LayerNorm đều equivariant với hoán vị. Không có PE, model thấy câu như một **tập** từ ("dog bites man" = "man bites dog", ΔP = 9·10<sup>−8</sup>). PE cộng vào mỗi vị trí một vector khác nhau, nên cùng một từ ở vị trí khác nhau có input khác nhau.
- **Multi-head khác một head ở đâu?** h head có W<sub>Q</sub>, W<sub>K</sub>, W<sub>V</sub> riêng, mỗi head d<sub>k</sub> = d<sub>model</sub>/h chiều, cho h phân phối attention khác nhau trên cùng câu. Kết quả được nối lại rồi trộn bằng W<sub>O</sub>. Một head chỉ có một phân phối cho mỗi query. Cùng số tham số.
- **d<sub>model</sub> = 128, 8 head:** d<sub>k</sub> = 16.
- **Vì sao attention matrix là T × T?** T query nhân với T key thì có T<sup>2</sup> score: mỗi token có một trọng số cho mọi token.
- **Residual connection:** y = x + F(x). Giữ thông tin gốc (F chỉ học phần sửa), tạo đường thẳng cho gradient (∂y/∂x = I + ∂F/∂x), giúp train sâu ổn định.
- **FFN làm gì?** Biến đổi phi tuyến (Linear → ReLU → Linear) **tại từng vị trí**, cùng trọng số cho mọi token. Attention trộn thông tin giữa các token, FFN xử lý thông tin trong từng token.
- **Vì sao Transformer song song hơn RNN?** Trong một lớp, output của mọi vị trí chỉ phụ thuộc vào input của lớp, không phụ thuộc output của vị trí trước, nên tính cùng lúc bằng phép nhân ma trận. RNN cần h<sub>t−1</sub> để tính h<sub>t</sub>.
- **Sequence dài gấp đôi:** attention matrix gấp 4 (128<sup>2</sup> = 16 384 → 256<sup>2</sup> = 65 536).
- **Vì sao attention cao ≠ "lý do" prediction?** Attention chỉ cho biết token nào được nhìn, không cho biết nó đẩy prediction theo hướng nào hay mạnh bao nhiêu (còn value, W<sub>O</sub>, FFN, classifier). Ở đây ρ(attention, ảnh hưởng thật) chỉ 0.46. Review 6577: attention nhìn đúng *mediocre* lẫn *superb* mà model vẫn sai.
- **LayerNorm:** chuẩn hóa mỗi vector token về trung bình 0, phương sai 1 theo chiều feature (rồi nhân γ, cộng β), giữ thang đo ổn định sau mỗi phép cộng residual.

## 6. Hướng tới LAB 08

Transformer ở đây học từ đầu trên khoảng 20 000 review, nên mỗi từ chỉ biết qua những review đó:

- "not bad at all" vẫn sai;
- *well, bad day* bị hiểu theo nghĩa đen;
- accuracy không vượt được TF-IDF.

Kiến trúc đã đủ (attention, PE, FFN, residual, LayerNorm, có thể xếp chồng), cái thiếu là **dữ liệu để học ngôn ngữ**. Pretrain một Transformer lớn trên lượng văn bản khổng lồ rồi fine-tune cho IMDb (BERT) là bước tự nhiên tiếp theo. Sau lab này em biết [CLS], positional embedding, các head, residual và LayerNorm nằm ở đâu trong BERT.
