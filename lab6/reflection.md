# Reflection — từ recurrent state đến attention

Số liệu lấy từ `self_attention.ipynb` / `results.csv` (IMDb test 25 000 review; cấu hình giống LAB 05, max_length 200).

| Model | Accuracy | F1 | Train time (6 epochs) | Params |
|---|---|---|---|---|
| LSTM (h<sub>T</sub>) | 0.8264 | 0.8177 | 42.1 s | 2 618 089 |
| Self-Attention, pooling trên H (ablation) | 0.8318 | 0.8222 | 12.7 s | 2 538 857 |
| **Self-Attention** (context = Attention(u, K, H)) | **0.8351** | 0.8259 | **12.5 s** | 2 538 857 |
| Self-Attention + positional embedding | 0.8304 | **0.8278** | 12.9 s | 2 558 857 |

LSTM không tất định trên GPU: 4 lần chạy cùng code cho accuracy 0.817–0.827. Ba model attention cho kết quả giống hệt nhau qua các lần chạy. Vì vậy chênh lệch khoảng 0.01 giữa LSTM và attention chưa đủ để nói model nào tốt hơn hẳn.

## 1. Đối chiếu prediction

| Prediction | Kết quả | |
|---|---|---|
| P1: score lớn hơn → weight lớn hơn theo hàm mũ | softmax([1, 2, 3]) = [0.090, 0.245, 0.665]; tỉ lệ 0.665 / 0.245 = e | Đúng |
| P2: score bằng nhau → đều 0.25 | Đúng (unit test) | Đúng |
| P3: [1, 1, 10] → khoảng 0.9998 | 0.99975 | Đúng |
| P4: 10 → 1 000 token thì attention matrix ×10 000 | 100 → 1 000 000 phần tử. Đo NumPy: T gấp đôi thì thời gian ×3.8–3.9 (1 000 → 2 000 → 4 000: 27.7 → 106.7 → 405.6 ms) | Đúng |
| P5: attention không luôn chọn đúng token quan trọng | ρ(β, ảnh hưởng thật) = 0.36; câu "…bad, but … excellent": β cao nhất ở *excellent* nhưng *bad* quyết định | Đúng |
| P6: T thấp → tập trung, T cao → phẳng | Query A: T = 0.5 / 1 / 2 → max weight 0.996 / 0.899 / 0.548, entropy 0.029 / 0.477 / 1.588 | Đúng |
| P7: Self-Attention ngang hoặc hơn LSTM, nhanh hơn, không nhạy thứ tự, yếu với phủ định | 0.835 so với 0.826; nhanh hơn 3.4 lần; đảo thứ tự từ chỉ đổi P khoảng 2·10<sup>−10</sup>; "Not good." → 0.984 positive | Đúng, nhưng em **không lường trước** việc model sai 5/6 câu ngắn tự viết |

**Bất ngờ nhất:** (1) Khi điểm pooling tính trên H, β gần như đều tuyệt đối (entropy / log L = 0.998) mà accuracy vẫn 0.832: attention có mặt trong model không có nghĩa là nó "tập trung" vào đâu. (2) Từ *like* đẩy câu "I like this movie." về negative (bỏ *like* thì P từ 0.028 lên 0.652). TF-IDF + LR cũng vậy, nên đây là thiên lệch của dữ liệu IMDb chứ không riêng của attention.

## 2. So sánh LAB 05 và LAB 06

| Property | RNN / LSTM / GRU | Attention (self-attention trong lab này) |
|---|---|---|
| Sequence order | Có: đọc tuần tự (đảo thứ tự từ làm P của LSTM đổi tới 0.38) | **Không** nếu không có positional encoding (đổi thứ tự: ΔP ≈ 2·10<sup>−10</sup>); có positional embedding thì ΔP tới 0.009 |
| Hidden state | Có, h<sub>t</sub> truyền qua từng bước | Không bắt buộc |
| Sequential computation | Có, T bước nối tiếp | Không bắt buộc: các phép nhân ma trận chạy song song |
| Direct token-to-token interaction | Hạn chế, phải đi qua chuỗi h<sub>t</sub> | Có: A[i, j] cho mọi cặp token |
| Long-range interaction | Khó (LAB 05: gradient tới token đầu của RNN chỉ còn khoảng 10<sup>−7</sup>) | Mọi cặp token cách nhau đúng 1 bước tính toán |
| Parallel computation | Hạn chế: 7.0 s / epoch | Dễ hơn: 2.1 s / epoch |
| Cost theo sequence length | Gần tuyến tính: T 800 → 1 600 thì thời gian 55 → 101 ms, bộ nhớ 319 → 451 MB | **O(T²)**: T 800 → 1 600 thì bộ nhớ 548 → 1 547 MB (×2.8, tiến tới ×4) |
| Context representation | Recurrent state h<sub>T</sub> | Weighted combination Σ β<sub>t</sub> h<sub>t</sub> |

## 3. Attention giải quyết vấn đề gì của recurrent processing?

- **Không phải nén cả sequence vào một h<sub>T</sub>:** context vector lấy trực tiếp từ mọi vị trí. Với review 200 token, token đầu và token cuối có vai trò như nhau trong phép tính, khác với LSTM nơi thông tin token đầu phải đi qua 200 bước.
- **Song song hóa:** train nhanh hơn LSTM 3.4 lần ở cùng số tham số; ở T = 1 600, một bước train vẫn nhanh hơn (78 ms so với 101 ms).
- **Linh hoạt theo query:** Exp 2 cho thấy cùng một câu, query sentiment chú ý vào *good* (0.90), query chủ đề phim chú ý vào *movie* (0.99), query thời gian chú ý vào *beginning* (0.95). Attention không phải một danh sách "từ quan trọng" cố định.

## 4. Attention tạo ra vấn đề mới nào?

- **Chi phí O(T²):** bộ nhớ GPU ở T = 1 600 là 1 547 MB so với 451 MB của LSTM; T = 10 000 cần 400 MB cho **một** attention matrix của một câu, một head. Attention không miễn phí và không nên tự động thay thế RNN cho mọi sequence dài.
- **Mất thứ tự:** không có positional encoding thì "dog bites man" = "man bites dog" (P giống hệt nhau), *not* không phủ định được *good*.
- **Trộn quá mức (oversmoothing):** hidden state sau self-attention trở nên giống nhau. Pooling trên H cho β gần như đều; value của một từ (*bad*) có thể lan sang nhiều token khác.
- **Dễ bị hiểu nhầm là lời giải thích:** heatmap của model đã train cho thấy *cat* chú ý vào *the* (0.33) và *is* (0.25), *bought* chú ý vào *that* (0.44): đó là pattern có ích cho việc phân loại sentiment chứ không phải quan hệ ngữ pháp. Ở Word2Vec với W = I, mỗi token chỉ chú ý vào chính nó (0.998), cho thấy phải có W<sub>Q</sub>, W<sub>K</sub> học được thì token mới tương tác với nhau.

## 5. Learning check (trả lời ngắn)

- **Attention score:** s<sub>i</sub> = q<sup>T</sup>k<sub>i</sub>, chia √d<sub>k</sub> trong scaled dot-product; đo độ khớp giữa câu hỏi (query) và nhãn của từng vị trí (key).
- **Vì sao dùng softmax:** biến score thành trọng số dương, tổng bằng 1, khả vi; khuếch đại score lớn theo hàm mũ.
- **Q, K, V:** Q là "đang tìm gì", K là "vị trí này có gì để được tìm thấy", V là "nội dung sẽ lấy ra". Tách K và V cho phép tiêu chí để chọn khác với thông tin được lấy (Bài 4: k<sub>1</sub>, k<sub>3</sub> được chọn, nhưng nội dung lấy ra là v<sub>1</sub>, v<sub>3</sub>).
- **Context vector:** c = Σ α<sub>i</sub> v<sub>i</sub>, trung bình có trọng số của các value.
- **Vì sao chia √d<sub>k</sub>:** phương sai của q·k tăng theo d<sub>k</sub> (độ lệch chuẩn 2.0 ở d<sub>k</sub> = 4, 32 ở d<sub>k</sub> = 1 024). Không chia thì softmax gần one-hot (max weight 0.97) và gradient qua softmax nhỏ (‖J‖ = 0.047); có chia thì max weight khoảng 0.32 và ‖J‖ khoảng 0.35 với mọi d<sub>k</sub>. Chia √d<sub>k</sub> tương đương temperature T = √d<sub>k</sub>.
- **Self-attention khác attention thông thường:** Q, K, V đều lấy từ cùng một sequence (XW<sub>Q</sub>, XW<sub>K</sub>, XW<sub>V</sub>), nên mỗi token vừa hỏi vừa được hỏi.
- **Sequence dài gấp đôi:** attention matrix gấp 4 lần.
- **Attention có chứng minh model đang reasoning không?** Không. Nó chỉ là phân phối trọng số; ở đây ρ(β, ảnh hưởng thật) = 0.36, và có trường hợp attention nhìn đúng từ nhưng model vẫn sai.
- **Vì sao xử lý long-range tốt hơn recurrent state:** đường đi giữa hai token bất kỳ chỉ có 1 bước (một phép nhân QK<sup>T</sup>), không phải đi qua T bước và T lần nhân Jacobian như RNN.

## 6. Hướng tới LAB 07

Các vấn đề quan sát được ở lab này gợi ý trực tiếp các thành phần của Transformer:

- **Positional encoding** để có lại thứ tự (đã thử một phiên bản đơn giản: đảo thứ tự bắt đầu làm P thay đổi).
- **Multi-head:** nhiều query khác nhau nhìn vào những phần khác nhau (như Exp 2).
- **Residual + LayerNorm** để token giữ lại danh tính, tránh trộn quá mức.
- Các biến thể attention hiệu quả hơn O(T²) cho sequence rất dài.
