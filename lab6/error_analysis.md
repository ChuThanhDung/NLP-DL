# Error analysis — Self-Attention classifier trên IMDb

Model phân tích: **Self-Attention** (model chính ở Exp 4, context = Attention(u, K, H)), test accuracy 0.8351, F1 0.8259. Số liệu lấy từ mục *Application* và *Error analysis* của `self_attention.ipynb`.

**Cách kiểm tra attention.** Với mỗi review, em ghi các token mà em cho là quan trọng, rồi xem:

- **β**: trọng số attention pooling của model trên từng token;
- **ΔP**: P(pos) thay đổi bao nhiêu khi bỏ (mask) đúng token đó (occlusion), ΔP = P(gốc) − P(bỏ token).

"Attention cao → token quyết định prediction" chỉ là **giả thuyết**; ΔP dùng để đối chiếu với hành vi thật của model. Lưu ý: review dài 200 token thì β đều là 1/200 = 0.005, nên β = 0.02 đã là gấp 4 lần mức đều.

## 1. Tổng quan

| | Pred Neg | Pred Pos |
|---|---|---|
| **Thật Neg** | 11 098 | 1 402 |
| **Thật Pos** | 2 720 | 9 780 |

- Model bỏ sót positive nhiều hơn báo nhầm (FN 2 720 so với FP 1 402): recall 0.782, precision 0.875.
- Accuracy giảm theo độ dài: 0–100 token 0.864; 100–200 token 0.858; 200–400 token 0.817; trên 400 token 0.759. 49.1% lỗi là review dài hơn 200 token (bị cắt), trong khi loại này chiếm 40.3% test.
- LSTM đoán đúng 35.1% trong số 4 122 review mà Self-Attention đoán sai, tức hai model mắc lỗi khác nhau một phần.

## 2. Ba mẫu đúng (chọn ngẫu nhiên)

| idx | True / Pred | P(pos) | Token em cho là quan trọng | Top β | ΔP khi bỏ | Attention có tập trung vào token đó? |
|---|---|---|---|---|---|---|
| 7323 | neg / neg | 0.002 | *horrible, worst, flawed* | horrible 0.02, horrible 0.02, worst 0.02, ridiculous 0.02 | ≈ 0 với mọi token | **Có.** Nhưng P đã bão hòa ở 0.002 nên bỏ một từ không đủ thay đổi; occlusion một token không kiểm chứng được |
| 175 | neg / neg | 0.020 | *disappointing, less tension* | disappointing 0.04 (cao nhất) | disappointing −0.03 (lớn nhất) | **Có**, và occlusion xác nhận: token β cao nhất cũng là token ảnh hưởng nhất |
| 1015 | neg / neg | 0.030 | *worst* | worst 0.02 (cao nhất) | worst −0.06 (lớn nhất) | **Có**, occlusion xác nhận |

Với ba mẫu đúng, attention trùng với trực giác của em và (khi P chưa bão hòa) trùng với token có ảnh hưởng lớn nhất. Đây là những review có từ cảm xúc mạnh, rõ ràng.

## 3. Bốn mẫu sai

| idx | True → Pred | P(pos) | Loại lỗi | Phân tích |
|---|---|---|---|---|
| 8658 | neg → pos | 1.000 | **ambiguous wording** | Toàn văn chỉ 8 từ: "More suspenseful, more subtle, much, much more disturbing…." Đây có lẽ là câu so sánh với một phim khác, không có từ chê trực tiếp. β dồn vào *suspenseful* (0.17) và *subtle* (0.15), là các từ thường đi với lời khen. Giả thuyết: attention vào từ khen dẫn tới positive. Không kiểm chứng được bằng occlusion vì P bão hòa ở 1.000 (mọi ΔP ≈ 0). Ngay cả người đọc cũng cần ngữ cảnh mới hiểu được, nên lỗi không phải do sequence modeling |
| 4118 | neg → pos | 0.999 | **noise** (nhãn) | "…one of the best Kung fu movies in the history of Cinema… The final fight… is a masterpiece!" Toàn bộ review là lời khen, nên nhãn negative nhiều khả năng sai. Thú vị là β không dồn vào *best / masterpiece / great* (top β là *case, actor, which* ≈ 0.03–0.04) mà prediction vẫn đúng theo nội dung. Như vậy prediction không đòi hỏi attention phải "nhìn đúng chỗ" |
| 15270 | pos → neg | 0.184 | **ambiguous wording + long-range** | Review mở đầu bằng "my family… returned home thoroughly disappointed… a warning to all 'conclusive story lovers' to stay away" (mỉa mai), sau đó khen ý nghĩa phim qua các câu hỏi tu từ chứa *monster, desperately, neglected, insanely*. Occlusion: *becoming* −0.07, *desperately* −0.05; token β cao nhất (*becoming*) trùng token có \|ΔP\| lớn nhất. Ở đây attention và occlusion khớp nhau, nhưng cả hai cho thấy model đọc nghĩa đen từng từ, không nắm được thái độ của người viết |
| 12990 | pos → neg | 0.219 | **rare usage + truncation** | 411 từ, bị cắt ở 200 token. Phần đầu là ghi chú về nhầm lẫn TV series ("My mistake… loosely based…") và cụm *terribly sweet* (terribly = "rất"). Occlusion: *terribly* −0.06, *loosely* −0.06, tức hai từ này đẩy về negative theo nghĩa thường gặp. Lời khen rõ nhất ("captured my heart and imagination") nằm sau token 200 nên bị cắt mất |

## 4. Application — câu tự viết

| Câu | P(pos) | Kết quả | Top β | Top \|ΔP\| (bỏ token) | Phân tích |
|---|---|---|---|---|---|
| I like this movie. | 0.028 | ✗ (negative) | this 0.38, like 0.21, i 0.21 | **like −0.624**, movie +0.028 | Bỏ *like* thì P lên 0.652: chính *like* kéo về negative. TF-IDF + LR cũng sai (0.176), hệ số LR: *like* −0.52, *this* −1.04, *movie* −0.89. Trong IMDb, *like* hay xuất hiện trong "looks like", "like a…" ở review chê. Lỗi do **thiên lệch từ vựng của dữ liệu + domain shift** (câu ngắn so với review dài); β không chỉ ra nguyên nhân (*like* chỉ đứng thứ 2) |
| I do not like this movie. | 0.000 | ✓ | do 0.31, this 0.21, not 0.12 | do −0.009; bỏ *not* thay đổi < 0.001 | Đúng nhưng **không vì hiểu phủ định**: câu không có *not* ở trên cũng bị đoán negative, và bỏ *not* gần như không đổi gì |
| I thought this movie would be bad, but it was actually excellent. | 0.311 | ✗ | **excellent 0.18**, actually 0.12, thought 0.11 | **bad −0.681**, excellent +0.297, would −0.224 | β dồn vào đúng token em cho là quan trọng (*excellent*), nhưng model vẫn sai. Attention matrix: *movie, would, be, was* chú ý gần như toàn bộ vào *bad*, nên value của *bad* lan vào nhiều hidden state. Bỏ *bad* thì P = 0.992. Loại lỗi: **contrast / long-range**. LSTM đúng (0.561) |
| The movie was extremely long and the acting was sometimes boring, but the final scene was absolutely fantastic. | 0.001 | ✗ (theo ý câu trong đề: kết luận ở cuối là khen) | boring 0.19, acting 0.14, absolutely 0.10 | **boring −0.617** | Attention và occlusion khớp: model dựa vào *boring* và bỏ qua "but … fantastic". Phép trộn không có thứ tự không biết *but* đảo trọng tâm câu |
| This movie is not bad at all. | 0.002 | ✗ | this 0.24, all 0.18, at 0.18 | bad −0.175, not −0.020 | **Negation**: *not* không đảo được *bad* |
| Not good. | 0.984 | ✗ | good 0.65, **not 0.35** | good +0.984, not −0.016 | **Negation**: attention *có* nhìn vào *not* (0.35) nhưng *not* gần như không ảnh hưởng. Mỗi token chỉ được cộng có trọng số, không có phép "not phủ định good" và không có thông tin vị trí |

Kết quả: 1/6 câu đúng, và câu đúng là đúng vì lý do khác (từ vựng), không phải vì hiểu phủ định. Trong khi đó, trên IMDb test model đạt 0.835. Câu ngắn tự viết là một **domain shift** rõ rệt so với review dài trung bình khoảng 230 từ.

## 5. Định lượng: attention có phải là lời giải thích?

| Đo trên 300 review test | Giá trị |
|---|---|
| Spearman ρ giữa β và \|ΔP\| (trung bình mỗi review) | **0.355** |
| \|ΔP\| khi bỏ 5 token có β cao nhất | **0.108** |
| \|ΔP\| khi bỏ 5 token ngẫu nhiên | 0.023 |
| \|ΔP\| khi bỏ 5 token có β thấp nhất | 0.058 |
| % review đổi nhãn khi bỏ top-5 β / 5 token ngẫu nhiên | 10.3% / 1.8% |
| % review có token β cao nhất = token ảnh hưởng nhất | 30.3% |
| 15 từ sentiment mạnh (\|hệ số LR\| ≥ 4: *bad, great, worst, excellent…*): phần β / phần token | 2.0% / 1.1% (×1.8) |

Trên 2 000 review: β max trung bình 0.027 (đều là 0.0073); 10% token có β cao nhất nhận 21.9% tổng β; entropy β / log L = 0.968.

- **Attention có liên quan đến token quyết định, nhưng chỉ một phần.** Bỏ 5 token β cao nhất ảnh hưởng gấp 4.7 lần bỏ ngẫu nhiên, và từ sentiment mạnh nhận gấp 1.8 lần phần token của chúng. Nhưng ρ chỉ 0.36, và chỉ 30% review có token β cao nhất trùng token ảnh hưởng nhất.
- **Token có β thấp vẫn ảnh hưởng** (bỏ 5 token β thấp nhất: 0.058, gấp 2.5 lần ngẫu nhiên), vì value của nó đã được trộn vào hidden state của các token khác qua lớp self-attention (ví dụ *bad* ở mục 4). β của lớp pooling không thấy được đường ảnh hưởng này.
- **Ablation:** khi điểm pooling tính trên H thay vì K, β gần như đều (entropy / log L = 0.998, β max 0.0098), tức model chỉ còn là average pooling nhưng accuracy vẫn 0.832. Một model có thể đạt kết quả tốt mà attention chẳng "chỉ" vào đâu cả.
- *(Nhóm từ có \|hệ số LR\| ≥ 2 bị lẫn các từ chức năng phổ biến như and, it, you, just, nên em không dùng nhóm này để kết luận.)*

**Kết luận:** không thể nói "attention nhìn sai từ nên model sai". Ở câu "I thought … but … excellent", attention nhìn **đúng** từ mà model vẫn sai; ở review 4118, attention không nhìn vào từ khen mà model vẫn ra positive. Attention weights là một tín hiệu cần đối chiếu (bằng occlusion, minimal pair), không phải lời giải thích.

## 6. Tổng hợp nguyên nhân lỗi

| Nguyên nhân | Ví dụ | Do thiếu thứ tự / cơ chế attention? |
|---|---|---|
| Negation | Not good.; This movie is not bad at all. | **Có**: self-attention không có thông tin vị trí, đổi thứ tự từ không đổi P (Exp 4) |
| Contrast / long-range | I thought … bad, but … excellent; … boring, but … fantastic | **Có**: value của từ tiêu cực lan qua attention, *but* không có vai trò đặc biệt |
| Ambiguous wording / mỉa mai | 8658, 15270 | Một phần: cần hiểu thái độ, không chỉ từng từ |
| Rare usage / thiên lệch từ vựng | *terribly sweet*; *like* bị coi là tiêu cực | Không: do dữ liệu (LR cũng sai) |
| Domain shift | Câu ngắn tự viết: 1/6 đúng | Không: do phân phối dữ liệu |
| Truncation | 12990; 49.1% lỗi là review > 200 token | Không: do max_length |
| Noise (nhãn) | 4118 (nội dung khen, nhãn negative) | Không |
