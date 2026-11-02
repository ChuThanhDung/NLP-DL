# Error analysis — Transformer classifier trên IMDb

Model phân tích: **Transformer (mean pooling, PE)**, 4 head, seed 42. Accuracy test 0.8338, F1 0.8357. Số liệu lấy từ mục 11–12 của `transformer_classification.ipynb`.

**Cách kiểm tra attention.** Với mỗi review, em ghi các token mà em cho là quan trọng, rồi đối chiếu hai đại lượng:

- **attention nhận được** r<sub>j</sub>: trung bình theo 4 head và theo mọi query của A[h, i, j], tức token j được các token khác "nhìn" bao nhiêu;
- **ΔP (occlusion)**: P(pos) thay đổi bao nhiêu khi che token j (bỏ khỏi key và khỏi mean pooling, các token khác giữ nguyên vị trí). ΔP = P(gốc) − P(che j), nên ΔP < 0 nghĩa là token đó kéo P về positive.

"Attention cao → token quyết định prediction" chỉ là **giả thuyết** về hành vi của model. ΔP đo hành vi thật. Review 200 token có mức đều 1/200 = 0.005, nên r = 0.05 đã gấp 10 lần mức đều.

## 1. Tổng quan

| | Pred Neg | Pred Pos |
|---|---|---|
| **Thật Neg** | 10 273 | 2 227 |
| **Thật Pos** | 1 929 | 10 571 |

- **Lỗi cân bằng hơn LSTM:** FP 2 227, FN 1 929; precision 0.84, recall 0.83 (trung bình 3 seed). LSTM thiên về đoán negative hơn: recall 0.80, precision 0.84.
- **Accuracy giảm theo độ dài:**

  | Độ dài review | 0–100 | 100–200 | 200–400 | > 400 |
  |---|---|---|---|---|
  | Accuracy | 0.856 | 0.859 | 0.816 | 0.756 |

  49.1% số lỗi là review dài hơn 200 token (bị cắt), trong khi loại này chỉ chiếm 40.3% tập test. **Truncation** là nguồn lỗi lớn nhất có thể đo được.
- **Hai model sai ở chỗ khác nhau một phần:** LSTM (seed 42) đoán đúng 42.8% trong số 4 156 review mà Transformer sai.

## 2. Ba mẫu đúng (chọn ngẫu nhiên)

| idx | True / Pred | P(pos) | Token em cho là quan trọng | Top attention nhận được | Top \|ΔP\| | Nhận xét |
|---|---|---|---|---|---|---|
| 16481 | pos / pos | 0.995 | *great job* | favorite 0.035, tries 0.033, confess 0.026 | mọi \|ΔP\| ≤ 0.001 | Attention vào *favorite* hợp lý. P đã bão hòa, nên che một token không làm thay đổi gì; occlusion một token không kiểm chứng được |
| 2289 | neg / neg | 0.222 | *bad, attempt* | **bad 0.068**, i'm 0.042, attempt 0.024 | **bad −0.180**, attempt −0.127 | **Khớp:** token nhận attention cao nhất cũng là token ảnh hưởng nhất. Review 548 từ, phần sau bị cắt nhưng phần đầu đủ để đúng |
| 19418 | pos / pos | 0.821 | *favorite, perfectly* | instead 0.045, baffled 0.040, favorite 0.022 | instead −0.092, **perfectly +0.065**, favorite +0.057 | **Khớp một phần:** *perfectly* đẩy P lên rõ nhưng không nằm trong top attention |

## 3. Năm mẫu sai

| idx | True → Pred | P(pos) | Loại lỗi | Phân tích |
|---|---|---|---|---|
| 18629 | pos → neg | 0.477 | **negation + ambiguous wording** | "What makes this **low budget** production one of my **favorite** movies? **Not** kidding…", sau đó "there is **no** hero… the catastrophe is **nothing** but…". Toàn bài là lời khen nhưng dùng nhiều từ phủ định / tiêu cực theo nghĩa đen. Attention: *no* 0.073, *favorite* 0.055, *nothing* 0.034. Occlusion: *favorite* +0.301, *no* −0.123, *nothing* −0.098. Ở đây attention và occlusion **khớp** nhau, nhưng model chỉ cộng tín hiệu từng từ: *no, nothing* bị hiểu là chê, không hiểu "not kidding" là nhấn mạnh. P = 0.477 nằm sát ngưỡng |
| 4639 | neg → pos | 0.898 | **ambiguous (mỉa mai)** | "Well, I am SO **glad** I watched this on HBO **instead of paying** for it in the theaters…". Nghĩa thật là chê (may mà không mất tiền), nhưng nghĩa đen là "vui". Occlusion: *well* +0.089, *instead* −0.082. *instead* có attention cao nhất (0.097) và đúng là kéo về negative, nhưng không đủ thắng các từ nghĩa đen tích cực |
| 21181 | pos → neg | 0.239 | **contrast + rare usage + truncation** | "'Citizen Kane' it sure isn't… but if you've had a **bad** day… I tentatively suggest this might just be your thing". *bad* ở đây chỉ "ngày tồi tệ" của người xem, không phải đánh giá phim. Attention: *bad* 0.082 (cao nhất). Occlusion: *bad* −0.222 (lớn nhất). Attention **chỉ đúng** từ gây lỗi, nhưng đó là vì model hiểu sai từ, không phải vì nó "hiểu". Review 277 từ: các lời khen sau token 200 ("great title track, … more, more!") bị cắt |
| 6577 | neg → pos | 0.997 | **contrast + domain-specific** | "a very **mediocre** film based on a **superb** series of stories and novels". *superb* nói về sách, *mediocre* nói về phim. Attention nhìn **cả hai** từ quan trọng (*mediocre* 0.171, *superb* 0.143), nhưng model chọn positive. P bão hòa ở 0.997 nên che từng token gần như không đổi gì (\|ΔP\| ≤ 0.005); occlusion một token không giải thích được. Model không biết từ khen áp dụng cho đối tượng nào |
| 12061 | neg → pos | 0.997 | **noise (nhãn)** | "This movie was **pure genius**… I give it **9.5/10**. Rent it today!". Review khen rõ ràng, nhãn negative gần như chắc chắn sai. Model đoán đúng theo nội dung. Attention vào *brilliant* (0.108), *favorite*, *magnificent* |

*(Review 8658 "More suspenseful, more subtle, much, much more disturbing…." mà LAB 06 đã phân tích vẫn bị đoán sai, P = 0.998. Attention dồn vào **subtle** 0.348.)*

## 4. Application — câu tự viết

| Câu | Loại | P(pos) Transformer | không PE | LSTM | Top attention | Top \|ΔP\| | Nhận xét |
|---|---|---|---|---|---|---|---|
| I really enjoyed this movie. | ngắn | **0.998** ✓ | 0.998 | 0.935 | enjoyed 0.281 | enjoyed +0.095 | Khớp |
| Great film. | ngắn | **0.998** ✓ | 0.998 | 0.908 | great 0.529 | great +0.002 (bão hòa) | |
| I went in with low expectations… one of the best films I have seen this year. | dài (53 từ) | **0.918** ✓ | 0.858 | 0.990 | expectations 0.104, chemistry 0.091, best 0.065 | **best +0.420**, cheap −0.059 | Token quyết định (*best*) chỉ đứng thứ 3 về attention |
| This movie is not good. | negation | **0.383** ✓ | 0.904 ✗ | 0.666 ✗ | good 0.277, **not 0.225** | **not −0.606**, this −0.448, good +0.350 | Model có PE dùng *not*: che *not* thì P lên 0.99. Model không PE thì sai |
| This movie is not bad at all. | negation | 0.005 ✗ | 0.002 ✗ | 0.159 ✗ | bad 0.291, not 0.170 | bad −0.036, not −0.010 | Cả 3 model sai. "not bad" (= khá tốt) hiếm gặp hơn "not good" |
| The acting was terrible, but the story was wonderful and I loved it. | contrast | 0.290 ✗ | **0.946** ✓ | **0.841** ✓ | **terrible 0.417**, loved 0.120 | **terrible −0.705**, acting −0.438, wonderful +0.261 | *terrible* thắng *but… wonderful… loved*. Model không PE đúng (có lẽ do may: không biết thứ tự nhưng cân bằng từ khác) |
| The visuals were stunning but the plot was a boring mess. | contrast | **0.010** ✓ | 0.005 | 0.073 | plot 0.187, visuals 0.175, boring 0.172 | plot −0.151, boring −0.092 | Đúng; *boring* thắng *stunning* |
| Well, it was a movie. Some parts worked, some did not. | ambiguous | 0.970 | 0.812 | 0.729 | parts 0.169, did 0.135, well 0.120 | **well +0.370**, parts +0.155 | Quá tự tin cho một câu trung tính. *well* bị coi là khen |
| I am not sure whether I liked it or not. | ambiguous | 0.655 | 0.762 | 0.711 | liked 0.259, not 0.136 | it +0.343, not −0.317, liked +0.272 | Gần trung tính hơn; *not* và *liked* kéo hai chiều |

Kết quả: model đúng 5/7 câu có nhãn, tốt hơn LAB 06 (1/6), nhưng vẫn sai "not bad at all" và câu contrast có *but*. Hai câu ambiguous cho P 0.66–0.97: model **không có cách nói "không chắc"**, vì được train trên review có nhãn rõ ràng.

## 5. Định lượng: attention có phải là lời giải thích?

| Đo trên 300 review test | Giá trị |
|---|---|
| Spearman ρ(attention nhận được, \|ΔP\|), trung bình mỗi review | **0.459** |
| \|ΔP\| khi che 5 token attention cao nhất | **0.233** |
| \|ΔP\| khi che 5 token ngẫu nhiên | 0.022 |
| \|ΔP\| khi che 5 token attention thấp nhất | 0.009 |
| % review đổi nhãn (che top 5 / che 5 ngẫu nhiên) | 28.0% / 2.7% |
| % review có token attention cao nhất = token ảnh hưởng nhất | 43.7% |

- **Attention liên quan khá rõ tới token quyết định** (mạnh hơn LAB 06, nơi ρ = 0.36 và tỉ lệ trùng 30%). Che 5 token được nhìn nhiều nhất làm P đổi gấp 10 lần che ngẫu nhiên.
- **Nhưng không đủ để gọi là lời giải thích:**
  - ρ chỉ 0.46, và 56% review có token ảnh hưởng nhất **không** phải token nhận attention cao nhất.
  - Câu dài trong application: *best* quyết định (+0.420) nhưng chỉ đứng thứ 3 về attention.
  - Review 6577: attention nhìn **đúng cả hai** từ quan trọng mà model vẫn sai.
  - Review 21181: attention "đúng" vào *bad*, nhưng đó là từ model hiểu sai.
- Attention đo token nào được **nhìn**, không đo token đó **đẩy** prediction theo hướng nào. Value của token, W<sub>O</sub>, FFN và classifier quyết định phần còn lại. Một token có thể được nhìn nhiều mà gần như không ảnh hưởng: "Great film." có *film* nhận 0.47 attention, ΔP ≈ 0.

**Kết luận:** attention visualization cho thấy phân phối trọng số, là một tín hiệu hữu ích để đặt giả thuyết. Muốn nói về hành vi của model thì phải kiểm tra bằng can thiệp (occlusion, minimal pair, đảo thứ tự). Đây là cách diễn giải của em (human interpretation), không phải điều model "nói ra".

## 6. Tổng hợp nguyên nhân lỗi

| Nguyên nhân | Ví dụ | Liên quan đến kiến trúc? |
|---|---|---|
| Truncation | 49.1% lỗi là review > 200 token; 21181 | Gián tiếp: max_length bị giới hạn vì chi phí O(T<sup>2</sup>) (Exp 5: tăng 128 → 256 thì accuracy 0.805 → 0.846) |
| Negation | *not bad at all*; 18629 (*no, nothing, not kidding*) | Một phần: có PE thì "not good" đã đúng, "not bad" thì chưa (thiếu dữ liệu và chỉ có 1 lớp) |
| Contrast | *terrible, but… wonderful*; 6577 (*mediocre… superb*) | Có: một lớp attention + mean pooling vẫn gần với "cộng tín hiệu từng từ", *but* không có vai trò đặc biệt |
| Ambiguous / mỉa mai | 4639 (*SO glad… instead of paying*) | Một phần: cần hiểu thái độ người viết |
| Rare usage | *bad day* (21181), *well* ở đầu câu | Không: thiên lệch dữ liệu |
| Domain-specific | 6577 (khen sách, chê phim) | Không: cần biết từ khen áp dụng cho đối tượng nào |
| Noise (nhãn) | 12061 (9.5/10 nhưng nhãn negative) | Không |
