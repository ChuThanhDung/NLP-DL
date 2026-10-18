# Error analysis — GRU (model tốt nhất) trên IMDb test

Số liệu lấy từ `sequence_classification.ipynb` (mục error analysis). Test 25 000 review; GRU đúng 83.2% (F1 0.822), lỗi 4 196 review.

Confusion matrix (hàng = thật, cột = dự đoán):

| | Pred Neg | Pred Pos |
|---|---|---|
| **Thật Neg** | 11 125 | 1 375 |
| **Thật Pos** | 2 821 | 9 679 |

Đặc điểm: GRU bỏ sót positive nhiều hơn (FN 2 821 so với FP 1 375). Accuracy giảm theo độ dài review: 0–100 token 86.9%, 100–200 token 86.7%, 200–400 token 79.8%, >400 token 73.9%. 53% lỗi thuộc review dài hơn 200 token, trong khi loại review này chỉ chiếm 40% test.

## 3 mẫu đúng (ngẫu nhiên, seed 7)

| idx | True | P(pos) GRU | P(pos) TF-IDF | Phân tích |
|---|---|---|---|---|
| 7218 | neg | 0.086 | 0.306 | "the worst movie I've ever seen" — tín hiệu mạnh, cục bộ ở đầu review |
| 16225 | pos | 0.977 | 0.770 | Review khen rõ ràng ("well more than just a remake"), dùng nhiều từ tích cực |
| 2803 | neg | 0.104 | 0.250 | Có phủ định ("are never developed"), GRU đọc đúng nhờ thứ tự từ |

## 3 mẫu sai

| idx | True | P(pos) GRU | Nguyên nhân | Có phải do sequence modeling? |
|---|---|---|---|---|
| 12061 | neg | 0.993 | "This movie was pure genius… it is hilarious" — review có vẻ khen rồi kết bằng châm biếm; nhãn là negative, mô hình đọc theo bề mặt | Một phần: đọc hết câu nhưng không nhận ra châm biếm (sarcasm) |
| 23867 | pos | 0.008 | Review 398 token, nhiều phủ định (6) và contrast ("I have no idea…"); phần khen nằm rải rác | Có: dependency xa và câu dài; nhưng TF-IDF cũng sai (P 0.18), nên cũng có nhiễu nhãn/khó |
| 18896 | pos | 0.008 | Review mỉa mai về phim B-movie ("B-Movie Nirvana"), phần lớn từ tiêu cực ("ludicrous, terrible, hammy") | Không hẳn: đây là ambiguous sentiment / sarcasm; TF-IDF cũng dự đoán âm |

## Mẫu sai do truncation

| idx | True | P(pos) GRU | P(pos) TF-IDF | Độ dài | Phân tích |
|---|---|---|---|---|---|
| 19227 | pos | 0.042 | 0.515 | 576 | Review dài; GRU chỉ đọc 200 token đầu (phần kể về phim, chưa tới lời khen) |
| 13602 | pos | 0.334 | 0.822 | 260 | Khen ở phần cuối bị cắt |
| 17238 | pos | 0.135 | 0.800 | 606 | Tương tự: phần kết bị cắt |

Với các mẫu này TF-IDF (đọc toàn bộ review) đúng, còn GRU sai vì bị truncate. Đây là lỗi do **sequence length/truncation** chứ không phải do kiến trúc recurrent.

## Kết luận

- Lỗi **không** phải do "GRU không hiểu câu": với 2 mẫu đúng được khen rõ và 1 mẫu phủ định, GRU đọc đúng.
- Lỗi chính đến từ (1) truncation ở 200 token (53% lỗi), (2) châm biếm/ambiguous sentiment, (3) nhiễu nhãn và review dài có nhiều tín hiệu trái chiều.
- Đối chiếu learning curve: GRU train loss giảm còn 0.10 ở epoch 6 trong khi validation loss tăng từ 0.38 (epoch 3) lên 0.48 → có overfitting; chọn epoch 3 làm model cuối.
