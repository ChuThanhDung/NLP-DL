# Error analysis — TF-IDF (unigram) + Logistic Regression trên IMDb test

Model phân tích: baseline `TF-IDF + Logistic Regression`, accuracy test = 0.8814. Các mẫu lấy từ test set (mục 10 của `experiment.ipynb`); `idx` là chỉ số dòng trong test set.

## 1. Confusion matrix

| | Pred Pos | Pred Neg |
|---|---|---|
| **Actual Pos** | TP = 10972 | FN = 1528 |
| **Actual Neg** | FP = 1437 | TN = 11063 |

- Tổng lỗi = 2965; trong đó **FN (1528) nhiều hơn FP (1437)** một chút.
- Precision = 0.8842, Recall = 0.8778: recall thấp hơn precision, tức là model **bỏ sót positive nhiều hơn là báo nhầm positive**.
- Metric phản ánh vấn đề này là **recall** (và precision cho hướng ngược lại).

## 2. Mẫu phân loại ĐÚNG (3 mẫu)

| idx | Text (rút gọn) | True | Predicted (P(pos)) | Analysis |
|---|---|---|---|---|
| 19241 | "Without a doubt, one of Tobe Hoppor's best! Epic storytellng, great special effects…" | positive | positive (0.957) | Có nhiều từ cảm xúc mạnh (*best, epic, great*) nên dễ phân loại. |
| 631 | "This is a terrible movie that only gets worse… The acting was bad… Dennis Weaver is such a great actor… DON'T WATCH THIS MOVIE!" | negative | negative (0.001) | Tín hiệu âm (*terrible, worse, bad*) nhiều hơn tín hiệu dương (*great*), nên tổng trọng số nghiêng về âm. |
| 1796 | "I'm not sure why I disliked this film so much… uninteresting and unlikeable… embarrassing… tedious… nauseating" | negative | negative (0.003) | Rất nhiều từ âm mạnh bù đắp cho cụm tích cực giả định ("if you liked it, then I'm happy for you"). |

## 3. Mẫu phân loại SAI (4 mẫu)

| idx | Text (rút gọn) | True | Predicted (P(pos)) | Category | Analysis |
|---|---|---|---|---|---|
| 21966 | "It isn't the greatest scifi flick I've ever seen, but it is not a bad movie. The acting is good… Don't let the nay sayers run you off." | positive | negative (0.499) | **Negation** | "not a bad movie" thực ra là khen, nhưng túi từ thấy *not* và *bad* như hai feature riêng, và *bad* có trọng số âm mạnh. |
| 9430 | "THANK GOD YOU'RE HERE is painful, positively painful… this turkey… Here's some lovely Alpo… the audience paid to laugh…" | negative | positive (0.526) | **Lexical ambiguity** | *positively* (trạng từ nhấn mạnh) và *thank god* trông như từ tích cực. |
| 21274 | "'Traffik 1989'… breaks down the opium/heroine trade… a junkie… Synergistically entertaining and educational… time well spent." | positive | negative (0.457) | **Domain-specific term** | Các từ chỉ chủ đề (*opium, heroine, junkie*) mang trọng số âm dù không phải đánh giá về phim. |
| 1608 | "…It's worth the effort, and PF is a lot better than Eraserhead… the best known non-mainstream director…" | negative | positive (0.926) | **Data noise** | Nội dung đọc như khen nhưng nhãn là negative, nhãn có thể không nhất quán. |

## 4. Tổng hợp

- Trong 4 mẫu sai có 4 nguyên nhân khác nhau: negation, lexical ambiguity, domain-specific term và data noise.
- Ba nguyên nhân đầu (negation, lexical ambiguity, domain-specific term) đều xuất phát từ việc TF-IDF coi mỗi từ là một feature độc lập, có một trọng số cố định bất kể ngữ cảnh.
- Nguyên nhân thứ tư (data noise) không phải lỗi của representation mà là nhãn không khớp với nội dung.
