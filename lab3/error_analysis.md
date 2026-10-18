# Error analysis — Word2Vec (Skip-gram, 100 chiều, window 5) trên text8

Số liệu lấy từ `word_embedding.ipynb`. "Đúng" = gần với trực giác con người; "sai/bất ngờ" = lệch trực giác. Bằng chứng corpus lấy từ đếm trực tiếp trong text8 (cửa sổ ±5, bỏ stopwords).

## 3 similarity đúng

| Cặp | Cosine | Quan sát | Giải thích |
|---|---|---|---|
| computer → computers | 0.80 (top-1) | computers là biến thể số nhiều | Cùng nghĩa, cùng context |
| car → automobile | 0.777 | automobile là top-2 của car | Đồng nghĩa; cả hai có context *race, driver* |
| disease → diseases / infection | 0.85 / 0.82 | top-3 neighbors của *disease* là *diseases, infectious, infection* | Biến thể hình thái và từ cùng trường nghĩa |

## 3 similarity sai hoặc bất ngờ

| Cặp | Cosine | Observed | Expected | Possible explanation | Evidence from corpus |
|---|---|---|---|---|---|
| doctor → supervillain / onizuka / hellboy | 0.60 | top-1, 3, 5 của *doctor* là nhân vật hư cấu | physician, nurse, patient | **Polysemy + domain bias**: *doctor* là tên nhân vật (*Doctor Who*) và học hàm (*juris doctor*) | "doctor who" 222 lần, "the doctor" 187 lần, "juris doctor" 16 lần; *medical doctor* chỉ 20 lần |
| banana → cayenne / daiquiri / lobster | 0.73 / 0.71 | banana gần đồ ăn/đồ uống | fruit, yellow, apple | **Context window + corpus nhỏ**: *banana* chỉ có 111 lần; context chủ yếu là công thức nấu ăn | "knock" xuất hiện 21 lần trong cửa sổ ±5 quanh *banana* (không phải cụm cố định); cụm "banana republic" 5 lần; không có "banana split" |
| good → bad | 0.78 (cao) | good và bad gần nhau | trái nghĩa, cosine thấp | **Distributional limitation**: hai từ thay thế được trong cùng câu (*a good/bad idea*, *good/bad luck*) | "good luck" 39 lần, "bad luck" 45 lần; "good idea" 37, "bad idea" 14 |

Đáng chú ý: *doctor* và *physician* có cos 0.466, nhưng *physician* không nằm trong top-5 của *doctor* trên text8. Đây là một lỗi recall của corpus nhỏ, không phải lỗi thuật toán.

Ngoài ra *bank* cho *banks, monetary, fund* (nghĩa tài chính) với cos 0.65–0.75, và cos(bank, river) chỉ 0.44 dù *river bank* có trong corpus: đây là **polysemy** (một vector cho hai nghĩa; xem `reflection.md`).

## Nguyên nhân tổng hợp

| Nguyên nhân | Ví dụ |
|---|---|
| Polysemy | doctor (Doctor Who / bác sĩ), bank (ngân hàng / bờ sông) |
| Domain bias của corpus | text8 nhiều về phim, truyện, nhân vật hư cấu → *doctor* gần nhân vật |
| Frequency thấp | banana (111 lần), cayenne (18 lần): vector không ổn định |
| Context giống nhau nhưng nghĩa khác | good / bad: cùng context đánh giá |
| Window | window 5 lấy cả chủ đề rộng (food, recipe) |

Đối với corpus arXiv (LAB 01 corpus), *doctor* cho *physician, clinician, physicians, nurse, patient* — đúng hơn nhiều, vì domain thay đổi. Vậy lỗi phần lớn đến từ **corpus**, không chỉ thuật toán.
