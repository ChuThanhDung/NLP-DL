# Error analysis — next-word prediction (trigram interpolation)

Model: trigram linear interpolation, λ = (0.2, 0.5, 0.3) cho (unigram add-one, bigram, trigram), chọn trên validation. Các trường hợp lấy từ vị trí ngẫu nhiên trong **test** (mục *Error analysis* của `experiments.ipynb`). C(·) là count trong train.

Tổng quan trên 3 000 vị trí test: top-1 accuracy = 0.195, top-5 accuracy = 0.374.

## 1. Dự đoán đúng

| Context | Model prediction | Expected | Probability | C(context) | C(context, expected) |
|---|---|---|---|---|---|
| in this | **paper** | paper | 0.306 | 8 002 | 4 453 |
| due to | **the** | the | 0.193 | 1 803 | 801 |

- *in this → paper*: cụm *in this paper* là khuôn mẫu cố định của abstract (4 453/8 002 lần *in this* được theo sau bởi *paper*). Context rất phổ biến nên ước lượng MLE đáng tin.
- *due to → the*: quan hệ ngữ pháp cục bộ (giới từ + mạo từ). Bigram/trigram nắm được tốt loại phụ thuộc ngắn này.

→ Đúng khi **context phổ biến** và từ tiếp theo được quyết định bởi **1–2 từ ngay trước**.

## 2. Dự đoán sai

| Context | Model prediction | Expected | P(expected) | C(context) | C(context, expected) | Nguyên nhân |
|---|---|---|---|---|---|---|
| such as | a (0.104) | amazon | 0.0001 | 3 705 | 1 | insufficient training data / domain |
| positioning iii | the (0.084) | boundary | 0.0000 | **0** | 0 | unseen n-gram, sparsity |
| ⟨unk⟩ we | propose (0.103) | conduct | 0.0036 | 74 | 0 | vocabulary limitation + context quá ngắn |

**Case 1 — "…by many online retailers *such as* → amazon".**
*such as* rất phổ biến nhưng đứng trước hàng nghìn từ khác nhau; *amazon* chỉ đi sau *such as* 1 lần trong train. Từ đúng phụ thuộc vào *online retailers* (cách 3 từ), nằm ngoài cửa sổ trigram. Nguyên nhân: **insufficient training data / domain** (tên công ty hiếm trong abstract ML) và **context quá ngắn**.

**Case 2 — "…centroid *positioning iii* → boundary".**
Context *positioning iii* **chưa từng xuất hiện** trong train (C = 0), nên model phải bỏ qua bậc trigram và dựa vào bigram *iii* → … và unigram, và đoán từ phổ biến *the*. P(boundary) ≈ 0 vì *iii boundary* cũng chưa gặp. Nguyên nhân: **unseen n-gram / sparsity**, đúng loại lỗi mà 50.8% trigram token trong test gặp phải.

**Case 3 — "demonstrate the effectiveness of ⟨unk⟩, *we* → conduct".**
Tên phương pháp hiếm bị thay bằng `<unk>` nên mất thông tin. Ngoài ra *we* thường được theo sau bởi *propose* (khuôn mẫu đầu abstract), còn *we conduct* xuất hiện sau phần giới thiệu phương pháp. Phân biệt hai trường hợp cần biết vị trí trong abstract (nhiều câu trước), trigram không thấy được. Nguyên nhân: **vocabulary limitation** (`<unk>`) + **context quá ngắn**.

## 3. Quan sát bổ sung (bảng next-word prediction)

- **the cat** → *this* (0.329): *cat* chỉ xuất hiện 18 lần trong corpus ML và *the cat* đúng 1 lần, nên model "tin" vào một quan sát duy nhất → **domain mismatch**.
- **we propose a novel** → thật là *dialogue* (P = 0.0003), model đoán *approach / method / framework*. Dự đoán hợp lý về ngôn ngữ nhưng không thể biết chủ đề bài báo → giới hạn của **context ngắn**.

## 4. Tổng hợp

| Nguyên nhân | Ví dụ |
|---|---|
| Unseen n-gram / sparsity | positioning iii → boundary |
| Insufficient data / domain mismatch | such as → amazon; the cat → this |
| Vocabulary limitation | ⟨unk⟩ we → conduct |
| Context quá ngắn | such as amazon (*online retailers* ở xa), we propose a novel → dialogue |

Ba trên bốn nhóm lỗi đến từ cùng một gốc: model chỉ nhìn 2 từ trước và ước lượng xác suất bằng đếm, nên hoặc thiếu dữ liệu cho context dài, hoặc không có context đủ dài.
