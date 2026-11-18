# Prediction — LAB 09

Ghi trước khi chạy baseline và Transformer.

## Prediction 1 — Pretrained Transformer so với classifier dựa trên word features

- **Prediction:** entity-level F1 tăng rõ, từ khoảng 0.75–0.80 (Logistic Regression trên word features) lên khoảng 0.90 (`bert-base-cased` fine-tuned). Recall tăng nhiều hơn precision.
- **Reason:**
  - Baseline chỉ nhìn chính từ đó, vài từ xung quanh và các cờ chữ hoa / chữ số. Nó nhận diện tốt entity đã thấy trong train, nhưng với từ chưa gặp chỉ dựa được vào chữ hoa. Chữ hoa không phân biệt được PER / ORG / LOC, và cũng không có ở các tiêu đề viết hoa toàn bộ của CoNLL.
  - Transformer có contextual representation của cả câu, nên phân biệt được "Washington said" (PER) với "in Washington" (LOC). Nó cũng ghép được tên lạ từ subword.
  - Baseline dự đoán từng token độc lập, nên dễ sinh chuỗi BIO bất hợp lệ (I-PER sau O) và sai biên.
- **Expected evidence:**
  - Token accuracy của hai model chênh ít (cả hai trên 0.9, vì phần lớn token là O), nhưng entity F1 chênh nhiều.
  - Khoảng cách lớn nhất ở ORG và MISC, và ở entity không xuất hiện trong train.

## Prediction 2 — Entity nhiều token dễ hay khó hơn entity 1 token?

- **Prediction:** khó hơn. F1 giảm dần theo độ dài: 1 token > 2 token > 3 token > 4+ token.
- **Reason:**
  - Entity-level chỉ tính đúng khi **mọi** token đúng cả biên lẫn loại. Nếu mỗi token đúng với xác suất p độc lập, entity n token đúng với xác suất khoảng pⁿ.
  - Entity dài ("United Nations Security Council") thường chứa từ thường ("of", "the", "Council") khiến model dễ cắt sớm hoặc tách đôi entity. Nó cũng hiếm hơn trong train.
  - Ngoại lệ có thể có: tên người 2 token (họ + tên) rất phổ biến và có mẫu rõ, nên nhóm 2 token có thể không kém nhóm 1 token. Nhóm 1 token lại chứa nhiều từ viết tắt / mơ hồ (tên đội bóng = tên thành phố).

## Prediction 3 — Entity hiếm trong training data có gây lỗi nhiều hơn không?

- **Prediction:** có. Entity không xuất hiện trong train (unseen) có F1 thấp hơn entity đã thấy (seen) khoảng 5–10 điểm với Transformer, và khoảng 20 điểm với baseline.
- **Reason:**
  - Với entity đã thấy, model có thể "nhớ" chuỗi ký tự kèm nhãn. Với entity mới, model phải suy ra từ ngữ cảnh và hình thái của từ.
  - Transformer bớt bị ảnh hưởng nhờ pretraining: nhiều tên riêng tuy không có trong CoNLL train nhưng đã gặp trong corpus pretrain. Nó cũng dùng được ngữ cảnh ("said", "in", "club").
  - Baseline gần như chỉ còn cờ chữ hoa cho từ mới, nên bị ảnh hưởng nặng hơn.
  - Entity hiếm thường bị tách thành nhiều subword. Với Strategy A, chỉ subword đầu được dự đoán, nên nếu mảnh đầu không mang thông tin (ví dụ "Sh" trong "Shoaib") thì dễ sai.
