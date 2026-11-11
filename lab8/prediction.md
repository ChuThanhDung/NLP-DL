# Prediction — LAB 08

Ghi trước khi chạy bất kỳ thí nghiệm nào.

## Prediction 1 — Pretrained + fine-tuning so với Transformer from scratch

- **Prediction:** pretrained (DistilBERT) fine-tuned tốt hơn rõ rệt: accuracy khoảng 0.90–0.92, so với khoảng 0.83–0.85 của Transformer from scratch (LAB 07 đạt 0.835).
- **Reason:** model from scratch phải học cả nghĩa từ lẫn task chỉ từ khoảng 20 000 review. Model pretrained đã có contextual representation học từ corpus rất lớn, nên phần cần học thêm chỉ là ánh xạ sang sentiment.
- **Expected evidence:**
  - accuracy và F1 trên test chênh ít nhất 5 điểm;
  - validation loss của model pretrained thấp ngay từ epoch 1;
  - đổi lại, train mỗi epoch chậm hơn nhiều vì model có khoảng 66 triệu tham số, so với khoảng 4 triệu.

## Prediction 2 — Freeze encoder, chỉ train classification head

- **Chọn: C. kém hơn fine-tuning toàn bộ.**
- **Giải thích:**
  - Frozen vẫn phân loại được, vì vector [CLS] / mean của encoder đã chứa thông tin về nghĩa của câu, mà sentiment là một phần thông tin đó. Dự đoán accuracy khoảng 0.80–0.85.
  - Tuy nhiên encoder được pretrain cho MLM, không phải cho sentiment, nên các đặc trưng cần cho sentiment không nằm sẵn theo một hướng tuyến tính rõ ràng. Fine-tuning cho phép điều chỉnh lại representation, nên dự đoán hơn khoảng 5–8 điểm.
  - Ngược lại, frozen train rất nhanh, vì chỉ cần chạy encoder một lần để lấy feature.

## Prediction 3 — Giảm training data xuống 10%

- **Prediction:** model from scratch giảm mạnh hơn. Ở 10% (khoảng 2 000 review), from scratch có thể xuống khoảng 0.70, còn pretrained vẫn khoảng 0.86–0.88.
- **Cơ sở:**
  - Pretraining đã cung cấp sẵn kiến thức về từ, cú pháp và ngữ cảnh. Với dữ liệu downstream ít, model pretrained chỉ cần học "hướng" sentiment trong không gian representation có sẵn.
  - Model from scratch phải học embedding của hơn 30 000 token từ đầu. Với 2 000 review, phần lớn token chỉ xuất hiện vài lần, nên embedding gần như ngẫu nhiên và model dễ overfit.
  - Do đó đường data-efficiency của pretrained sẽ phẳng hơn: khoảng cách giữa 10% và 100% nhỏ hơn.

## Dự đoán phụ (Exp 4, Exp 5)

- **Max length 64 / 128 / 256:** accuracy tăng theo T, vì review IMDb dài (trung vị khoảng 230 từ), nhưng mức tăng nhỏ dần. Thời gian và bộ nhớ tăng nhanh hơn tuyến tính vì attention có chi phí O(T²).
- **Domain shift (train IMDb → test Yelp / Amazon):** accuracy giảm nhưng không sụp đổ, khoảng 0.85–0.88, vì từ chỉ cảm xúc ("great", "terrible") là chung giữa các domain. From scratch sẽ giảm mạnh hơn, vì vocabulary học từ phim không phủ được từ về đồ ăn và sản phẩm.
