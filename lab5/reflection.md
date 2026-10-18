# Reflection — từ Average Word2Vec đến sequence model

Số liệu lấy từ `sequence_classification.ipynb` / `results.csv` (IMDb test, 25 000 review).

| Model | Params | Accuracy | F1 | Thời gian train |
|---|---|---|---|---|
| TF-IDF + LR (baseline) | 40 791 | 0.8882 | 0.8883 | 7.1 s |
| RNN (max_len 200) | 2 529 769 | 0.7408 | 0.7482 | 37.3 s |
| LSTM (max_len 200) | 2 618 089 | 0.8156 | 0.8117 | 42.3 s |
| GRU (max_len 200) | 2 588 649 | 0.8322 | 0.8219 | 40.4 s |

## 1. Đối chiếu prediction

- **P1 (word order):** đúng về khả năng: RNN thay đổi h<sub>t</sub> theo thứ tự (phân tích trong `calculations.md`, Bài 3). Nhưng em đã sai về accuracy: TF-IDF + LR (0.888) **tốt hơn** cả GRU (0.832) trên dữ liệu này. Khả năng xử lý thứ tự không tự động cho kết quả tốt hơn.
- **P2 (LSTM vs RNN về dependency xa):** đúng. Gradient theo vị trí cho thấy RNN gần như không nhận tín hiệu từ token đầu (‖∂L/∂x<sub>1</sub>‖ / ‖∂L/∂x<sub>200</sub>‖ ≈ 1.7·10<sup>−7</sup>), LSTM là 6.4·10<sup>−2</sup>, GRU là 0.29. RNN chỉ đạt 0.74 accuracy, thấp nhất.
- **P3 (độ dài chuỗi):** đúng: thời gian train gần tuyến tính theo độ dài (50 → 17.5 s; 100 → 25.6 s; 200 → 42.3 s; 400 → 69.9 s).
- **P4 (truncate 30 token):** đúng hướng và mạnh hơn em nghĩ: chỉ giữ khoảng 13% token, model tệ đi rõ rệt. Với max_len 50, LSTM chỉ đạt 0.746.
- **P5 (LSTM luôn tốt hơn RNN):** đúng trên IMDb (LSTM 0.816 so với RNN 0.741). Không kiểm chứng được trên dataset khác.

## 2. Thí nghiệm sequence length: trade-off

- Tăng max_len từ 50 → 400 làm F1 tăng từ 0.752 lên 0.847, nhưng thời gian train tăng gấp 4 lần (17.5 s → 69.9 s) và bộ nhớ GPU tăng từ 220 MB lên 568 MB.
- **Cắt đầu hay cắt cuối:** với max_len 100, giữ 100 token **cuối** cho F1 0.828, còn giữ 100 token **đầu** cho F1 0.785. Kết luận của review thường nằm ở cuối, nên giữ phần cuối có lợi hơn.
- Có trade-off giữa context và chi phí tính toán: context dài hơn cho accuracy cao hơn nhưng tốn thời gian gần tuyến tính, và RNN thuần còn không tận dụng được context dài.

## 3. Quan sát về TF-IDF

TF-IDF đọc toàn bộ review nên không bị ảnh hưởng bởi truncation, và trên IMDb nó vẫn mạnh hơn các sequence model được train từ đầu với 25 000 review. Đây là dấu hiệu rằng dữ liệu chưa đủ lớn cho một recurrent model có 2.5M tham số học biểu diễn từ từ đầu. Ở `LAB 04`, Word2Vec pretrain trên chính corpus cũng không giúp vượt TF-IDF.

## 4. Giới hạn của recurrent model (dẫn tới Attention)

- Vẫn đọc tuần tự: không song song hóa theo thời gian, nên chậm khi chuỗi dài.
- Thông tin từ token đầu vẫn phải đi qua nhiều bước: gradient ở token 1 nhỏ hơn token 200 khoảng 6·10<sup>6</sup> lần trong RNN, khoảng 3.4 lần trong GRU và khoảng 16 lần trong LSTM.
- Với review dài, model phải nén toàn bộ thông tin vào h<sub>T</sub>, trong khi phần kết luận có thể nằm ở bất kỳ đâu. Attention cho phép token truy cập trực tiếp các token khác mà không phải đi qua chuỗi.

## 5. Bài học rút ra

- Đừng chỉ nhìn accuracy: GRU và LSTM khác nhau về cách nhớ dependency xa (gradient), không chỉ về số điểm.
- Overfitting: GRU train loss xuống 0.11 nhưng validation loss tăng sau epoch 3; chọn epoch theo validation là cần thiết.
- Với dữ liệu vừa phải, representation đơn giản và mạnh (TF-IDF) có thể thắng kiến trúc phức tạp.
