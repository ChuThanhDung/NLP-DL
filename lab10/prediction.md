# Prediction — LAB 10

Ghi trước khi chạy baseline và Transformer.

## Prediction 1 — Question và context có nhiều từ giống nhau, nhưng chỉ một vị trí trả lời đúng

- **Prediction:** có, model dễ bị nhầm, nhưng mức độ khác nhau giữa hai model.
  - **Baseline** (chọn câu có nhiều từ trùng với câu hỏi) bị nhầm nặng: câu nào lặp lại nhiều từ của câu hỏi sẽ được chọn, dù không chứa đáp án.
  - **Transformer** ít bị hơn nhưng vẫn có. Dự đoán EM giảm khoảng 5–10 điểm khi trong context có một câu "giống câu hỏi" mà không chứa đáp án.
- **Reason:**
  - Trên SQuAD, câu chứa đáp án thường lặp lại nhiều từ của câu hỏi, vì người tạo câu hỏi viết dựa trên chính câu đó. Model dễ học "phím tắt": tìm vùng trùng từ nhiều nhất rồi lấy cụm gần đó có kiểu phù hợp (tên người cho "who", số cho "how many").
  - Self-attention giữa question và context giúp Transformer kiểm tra quan hệ, không chỉ đếm từ trùng. Nhưng model fine-tune trên SQuAD vẫn thừa hưởng phím tắt này.
- **Kiểm chứng:**
  - Chia câu hỏi theo mức trùng từ với câu chứa đáp án: nhóm trùng ít sẽ có EM / F1 thấp hơn rõ, nhất là với baseline.
  - Chèn một câu distractor giống câu hỏi nhưng mang đáp án giả: EM sẽ giảm.

## Prediction 2 — Answer span càng dài thì dự đoán biên có khó hơn không?

- **Prediction:** có. EM giảm mạnh theo độ dài: 1 word > 2–3 > 4–6 > 7+. Nhóm 7+ có thể chỉ còn khoảng một nửa EM của nhóm 1 word. F1 giảm ít hơn nhiều.
- **Reason:**
  - EM đòi cả start lẫn end đúng tuyệt đối. Với đáp án dài, có nhiều điểm dừng "hợp lý" (có lấy mệnh đề phụ không, có lấy "the" không, có lấy đơn vị không). Gold annotation cũng không nhất quán ở biên.
  - Start và end được dự đoán gần như độc lập, nên span càng dài thì end càng xa start và càng khó khớp.
  - Đáp án dài thường là cụm mô tả hoặc lý do ("because…"), không có dấu hiệu rõ như tên riêng hay con số.
  - F1 vẫn cao vì lệch biên một vài từ chỉ mất vài token trong overlap.
  - Dự đoán thêm: lỗi **end** sẽ nhiều hơn lỗi **start**. Start thường là đầu một cụm danh từ, dễ nhận ra; còn end phải quyết định cắt ở đâu.

## Prediction 3 — Context chứa nhiều thông tin không liên quan và distractor

- **Prediction:**
  - **Thông tin không liên quan** (câu từ bài khác, không trùng từ với câu hỏi): performance gần như không đổi, giảm dưới 2 điểm F1. Model đã quen với context dài, nhiều câu không liên quan.
  - **Distractor** (câu trùng nhiều từ với câu hỏi nhưng chứa đáp án sai): performance giảm mạnh. Dự đoán F1 giảm 15–30 điểm, vì model chọn theo mức trùng từ.
  - Context dài hơn còn có thể bị cắt thành nhiều window, tăng thêm cơ hội chọn nhầm.
- **Reason:** model không "hiểu" sự thật, nó tìm span phù hợp nhất với câu hỏi. Distractor được thiết kế đúng để khớp câu hỏi hơn cả câu thật.

## Prediction 4 — Đổi thứ tự câu trong context nhưng giữ nguyên đáp án

- **Prediction:** ảnh hưởng nhỏ. EM / F1 giảm khoảng 1–3 điểm, phần lớn dự đoán giữ nguyên.
- **Reason:**
  - Đáp án nằm trong một câu, và phần lớn câu hỏi SQuAD chỉ cần đọc đúng câu đó, nên đổi thứ tự câu không làm mất thông tin.
  - Phần giảm nhỏ đến từ:
    1. Câu hỏi cần nối hai câu liền nhau (đại từ "it", "he" trỏ về câu trước). Đổi thứ tự làm đứt liên kết.
    2. Model có thể đã học thiên lệch vị trí: đáp án SQuAD hay nằm ở đầu đoạn.
    3. Đáp án bị dời sang window khác khi context dài hơn max_length.
