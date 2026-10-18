# Reflection — từ Word2Vec đến Transformer

Số liệu lấy từ `word_embedding.ipynb` / `results.csv` (model chính: Skip-gram, 100 chiều, window 5, text8).

## Đối chiếu prediction

| Prediction | Kết quả | Đúng/sai |
|---|---|---|
| P1: doctor–physician gần nhất; banana, car xa | doctor–physician 0.466 ≈ doctor–hospital 0.468; banana–car = **0.450**, cao hơn nhiều so với banana–doctor (0.038) | Đúng một phần: không có từ y tế nào "gần nhất" rõ rệt; banana–car gần nhau ngoài dự đoán |
| P2: window lớn → chủ đề, window nhỏ → chức năng | window 2 → 5 → 10: doctor–hospital 0.437 → 0.468 → 0.477 (tăng). Co-occurrence: window 1 cho *doctor → physician*, window 5 cho *doctor → patient, medical* | Đúng |
| P3: dimension không chắc chắn tăng chất lượng | 50 / 100 / 300: WordSim 0.662 / **0.689** / 0.676; analogy 0.294 / 0.386 / 0.390; IMDb 0.723 / 0.754 / **0.785**; thời gian 55 / 52 / 207 s | Đúng: intrinsic bão hòa ở 100, downstream vẫn tăng |
| P4: 100 câu thì không chắc chắn | cos(doctor, physician) = 0.54 ± 0.04 qua 5 seed, nhưng top-3 neighbors của doctor là *fake, body, mystical, helper* | Đúng: con số có vẻ cao nhưng không gian không có nghĩa |

## Bảng so sánh representation

| Representation | Context-dependent? | Sparse/Dense | Một từ có nhiều vector? |
|---|---|---|---|
| TF-IDF | Không (biểu diễn document, không biểu diễn nghĩa của từ) | Sparse (99.8% zeros ở LAB 01) | Không: một từ = một chiều cố định |
| Co-occurrence | Không (đếm trên toàn corpus) | Sparse (density 0.9–3.5% với V = 10 000) | Không: một hàng / từ |
| Word2Vec | Không (static, học từ context nhưng cố định sau khi train) | Dense (100 chiều) | Không: một vector / từ |
| Contextual embedding | **Có**: vector tính lại theo câu | Dense | **Có**: mỗi lần xuất hiện một vector |

## Tại sao *bank* cần contextual representation?

text8 dùng *bank* theo cả hai nghĩa: từ đứng trước là *west/left/south…* (khoảng 307 lần, nghĩa địa lý, kể cả West Bank) và *central/world/national…* (khoảng 499 lần, tài chính). Word2Vec gộp tất cả vào **một** vector, kéo về nghĩa phổ biến hơn: 10 neighbors gần nhất đều là tài chính (*banks, monetary, fund, banking, ecb…*), cos(bank, river) = 0.44 < cos(bank, financial) = 0.59. Vector của *bank* trong "I deposited money in the bank" và "we sat on the river bank" **giống hệt nhau**, dù chỉ riêng context của hai câu đã phân biệt được hai nghĩa (cos(context, money) = 0.76 so với 0.40; cos(context, river) = 0.41 so với 0.68). Một representation tính vector của từ *từ chính context đó* (contextual embedding, self-attention) sẽ tách được hai nghĩa.

## Bài học rút ra

- **Distributional hypothesis hoạt động, nhưng "similar context" ≠ "synonym":** *good–bad* = 0.78 (trái nghĩa, cùng context); *doctor* gần *supervillain, hellboy* vì *doctor who* (222 lần) và truyện tranh.
- **Corpus quyết định nghĩa:** *transformer* trong text8 gần *resistor, mosfet*; trong arXiv gần *bert, xlnet, seq2seq*. *doctor* trong arXiv gần *physician, clinician*.
- **PPMI là bước quan trọng với count vectors:** raw count cho computer–river = 0.88 (vì cùng context *the, of*); PPMI giảm xuống 0.04.
- **Semantic search (LAB 01 → LAB 03):** avg embedding đưa relevant đầu tiên của *heart attack prediction* từ rank 18 lên rank 7 và tăng MRR (0.642 → 0.664), nhưng P@5 giảm (0.525 → 0.300). Hybrid lexical + dense cân bằng hơn (P@5 0.475, MRR 0.664). Lưu ý: nhãn relevance được tạo bằng pooling từ các hệ lexical nên có thể đánh giá thấp phương pháp dense. cos("heart attack", "myocardial infarction") = 0.60 với Word2Vec, so với 0 với TF-IDF.
- **Static embedding vẫn thiếu:** một vector cho mọi nghĩa, không xét thứ tự từ → động lực cho sequence models (LAB 05) và contextual embeddings / Transformer.
