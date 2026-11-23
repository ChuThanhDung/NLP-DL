# Error analysis — LAB 10

Model: DistilBERT fine-tuned, max_length 384, stride 128. Tập test là SQuAD v1.1 dev (10 570 câu hỏi). Số liệu lấy từ `qa_finetuning.ipynb`.

## 1. Phân loại lỗi theo start / end

Mỗi dự đoán được so với gold gần nhất (gold có F1 cao nhất), theo vị trí ký tự trong context. Start (hoặc end) được coi là đúng nếu chênh lệch chỉ là khoảng trắng, dấu câu hoặc mạo từ.

| Nhóm | Số câu | % | Ghi chú |
|---|---|---|---|
| correct (start và end đúng) | 7 400 | 70.0% | trùng EM (71.07%) sau chuẩn hóa, trừ một số câu khớp một gold khác |
| **wrong start** (end đúng) | **759** | 7.2% | dự đoán dài hơn gold 59.6%; F1 trung bình 0.61 |
| **wrong end** (start đúng) | **669** | 6.3% | dự đoán dài hơn gold 66.8%; F1 trung bình 0.56 |
| wrong boundary (sai cả hai, có chồng lấn) | 139 | 1.3% | F1 trung bình 0.41 |
| wrong span (không chồng lấn) | 1 603 | 15.2% | 46.5% ở cùng câu với gold, 53.5% ở câu khác |

- **Wrong start nhiều hơn wrong end** (759 so với 669, tỉ lệ end / start = 0.88). Điều này **ngược với dự đoán** của mình (dự đoán end khó hơn).
- **Giả thuyết:** trong SQuAD, phần khó quyết định thường là **đầu** đáp án: có lấy từ bổ nghĩa đứng trước không ("over 14,000" hay "14,000", "within the chloroplast's stroma" hay "chloroplast's stroma", "afternoon of May 2" hay "May 2"). Còn đuôi đáp án thường dừng ở dấu phẩy / dấu chấm, một tín hiệu rõ hơn.
  - Người gán nhãn cũng không nhất quán với các từ như "over", "about", "the": cùng loại câu hỏi số lượng nhưng có câu gold giữ "about", có câu không.
- **Cả hai loại lỗi đều giữ được phần lớn đáp án** (F1 trung bình 0.56–0.61). Đây là nguồn chính của 1 837 câu "EM = 0 nhưng F1 > 0".
- **Wrong span (15.2%) là lỗi nặng nhất,** F1 = 0. Một nửa vẫn nằm trong đúng câu chứa đáp án: model tìm đúng chỗ nhưng chọn sai cụm, thường là một cụm cùng kiểu khác trong câu. Nửa còn lại nằm ở câu khác (distractor trong context).
- **Lỗi do sliding window rất ít:** chỉ 1.1% lỗi wrong span có dự đoán nằm ở window thứ hai trở đi.

## 2. Bảng lỗi

### 2.1. Wrong start (end đúng)

| Question | Gold | Prediction | Error type | Explanation |
|---|---|---|---|---|
| As a result of a compromise, about how many Chinese schools became National Type schools? | 60 | Over 60 | wrong start / multiple plausible spans | "Over 60" mới là cách nói đúng của context. Gold bỏ "Over" dù câu hỏi có "about how many". Ranh giới phụ thuộc người gán nhãn. |
| Approximately how books did Alexander Dyce bequeathed to the museum? | over 14,000 | 14,000 | wrong start / question ambiguity | Ngược với câu trên: ở đây gold giữ "over", model bỏ. Hai câu cho thấy annotation không nhất quán về từ chỉ mức độ, nên model không thể đúng cả hai. |
| Where is the Z-ring? | within the chloroplast's stroma | chloroplast's stroma | wrong start / partial answer | Câu hỏi "Where" thì gold giữ giới từ "within". Model bỏ giới từ, kiểu đáp án hay gặp ở câu "where" khác của SQuAD. |
| What are the laws of physics of Galileo, in reference to objest in motion and rest? | the same in every inertial frame of reference | every inertial frame of reference | wrong start / partial answer | Model lấy cụm danh từ, gold lấy cả vị ngữ "the same in". Câu hỏi có lỗi chính tả ("objest"). |
| What are the two principal Asian-American groups living in the west side neighborhood of Fresno? | Hmong or Laotian | Mexican-American and Asian-American (principally Hmong or Laotian | wrong start / distractor | Model bắt đầu từ cụm liệt kê "Mexican-American and Asian-American", có từ khóa "Asian-American" của câu hỏi, rồi kéo dài tới đáp án thật. |
| When was Time Warner cable forced to restore ABC stations to affected markets? | afternoon of May 2. | May 2 | wrong start / partial answer | Model chọn phần ngày (kiểu đáp án quen của "when"), bỏ "afternoon of". |
| What distinguishes stromal thylakoids? | are in contact with the stroma | granal thylakoids, which are arranged in grana, and stromal thylakoids, which are in contact with the stroma | wrong start / long context | Câu context so sánh hai loại; model bắt đầu từ loại đầu tiên và lấy cả mệnh đề so sánh. |

### 2.2. Wrong end (start đúng)

| Question | Gold | Prediction | Error type | Explanation |
|---|---|---|---|---|
| With what body must a pharmacy technician register? | General Pharmaceutical Council (GPhC) | General Pharmaceutical Council | wrong end / multiple plausible spans | Tên viết tắt trong ngoặc là phần thông tin thừa; cả hai đáp án đều hợp lý. |
| What was the total nominal GDP of Warsaw in 2010? | 191.766 billion PLN | 191.766 billion | wrong end / partial answer | Model dừng trước đơn vị tiền tệ. Đơn vị là token hiếm ("PLN"), ít khi nằm trong đáp án số trong train. |
| What was the overly generous royalty amount that Tesla had been receiving? | $2.50 per AC horsepower | $2.50 | wrong end / partial answer | Câu hỏi "amount", nên model trả số tiền; gold muốn cả đơn vị tính ("per AC horsepower"). |
| Upon learning of a French scounting party in the area, what did Washington do? | surprised the Canadians on May 28 | surprised | wrong end / partial answer | Đáp án là một hành động (cụm động từ). Model chỉ lấy động từ, quá ngắn. Câu hỏi "what did X do" khó định biên. |
| Along with drought, what is one other factor that is pushing the Amazon rainforest towards a tipping point? | deforestation on regional climate | deforestation on regional climate, are pushing the rainforest towards a "tipping point" where it would irreversibly start to die | wrong end / long context | Model kéo end qua dấu phẩy, sang cả mệnh đề lặp lại câu hỏi ("pushing … tipping point"): end bị hút về phía các từ khóa của câu hỏi. |
| What style was the Warsaw Philharmony edifice built in? | socialist realism | socialist realism style | wrong end / multiple plausible spans | Thêm "style": vô hại về nghĩa nhưng mất EM. |

### 2.3. Wrong span (không chồng lấn với gold)

| Question | Gold | Prediction | Error type | Explanation |
|---|---|---|---|---|
| Why is Priestley usually given credit for being first to discover oxygen? | published his findings first | priority | wrong span / question ambiguity | "Priority" (quyền ưu tiên công bố) là một cách trả lời "why" hợp lý theo nghĩa. Câu hỏi "why" có nhiều span đúng mà gold chỉ có một. |
| For which project did Deke Slayton work with before becoming Director of Flight Crew Operations? | Mercury | Gemini and Apollo programs | wrong span / distractor | Context liệt kê nhiều chương trình. Model chọn chương trình sau thời điểm được hỏi, tức không hiểu quan hệ "before". |
| Where did Apollo 1's crew conduct tests at Kennedy Space Center? | altitude chamber | North American | wrong span / distractor | Chọn một tên riêng (tên công ty) gần các từ khóa, thay vì địa điểm cụ thể. |
| The epidemic in Newcastle was the most what in any British city at the time? | devastating loss | 47% | wrong span / question ambiguity | Câu hỏi kiểu điền chỗ trống ("the most what"), khác dạng câu hỏi thường gặp. Model chọn con số nổi bật gần đó. |
| What rates of health and social problems are in countries with high inequality? | higher | lower | wrong span / distractor | Context nói "lower" cho nước bình đẳng và "higher" cho nước bất bình đẳng. Model chọn nhầm vế: cùng kiểu, cùng câu trúc, chỉ khác điều kiện. |
| If q=9 and a=3,6 or 9, how many primes would be in the progression? | at most one prime number | 1, 2, 4, 5, 7, and 8 | wrong span / insufficient reasoning | Đòi suy luận toán học. Model chỉ thấy dãy số gần từ khóa "q = 9". |
| Where was the Gate of King Hugo? | Tours | Huguon | wrong span / context-dependent | Model chọn tên gọi khác của cổng ("Huguon") thay vì nơi đặt cổng, ở một câu khác. |

## 3. Exp 3 — EM = 0 nhưng F1 > 0

1 837 câu hỏi (17.4%) có EM = 0 nhưng F1 > 0. Ví dụ, tính tay theo token sau chuẩn hóa:

| Question | Gold | Prediction | Overlap | P | R | F1 | Vì sao EM = 0 |
|---|---|---|---|---|---|---|---|
| …what day of the week is the traditional Media Day held prior to a Super Bowl? | Tuesday | Tuesday afternoon | 1 | 1/2 | 1/1 | 0.667 | thừa 1 từ |
| How did Luther respond to Agricola? | six series of theses against Agricola | six series of theses against Agricola and the antinomians | 6 | 6/8 | 6/6 | 0.857 | end quá dài: kéo thêm đối tượng thứ hai |
| How many were killed by plague in Naples in 1656? | about half of Naples' 300,000 inhabitants | about half | 2 | 2/2 | 2/6 | 0.500 | end quá ngắn: thiếu phần "của cái gì" |
| What could justify restrictions on freedom of establishment? | creditor protection, labour rights to participate in work, or the public interest in collecting taxes | creditor protection | 2 | 2/2 | 2/14 | 0.250 | gold là danh sách 3 ý, model chỉ lấy ý đầu |
| Which Doctor was the current Doctor during the 50th Anniversary special? | the Eleventh Doctor | The Day of the Doctor | 1 | 1/3 | 1/2 | 0.400 | sai span hoàn toàn (tên tập phim), chỉ trùng từ "Doctor" |
| What causes strain in structures? | stress tensor | The stress tensor accounts for forces that cause all strains (deformations) including also tensile stresses and compressions | 2 | 2/16 | 2/2 | 0.222 | lấy cả câu thay vì cụm danh từ |

- F1 > 0 không có nghĩa là "gần đúng". Ví dụ "The Day of the Doctor" có F1 0.40 chỉ nhờ trùng một từ phổ biến, dù sai hoàn toàn về nghĩa.
- Ngược lại, "Tuesday afternoon" đúng về nghĩa nhưng EM = 0.
- Hai metric bổ sung cho nhau: EM đo việc chọn đúng biên, F1 đo việc tìm đúng vùng.

## 4. Phân loại theo nguyên nhân

| Nguyên nhân | Bằng chứng |
|---|---|
| **Partial answer / boundary** | 1 567 câu (wrong start + wrong end + wrong boundary), F1 trung bình khoảng 0.5–0.6. Theo độ dài: đáp án 7+ word có EM 43.1% (dự đoán trung bình 6.7 word, gold 10.8 word → **quá ngắn**); đáp án 1 word có dự đoán trung bình 1.98 word → **quá dài**. Model kéo độ dài về khoảng 2–5 word quen thuộc. |
| **Distractor** | Lỗi wrong span ở cùng câu (46.5%) thường là cụm cùng kiểu (số khác, tên khác). Thí nghiệm chèn câu distractor: F1 từ 80.6 xuống **22.4**, 70.5% câu trả lời là đáp án giả. |
| **Multiple plausible spans / question ambiguity** | "60" và "Over 60", "General Pharmaceutical Council (GPhC)", câu hỏi "why". Một phần lỗi EM là do gold không nhất quán, chứ không do model. |
| **Insufficient context / reasoning** | Câu hỏi đòi suy luận (toán, thời gian "before", so sánh "higher" / "lower"): model chọn cụm khớp từ khóa mà không suy luận. |
| **Long context / truncation** | Ít ảnh hưởng ở T = 384: 99% câu hỏi chỉ có 1 window, và chỉ 1.1% lỗi wrong span nằm ở window sau. Ở T = 128 (2.4 window / câu) EM chỉ thấp hơn 0.6 điểm. |
| **Tokenization** | 0.29% đáp án không trùng biên token (`tokenizer_alignment.ipynb`), không đáng kể. Text dự đoán luôn cắt từ context gốc nên không bị lỗi ghép subword. |
| **Lexical overlap** | Câu hỏi ít trùng từ với câu chứa đáp án (< 0.4) có F1 72.5, so với 86.7 khi trùng nhiều (≥ 0.7). Model vẫn phụ thuộc vào trùng từ, nhưng ít hơn nhiều so với baseline (15.3 so với 38.5). |

## 5. Document QA

Trên 25 câu hỏi tự viết về 3 tài liệu (`document_qa.py`): EM 88.0, F1 93.4. Hai lỗi đáng chú ý:

- **"How many points is the baseline worth in LAB 10?"** → "5 points" (gold "10 points"). Tài liệu có ba con số điểm liền nhau (15 / 10 / 5 points). Model chọn số của reflection: đây là lỗi distractor trong cùng câu.
- **"Where do the passages of SQuAD come from?"** → "the corresponding passage" (score 2.25, rất thấp). Câu hỏi dùng "passages … come from", còn tài liệu nói "written … on a set of Wikipedia articles". Model không nối được hai cách diễn đạt.

Với 3 câu hỏi **không có đáp án** trong tài liệu, model vẫn trả về một span ("LAB 10", "grows quadratically…", "researchers at Stanford University"). Đây là hạn chế cơ bản của extractive QA trên SQuAD v1.1. Tuy nhiên score của các câu này (−0.52; 2.98; −0.29) thấp hơn hẳn score của câu có đáp án (median 14.1, thấp nhất 2.25). Một ngưỡng score khoảng 3–5 có thể dùng để trả lời "không tìm thấy", và đó là ý tưởng của SQuAD 2.0.
