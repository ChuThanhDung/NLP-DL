"""LAB 10 — Document QA: user question → document (đã cho sẵn) → QA model → answer span.

Dùng model DistilBERT đã fine-tune trong qa_finetuning.ipynb (thư mục model_qa/, max_length 384, stride 128).
Context được đưa trực tiếp (không có retrieval — đó là LAB 11). Với nhiều tài liệu, hỏi từng tài liệu rồi lấy
span có điểm cao nhất.

Chạy: python document_qa.py                       (đánh giá bộ câu hỏi có gold + câu hỏi không có đáp án)
      python document_qa.py "When is LAB 10 due?"  (hỏi một câu trên các tài liệu dưới đây)
"""
import sys

import numpy as np
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

import qa_model as qm

DOCS = {
    "course_faq": (
        "This FAQ describes the course Natural Language Processing and Applications, taught in the first semester of 2026 at "
        "the VNU University of Science. The instructor and teaching assistant is Pham Ngoc Hai. Each lab must be submitted "
        "before 23:59 on its deadline. LAB 08 covers pretrained Transformers and BERT fine-tuning and is due on 11 November "
        "2026. LAB 09 covers named entity recognition and information extraction on the CoNLL-2003 dataset and is due on "
        "18 November 2026. LAB 10 covers extractive question answering on SQuAD and is due on 25 November 2026. Every lab "
        "is graded out of 100 points. In LAB 10, the tokenization and span alignment component is worth 15 points, the "
        "baseline is worth 10 points and the reflection is worth 5 points. Students may use AI tools for documentation, "
        "debugging and boilerplate code, but they must not ask AI to write the error analysis or the reflection. LAB 11 "
        "moves from a known context to semantic retrieval over a large document collection, and LAB 12 combines retrieval "
        "with generation, an approach known as retrieval-augmented generation or RAG."
    ),
    "transformers": (
        "The Transformer architecture was introduced in 2017 in the paper Attention Is All You Need by researchers at Google. "
        "It replaced recurrence with self-attention, which lets every token attend to every other token in a single step. "
        "Because attention compares all pairs of positions, its cost grows quadratically with the sequence length. "
        "BERT, released by Google in 2018, is an encoder-only Transformer pretrained with masked language modeling, in which "
        "about 15 percent of the input tokens are hidden and the model must predict them from both sides of the context. "
        "The base version of BERT has 12 layers, a hidden size of 768 and about 110 million parameters, and it reads at most "
        "512 tokens. DistilBERT is a smaller model obtained by knowledge distillation from BERT; it keeps 6 of the 12 layers, "
        "has about 66 million parameters and runs roughly 60 percent faster while retaining about 97 percent of BERT's "
        "language understanding. DistilBERT does not use token type embeddings, so the question and the context are "
        "separated only by the special separator token. For extractive question answering, a linear layer on top of the "
        "encoder produces two scores for each token: one for being the start of the answer and one for being the end. "
        "Long documents that do not fit into the maximum sequence length are split into overlapping windows, and the "
        "number of tokens shared by two consecutive windows is called the stride. "
        "Because self-attention itself ignores word order, the original Transformer added sinusoidal positional encodings "
        "to the input embeddings, so that each position receives a fixed pattern of sine and cosine values. The base model "
        "described in the paper used an encoder and a decoder with 6 layers each, a model dimension of 512 and 8 attention "
        "heads, so that each head works with vectors of dimension 64. Every layer combines multi-head attention with a "
        "position-wise feed-forward network, and each sub-layer is wrapped by a residual connection followed by layer "
        "normalization. The model was first evaluated on machine translation, where it outperformed recurrent and "
        "convolutional systems on English-to-German and English-to-French benchmarks while requiring much less training time."
    ),
    "squad": (
        "The Stanford Question Answering Dataset, known as SQuAD, was released in 2016 by researchers at Stanford University. "
        "Its first version contains more than 100,000 questions written by crowdworkers on a set of Wikipedia articles, and "
        "the answer to every question is a segment of text from the corresponding passage. SQuAD 2.0, released in 2018, "
        "added over 50,000 unanswerable questions written to look similar to answerable ones, so that a system must also "
        "decide when no answer is supported by the passage. Systems are evaluated with two metrics: exact match, which "
        "requires the predicted answer to be identical to one of the reference answers after normalization, and F1, which "
        "measures the overlap of words between the prediction and the reference. Before comparison, answers are converted "
        "to lowercase and punctuation and the articles a, an and the are removed. Each question in the development set has "
        "up to three reference answers, and the score of a prediction is the maximum over these references."
    ),
}

# Câu hỏi có gold (tự viết, đáp án là một đoạn của tài liệu). Một số dùng từ khác với tài liệu (diễn đạt lại).
GOLD = [
    ("course_faq", "Who is the instructor of the course?", ["Pham Ngoc Hai"]),
    ("course_faq", "When is LAB 10 due?", ["25 November 2026"]),
    ("course_faq", "Which dataset is used for named entity recognition in LAB 09?", ["CoNLL-2003", "the CoNLL-2003 dataset"]),
    ("course_faq", "How many points is the baseline worth in LAB 10?", ["10 points", "10"]),
    ("course_faq", "What is the latest time a lab can be handed in on its deadline day?", ["23:59", "before 23:59"]),
    ("course_faq", "At which university is the course taught?", ["VNU University of Science", "the VNU University of Science"]),
    ("course_faq", "What does RAG stand for?", ["retrieval-augmented generation"]),
    ("course_faq", "What are students not allowed to ask AI to write?", ["the error analysis or the reflection", "error analysis or the reflection"]),
    ("transformers", "In which year was the Transformer introduced?", ["2017"]),
    ("transformers", "What did the Transformer replace recurrence with?", ["self-attention"]),
    ("transformers", "How does the cost of attention grow with sequence length?", ["quadratically"]),
    ("transformers", "What fraction of input tokens does BERT hide during pretraining?", ["about 15 percent", "15 percent"]),
    ("transformers", "How many layers does DistilBERT keep?", ["6 of the 12 layers", "6"]),
    ("transformers", "How is DistilBERT obtained from BERT?", ["by knowledge distillation from BERT", "knowledge distillation"]),
    ("transformers", "What is the maximum number of tokens BERT can read?", ["512 tokens", "512"]),
    ("transformers", "What is the overlap between two consecutive windows called?", ["the stride", "stride"]),
    ("transformers", "Which company created BERT?", ["Google"]),
    ("transformers", "How many attention heads did the base Transformer use?", ["8"]),
    ("transformers", "What kind of positional encodings did the original Transformer use?", ["sinusoidal positional encodings", "sinusoidal"]),
    ("transformers", "On which task was the Transformer first evaluated?", ["machine translation"]),
    ("squad", "Who released SQuAD?", ["researchers at Stanford University", "Stanford University"]),
    ("squad", "How many unanswerable questions were added in SQuAD 2.0?", ["over 50,000", "50,000"]),
    ("squad", "Where do the passages of SQuAD come from?", ["Wikipedia articles", "a set of Wikipedia articles"]),
    ("squad", "Which metric requires the prediction to be identical to a reference answer?", ["exact match"]),
    ("squad", "How many reference answers can a development question have?", ["up to three", "three"]),
]
# Không có đáp án trong tài liệu: extractive QA (SQuAD v1.1) vẫn buộc phải trả về một span.
UNANSWERABLE = [
    ("course_faq", "What is the name of the room where the labs take place?"),
    ("transformers", "How much did it cost to train BERT?"),
    ("squad", "Who won the SQuAD leaderboard in 2020?"),
]


class DocumentQA:
    def __init__(self, path="model_qa", max_len=384, stride=128):
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForQuestionAnswering.from_pretrained(path).to(qm.DEVICE).eval()
        self.max_len, self.stride = max_len, stride

    def answer(self, question, context):
        """→ dict text, char_start, char_end, score (start_logit + end_logit), window, n_windows."""
        ex = [{"id": "q", "question": question, "context": context, "answers": {"text": [""], "answer_start": [0]}}]
        feats = qm.prepare_eval_features(self.tok, ex, self.max_len, self.stride)
        S, E, _ = qm.predict_logits(self.model, feats)
        return {**qm.postprocess(ex, feats, S, E)["q"], "n_windows": len(feats)}

    def ask(self, question, docs=DOCS):
        """Hỏi trên nhiều tài liệu: chạy từng tài liệu, lấy span có điểm cao nhất."""
        res = {name: self.answer(question, text) for name, text in docs.items()}
        best = max(res, key=lambda n: res[n]["score"])
        return {"document": best, **res[best]}


def evaluate(qa, verbose=True):
    rows = []
    for doc, q, golds in GOLD:
        a = qa.answer(q, DOCS[doc])
        em, f = qm.score_one(a["text"], golds)
        rows.append((doc, q, golds[0], a["text"], em, f, a["score"], a["n_windows"]))
    em = 100 * np.mean([r[4] for r in rows]); f = 100 * np.mean([r[5] for r in rows])
    if verbose:
        print(f"{len(rows)} câu hỏi có gold: EM {em:.2f} | F1 {f:.2f}")
        for doc, q, g, p, e, f1_, sc, nw in rows:
            mark = "✓" if e else ("~" if f1_ > 0 else "✗")
            print(f"  {mark} [{doc}, {nw} window] {q}\n      gold: {g!r} | pred: {p!r} | F1 {f1_:.2f} | score {sc:.2f}")
        print("\nCâu hỏi KHÔNG có đáp án trong tài liệu (model vẫn phải trả về một span):")
        for doc, q in UNANSWERABLE:
            a = qa.answer(q, DOCS[doc])
            print(f"  [{doc}] {q}\n      pred: {a['text']!r} | score {a['score']:.2f}")
        scores = [r[6] for r in rows]
        print(f"\nScore của câu có đáp án: min {min(scores):.2f}, median {np.median(scores):.2f}")
        print("\nHỏi trên cả 3 tài liệu (chọn tài liệu theo score):")
        for q in ["When is LAB 10 due?", "How many parameters does DistilBERT have?", "When was SQuAD released?"]:
            r = qa.ask(q)
            print(f"  {q} → {r['text']!r} (từ {r['document']}, score {r['score']:.2f})")
    return {"em": em, "f1": f, "n": len(rows)}


if __name__ == "__main__":
    qa = DocumentQA()
    toks = {n: len(qa.tok(t)["input_ids"]) for n, t in DOCS.items()}
    print("Số token mỗi tài liệu:", toks, "(> 384 − câu hỏi → nhiều window)\n")
    if len(sys.argv) > 1:
        r = qa.ask(" ".join(sys.argv[1:]))
        print(f"Answer: {r['text']!r} (tài liệu {r['document']}, score {r['score']:.2f})")
    else:
        evaluate(qa)
