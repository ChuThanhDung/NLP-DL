"""LAB 09 — Information Extraction (Option A — News): raw text → NER → entity → JSON.

Dùng model BERT (Strategy A) đã fine-tune trong ner_finetuning.ipynb (thư mục model_ner/).
Đánh giá trên 30 câu tin tức tự viết, có gold annotation, tính P / R / F1 ở mức entity.

Chạy: python information_extraction.py            (đánh giá 30 câu + in ví dụ JSON)
      python information_extraction.py "text ..."  (trích xuất một đoạn văn bản)
"""
import json
import re
import sys
from collections import Counter

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from baseline_ner import LABELS, bio_to_spans, prf, word_predictions

FIELD = {"PER": "PERSON", "ORG": "ORGANIZATION", "LOC": "LOCATION", "MISC": "MISC"}
_WORD = re.compile(r"[A-Z](?:\.[A-Z])+\.?|'s\b|\w+(?:-\w+)*|[^\w\s]")


def split_words(text):
    """Tách word giống CoNLL: dấu câu và 's thành token riêng; giữ nguyên 'U.S.', 'Saint-Germain', '2-1'."""
    return _WORD.findall(text)


class Extractor:
    def __init__(self, path="model_ner", device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForTokenClassification.from_pretrained(path).to(self.device).eval()

    @torch.no_grad()
    def tag(self, words):
        enc = self.tok(words, is_split_into_words=True, truncation=True, max_length=128, return_tensors="pt")
        logits = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits[0]
        return word_predictions(enc.word_ids(0), logits.argmax(-1).tolist(), len(words))

    def entities(self, text):
        """→ list (entity text, loại), theo thứ tự xuất hiện."""
        words = split_words(text)
        return [(" ".join(words[a:e]), t) for a, e, t in bio_to_spans(self.tag(words))]

    def extract(self, text):
        """→ dict cấu trúc {"PERSON": [...], "ORGANIZATION": [...], "LOCATION": [...], "MISC": [...]} (bỏ trùng)."""
        out = {v: [] for v in FIELD.values()}
        for s, t in self.entities(text):
            if s not in out[FIELD[t]]:
                out[FIELD[t]].append(s)
        return {k: v for k, v in out.items() if v}


# 30 câu tin tức tự viết (không lấy từ CoNLL). Gold: (chuỗi entity đúng như sau khi tách word, loại).
GOLD = [
    ("Barack Obama visited Paris on Monday.", [("Barack Obama", "PER"), ("Paris", "LOC")]),
    ("Apple chief executive Tim Cook met Chinese officials in Beijing.", [("Apple", "ORG"), ("Tim Cook", "PER"), ("Chinese", "MISC"), ("Beijing", "LOC")]),
    ("The United Nations Security Council will meet in New York next week.", [("United Nations Security Council", "ORG"), ("New York", "LOC")]),
    ("Manchester United beat Chelsea 2-1 at Old Trafford on Saturday.", [("Manchester United", "ORG"), ("Chelsea", "ORG"), ("Old Trafford", "LOC")]),
    ("German Chancellor Olaf Scholz arrived in Warsaw for talks with Polish leaders.", [("German", "MISC"), ("Olaf Scholz", "PER"), ("Warsaw", "LOC"), ("Polish", "MISC")]),
    ("Microsoft agreed to buy Activision Blizzard for $69 billion.", [("Microsoft", "ORG"), ("Activision Blizzard", "ORG")]),
    ("Vietnamese Prime Minister Pham Minh Chinh visited Tokyo to meet Japanese business leaders.", [("Vietnamese", "MISC"), ("Pham Minh Chinh", "PER"), ("Tokyo", "LOC"), ("Japanese", "MISC")]),
    ("Washington said the sanctions against Russia would remain in place.", [("Washington", "LOC"), ("Russia", "LOC")]),
    ("George Washington University announced a new research centre.", [("George Washington University", "ORG")]),
    ("Denzel Washington won an Oscar for his role in Training Day.", [("Denzel Washington", "PER"), ("Oscar", "MISC"), ("Training Day", "MISC")]),
    ("Shares of Toyota fell 3 percent on the Tokyo Stock Exchange.", [("Toyota", "ORG"), ("Tokyo Stock Exchange", "ORG")]),
    ("The European Central Bank kept interest rates unchanged, said Christine Lagarde.", [("European Central Bank", "ORG"), ("Christine Lagarde", "PER")]),
    ("Rafael Nadal withdrew from the French Open because of a hip injury.", [("Rafael Nadal", "PER"), ("French Open", "MISC")]),
    ("Floods in northern Italy forced thousands to leave their homes near Bologna.", [("Italy", "LOC"), ("Bologna", "LOC")]),
    ("Jordan scored 30 points as the Chicago Bulls defeated the Boston Celtics.", [("Jordan", "PER"), ("Chicago Bulls", "ORG"), ("Boston Celtics", "ORG")]),
    ("King Abdullah of Jordan met the Egyptian president in Cairo.", [("Abdullah", "PER"), ("Jordan", "LOC"), ("Egyptian", "MISC"), ("Cairo", "LOC")]),
    ("Amazon opened a new warehouse in Ohio, creating 1,000 jobs.", [("Amazon", "ORG"), ("Ohio", "LOC")]),
    ("Brazilian police arrested the former governor of Rio de Janeiro.", [("Brazilian", "MISC"), ("Rio de Janeiro", "LOC")]),
    ("The World Health Organization warned of a new flu strain in Southeast Asia.", [("World Health Organization", "ORG"), ("Southeast Asia", "LOC")]),
    ("Elon Musk said Tesla would build a factory near Berlin.", [("Elon Musk", "PER"), ("Tesla", "ORG"), ("Berlin", "LOC")]),
    ("Kenya's Eliud Kipchoge won the Berlin Marathon for a fifth time.", [("Kenya", "LOC"), ("Eliud Kipchoge", "PER"), ("Berlin Marathon", "MISC")]),
    ("The Kremlin denied reports that Vladimir Putin was ill.", [("Kremlin", "ORG"), ("Vladimir Putin", "PER")]),
    ("Samsung Electronics reported record profits driven by chip sales in the United States.", [("Samsung Electronics", "ORG"), ("United States", "LOC")]),
    ("Hanoi and Ho Chi Minh City recorded heavy rain on Tuesday.", [("Hanoi", "LOC"), ("Ho Chi Minh City", "LOC")]),
    ("Novak Djokovic beat Carlos Alcaraz in the Wimbledon final.", [("Novak Djokovic", "PER"), ("Carlos Alcaraz", "PER"), ("Wimbledon", "MISC")]),
    ("The International Monetary Fund cut its growth forecast for Argentina.", [("International Monetary Fund", "ORG"), ("Argentina", "LOC")]),
    ("Paris Saint-Germain signed a Brazilian striker from Santos.", [("Paris Saint-Germain", "ORG"), ("Brazilian", "MISC"), ("Santos", "ORG")]),
    ("Officials at NASA said the Artemis mission would launch from Florida.", [("NASA", "ORG"), ("Artemis", "MISC"), ("Florida", "LOC")]),
    ("Prime Minister Narendra Modi spoke to reporters in New Delhi after the G20 summit.", [("Narendra Modi", "PER"), ("New Delhi", "LOC"), ("G20", "MISC")]),
    ("A spokesman for Volkswagen said the strike at its Wolfsburg plant had ended.", [("Volkswagen", "ORG"), ("Wolfsburg", "LOC")]),
]


def evaluate(ex, verbose=True):
    """Entity-level: một entity dự đoán đúng khi trùng cả chuỗi lẫn loại với một entity gold của cùng câu."""
    tp = fp = fn = 0
    per_type = Counter()
    errors = []
    for text, g in GOLD:
        p = ex.entities(text)
        gc, pc = Counter(g), Counter(p)
        hit = gc & pc
        tp += sum(hit.values()); fp += sum((pc - gc).values()); fn += sum((gc - pc).values())
        for (s, t), n in hit.items(): per_type[(t, "tp")] += n
        for (s, t), n in (pc - gc).items(): per_type[(t, "fp")] += n
        for (s, t), n in (gc - pc).items(): per_type[(t, "fn")] += n
        if gc != pc:
            errors.append((text, sorted((gc - pc).elements()), sorted((pc - gc).elements())))
    p, r, f = prf(tp, fp, fn)
    if verbose:
        print(f"30 câu, {sum(len(g) for _, g in GOLD)} entity gold | TP {tp} FP {fp} FN {fn} | P {p:.4f} R {r:.4f} F1 {f:.4f}")
        for t in FIELD:
            print(f"  {t:4}: P {prf(per_type[(t, 'tp')], per_type[(t, 'fp')], per_type[(t, 'fn')])[0]:.3f} "
                  f"R {prf(per_type[(t, 'tp')], per_type[(t, 'fp')], per_type[(t, 'fn')])[1]:.3f} "
                  f"F1 {prf(per_type[(t, 'tp')], per_type[(t, 'fp')], per_type[(t, 'fn')])[2]:.3f} "
                  f"(gold {per_type[(t, 'tp')] + per_type[(t, 'fn')]})")
        print(f"\n{len(errors)} câu có lỗi:")
        for text, miss, extra in errors:
            print(f"  - {text}\n      thiếu: {miss}\n      thừa:  {extra}")
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


if __name__ == "__main__":
    ex = Extractor()
    if len(sys.argv) > 1:
        print(json.dumps(ex.extract(" ".join(sys.argv[1:])), ensure_ascii=False, indent=2))
    else:
        for text in [GOLD[0][0], GOLD[1][0]]:
            print(text); print(json.dumps(ex.extract(text), ensure_ascii=False, indent=2)); print()
        evaluate(ex)
