"""LAB 09 — Baseline NER: word features → Logistic Regression (mỗi token dự đoán độc lập).

Cũng chứa các hàm dùng chung cho notebook và information_extraction.py:
- load_conll(): CoNLL-2003 (bản parquet của eriktks/conll2003);
- bio_to_spans(): chuỗi BIO → danh sách entity (start, end, type), end exclusive;
- entity_scores() / entity_report(): precision, recall, F1 ở mức entity (khớp đúng biên + loại);
- align_labels() / word_predictions(): chuyển nhãn word ↔ subword token (Strategy A / B).

Chạy: python baseline_ner.py
"""
import time
from collections import Counter

import numpy as np

LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
TYPES = ["PER", "ORG", "LOC", "MISC"]


def load_conll():
    """train 14 041 / validation 3 250 / test 3 453 câu; annotation theo word."""
    from datasets import load_dataset
    ds = load_dataset("eriktks/conll2003", revision="refs/convert/parquet")
    assert ds["train"].features["ner_tags"].feature.names == LABELS
    return {k: {"tokens": list(ds[k]["tokens"]), "tags": [[LABELS[t] for t in s] for s in ds[k]["ner_tags"]]}
            for k in ["train", "validation", "test"]}


# ---------------------------------------------------------------- BIO → span, metric
def bio_to_spans(tags):
    """['B-PER','I-PER','O','B-LOC'] → [(0, 2, 'PER'), (3, 4, 'LOC')].

    I-X đứng sau O hoặc sau một loại khác được coi là mở entity mới (giống seqeval mặc định).
    """
    spans, cur = [], None
    for i, t in enumerate(tags):
        if t == "O":
            if cur:
                spans.append(tuple(cur)); cur = None
            continue
        prefix, etype = t.split("-", 1)
        if prefix == "B" or cur is None or cur[2] != etype:
            if cur:
                spans.append(tuple(cur))
            cur = [i, i + 1, etype]
        else:
            cur[1] = i + 1
    if cur:
        spans.append(tuple(cur))
    return spans


def entity_counts(gold_seqs, pred_seqs, key=None):
    """Đếm TP / FP / FN ở mức entity. key(span, words_idx) → nhóm (vd: loại, độ dài); None = tổng."""
    cnt = Counter()
    for sid, (g, p) in enumerate(zip(gold_seqs, pred_seqs)):
        gs, ps = set(bio_to_spans(g)), set(bio_to_spans(p))
        for s in gs & ps:
            cnt[(key(s, sid) if key else "all", "tp")] += 1
        for s in ps - gs:
            cnt[(key(s, sid) if key else "all", "fp")] += 1
        for s in gs - ps:
            cnt[(key(s, sid) if key else "all", "fn")] += 1
    return cnt


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def entity_scores(gold_seqs, pred_seqs):
    c = entity_counts(gold_seqs, pred_seqs)
    p, r, f = prf(c[("all", "tp")], c[("all", "fp")], c[("all", "fn")])
    return {"precision": p, "recall": r, "f1": f,
            "tp": c[("all", "tp")], "fp": c[("all", "fp")], "fn": c[("all", "fn")]}


def entity_report(gold_seqs, pred_seqs, key=lambda s, sid: s[2]):
    """Bảng P / R / F1 theo nhóm. Mặc định nhóm theo loại entity. Precision / recall của một nhóm
    dùng nhóm của span dự đoán cho FP và của span gold cho FN."""
    c = entity_counts(gold_seqs, pred_seqs, key)
    rows = {}
    for g in sorted({k[0] for k in c}, key=str):
        tp, fp, fn = c[(g, "tp")], c[(g, "fp")], c[(g, "fn")]
        p, r, f = prf(tp, fp, fn)
        rows[g] = {"precision": p, "recall": r, "f1": f, "support": tp + fn, "tp": tp, "fp": fp, "fn": fn}
    return rows


def token_accuracy(gold_seqs, pred_seqs):
    g = [t for s in gold_seqs for t in s]
    p = [t for s in pred_seqs for t in s]
    return float(np.mean([a == b for a, b in zip(g, p)]))


def token_f1(gold_seqs, pred_seqs):
    """Token-level F1 (micro) trên các token có nhãn khác O — đối chiếu với entity-level."""
    g = [t for s in gold_seqs for t in s]
    p = [t for s in pred_seqs for t in s]
    tp = sum(a == b != "O" for a, b in zip(g, p))
    fp = sum(b != "O" and a != b for a, b in zip(g, p))
    fn = sum(a != "O" and a != b for a, b in zip(g, p))
    return prf(tp, fp, fn)


# ---------------------------------------------------------------- alignment word ↔ subword
IGNORE = -100


def align_labels(word_ids, word_tags, strategy="A"):
    """Nhãn word-level → nhãn token-level theo word_ids() của tokenizer.

    A: subword đầu của mỗi word nhận nhãn word; subword sau, [CLS], [SEP], [PAD] → IGNORE (-100).
    B: subword sau cũng được học: nhận I-X nếu word là B-X / I-X, nhận O nếu word là O.
    Trả về list chỉ số nhãn (int) cùng độ dài với word_ids.
    """
    out, prev = [], None
    for wid in word_ids:
        if wid is None:
            out.append(IGNORE)
        elif wid != prev:
            out.append(LABELS.index(word_tags[wid]))
        elif strategy == "A":
            out.append(IGNORE)
        else:
            t = word_tags[wid]
            out.append(LABELS.index("I-" + t[2:] if t != "O" else "O"))
        prev = wid
    return out


def word_predictions(word_ids, token_pred_ids, n_words):
    """Dự đoán token-level → nhãn word-level: lấy dự đoán ở subword đầu của mỗi word (dùng cho cả A và B,
    để hai strategy được đánh giá trên cùng đơn vị word như gold). Word bị cắt bởi truncation → O."""
    tags, prev = ["O"] * n_words, None
    for wid, p in zip(word_ids, token_pred_ids):
        if wid is not None and wid != prev:
            tags[wid] = LABELS[p]
        prev = wid
    return tags


# ---------------------------------------------------------------- baseline
def token_features(words, i):
    """Feature cho từ thứ i: dạng từ, chữ hoa, tiền tố / hậu tố, 2 từ mỗi bên."""
    w = words[i]

    def ctx(j):
        return words[j] if 0 <= j < len(words) else ("<S>" if j < 0 else "</S>")

    f = {
        "bias": 1.0,
        "w_lower=" + w.lower(): 1.0,
        "w_raw=" + w: 1.0,
        "is_title": float(w.istitle()),
        "is_upper": float(w.isupper()),
        "is_digit": float(w.isdigit()),
        "has_digit": float(any(c.isdigit() for c in w)),
        "has_hyphen": float("-" in w),
        "pre2=" + w[:2].lower(): 1.0,
        "pre3=" + w[:3].lower(): 1.0,
        "suf2=" + w[-2:].lower(): 1.0,
        "suf3=" + w[-3:].lower(): 1.0,
        "first_word": float(i == 0),
    }
    for d in (-2, -1, 1, 2):
        c = ctx(i + d)
        f[f"w{d:+d}=" + c.lower()] = 1.0
        f[f"w{d:+d}_title"] = float(c.istitle())
    return f


def featurize(sentences):
    return [token_features(s, i) for s in sentences for i in range(len(s))]


def train_baseline(data, C=1.0, max_iter=300):
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    vec = DictVectorizer()
    X = vec.fit_transform(featurize(data["train"]["tokens"]))
    y = [t for s in data["train"]["tags"] for t in s]
    clf = LogisticRegression(C=C, max_iter=max_iter)
    clf.fit(X, y)
    return vec, clf


def predict_baseline(vec, clf, sentences):
    flat = clf.predict(vec.transform(featurize(sentences)))
    out, k = [], 0
    for s in sentences:
        out.append(list(flat[k:k + len(s)])); k += len(s)
    return out


def _test_bio_to_spans():
    assert bio_to_spans(["B-PER", "I-PER", "O", "B-ORG", "O", "B-LOC", "O"]) == [(0, 2, "PER"), (3, 4, "ORG"), (5, 6, "LOC")]
    assert bio_to_spans(["B-ORG", "I-ORG", "I-ORG", "I-ORG"]) == [(0, 4, "ORG")]
    assert bio_to_spans(["O", "I-PER", "I-PER"]) == [(1, 3, "PER")]               # I sau O: mở entity mới
    assert bio_to_spans(["B-PER", "I-LOC"]) == [(0, 1, "PER"), (1, 2, "LOC")]      # đổi loại: tách
    assert bio_to_spans(["B-LOC", "B-LOC"]) == [(0, 1, "LOC"), (1, 2, "LOC")]      # B liền B: 2 entity
    # ví dụ của đề (mục 19): New York → B-LOC O
    g, p = [["B-LOC", "I-LOC"]], [["B-LOC", "O"]]
    assert token_accuracy(g, p) == 0.5 and entity_scores(g, p)["f1"] == 0.0


if __name__ == "__main__":
    from seqeval.metrics import f1_score as seq_f1
    _test_bio_to_spans()
    print("bio_to_spans: 6 test pass")
    data = load_conll()
    print({k: len(v["tokens"]) for k, v in data.items()})
    t0 = time.time()
    vec, clf = train_baseline(data)
    print(f"train {time.time() - t0:.0f}s, {len(vec.vocabulary_)} feature")
    for split in ["validation", "test"]:
        pred = predict_baseline(vec, clf, data[split]["tokens"])
        gold = data[split]["tags"]
        s = entity_scores(gold, pred)
        print(f"{split}: entity P {s['precision']:.4f} R {s['recall']:.4f} F1 {s['f1']:.4f} | "
              f"token acc {token_accuracy(gold, pred):.4f} | seqeval F1 {seq_f1(gold, pred):.4f}")
