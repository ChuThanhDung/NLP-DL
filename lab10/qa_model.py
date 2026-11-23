"""LAB 10 — Extractive QA trên SQuAD v1.1: (question, context) → (start, end).

Gồm:
- load_squad(): chia dữ liệu (train / validation tách theo bài Wikipedia, test = SQuAD dev chính thức);
- char_to_token_span(): đáp án theo ký tự → (start_token, end_token) qua offset mapping;
- encode_windows(): tokenize question + context và tự cắt sliding window (stride);
- prepare_train_features() / prepare_eval_features(): window → nhãn start / end hoặc offset để dự đoán;
- train_qa(): fine-tune, loss = (L_start + L_end) / 2, có checkpoint theo epoch;
- predict_logits() / postprocess(): chọn span (start ≤ end, độ dài ≤ 30 token, nằm trong context), cắt text gốc theo offset;
- normalize_answer(), exact_match(), f1(), evaluate(): EM / F1 theo cách chấm của SQuAD;
- baseline_answer(): keyword overlap → câu ứng viên → span ứng viên.
"""
import collections
import math
import os
import random
import re
import string
import time

import numpy as np
import torch
import torch.nn.functional as F

MODEL_NAME = "distilbert-base-uncased"
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ---------------------------------------------------------------- dữ liệu
def load_squad(n_train=30000, n_val_titles=25, n_val=2000, seed=SEED):
    """train: n_train câu hỏi lấy ngẫu nhiên từ các bài còn lại; validation: n_val câu hỏi từ n_val_titles bài
    Wikipedia giữ riêng (không trùng context với train); test: SQuAD v1.1 dev (10 570 câu hỏi, 48 bài)."""
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad")
    rng = np.random.default_rng(seed)
    titles = sorted(set(ds["train"]["title"]))
    val_titles = set(rng.choice(titles, n_val_titles, replace=False))
    tr = [i for i, t in enumerate(ds["train"]["title"]) if t not in val_titles]
    va = [i for i, t in enumerate(ds["train"]["title"]) if t in val_titles]
    tr = sorted(rng.choice(tr, min(n_train, len(tr)), replace=False).tolist())
    va = sorted(rng.choice(va, min(n_val, len(va)), replace=False).tolist())
    to_list = lambda d: [dict(r) for r in d]
    return {"train": to_list(ds["train"].select(tr)), "validation": to_list(ds["train"].select(va)),
            "test": to_list(ds["validation"])}


# ---------------------------------------------------------------- căn chỉnh ký tự ↔ token
def char_to_token_span(offsets, sequence_ids, ans_start, ans_end):
    """offsets: list (start_char, end_char) của từng token; sequence_ids: 0 = question, 1 = context, None = special.
    ans_end là vị trí sau ký tự cuối (exclusive). Trả về (start_token, end_token) inclusive, hoặc (0, 0) — vị trí
    [CLS] — nếu đáp án không nằm trọn trong phần context của window này (bị cắt hoặc ở window khác).

    Đáp án bắt đầu / kết thúc giữa một token vẫn được gán cho token chứa ký tự đó (start ≤ ans_start < end)."""
    ctx = [i for i, s in enumerate(sequence_ids) if s == 1]
    if ans_end <= ans_start or not ctx or offsets[ctx[0]][0] > ans_start or offsets[ctx[-1]][1] < ans_end:
        return 0, 0
    start_tok = end_tok = None
    for i in ctx:
        s, e = offsets[i]
        if start_tok is None and s <= ans_start < e:
            start_tok = i
        if s < ans_end <= e:
            end_tok = i
    if start_tok is None:            # ans_start rơi vào khoảng trắng giữa hai token → token kế tiếp
        start_tok = next(i for i in ctx if offsets[i][0] >= ans_start)
    if end_tok is None:
        end_tok = max(i for i in ctx if offsets[i][1] <= ans_end)
    return start_tok, end_tok


def encode_windows(tokenizer, examples, max_len=384, stride=128, max_question=64):
    """Tự cắt sliding window: [CLS] question [SEP] context[i : i + L] [SEP], window sau bắt đầu sau window trước
    L − stride token (stride = số token context chồng lấn giữa hai window liên tiếp), cho tới khi phủ hết context.

    Không dùng return_overflowing_tokens của tokenizer: với transformers 5.17, cặp (question, context) chỉ sinh tối đa
    2 window, phần context sau window thứ hai bị bỏ (xem tokenizer_alignment.ipynb).
    → dict list: input_ids, offset_mapping, sequence_ids, overflow_to_sample_mapping (cùng ý nghĩa như của tokenizer)."""
    q = tokenizer([e["question"].strip() for e in examples], add_special_tokens=False, return_offsets_mapping=True)
    c = tokenizer([e["context"] for e in examples], add_special_tokens=False, return_offsets_mapping=True)
    cls, sep = tokenizer.cls_token_id, tokenizer.sep_token_id
    out = {"input_ids": [], "offset_mapping": [], "sequence_ids": [], "overflow_to_sample_mapping": []}
    for i in range(len(examples)):
        q_ids, q_off = q["input_ids"][i][:max_question], q["offset_mapping"][i][:max_question]
        c_ids, c_off = c["input_ids"][i], c["offset_mapping"][i]
        L = max_len - len(q_ids) - 3                                 # số token context mỗi window
        step = max(1, L - stride)
        start = 0
        while True:
            end = min(start + L, len(c_ids))
            out["input_ids"].append([cls] + q_ids + [sep] + c_ids[start:end] + [sep])
            out["offset_mapping"].append([(0, 0)] + list(map(tuple, q_off)) + [(0, 0)] + list(map(tuple, c_off[start:end])) + [(0, 0)])
            out["sequence_ids"].append([None] + [0] * len(q_ids) + [None] + [1] * (end - start) + [None])
            out["overflow_to_sample_mapping"].append(i)
            if end >= len(c_ids):
                break
            start += step
    return out


def prepare_train_features(tokenizer, examples, max_len=384, stride=128):
    """Mỗi example có thể sinh nhiều feature (window). Nhãn start / end lấy từ đáp án đầu tiên."""
    enc = encode_windows(tokenizer, examples, max_len, stride)
    feats = []
    for k in range(len(enc["input_ids"])):
        ex = examples[enc["overflow_to_sample_mapping"][k]]
        a_s = ex["answers"]["answer_start"][0]
        a_e = a_s + len(ex["answers"]["text"][0])
        s, e = char_to_token_span(enc["offset_mapping"][k], enc["sequence_ids"][k], a_s, a_e)
        feats.append({"input_ids": enc["input_ids"][k], "start": s, "end": e,
                      "example": enc["overflow_to_sample_mapping"][k]})
    return feats


def prepare_eval_features(tokenizer, examples, max_len=384, stride=128):
    """Như train, thêm offset (None ở token không thuộc context) để cắt text gốc khi dự đoán."""
    enc = encode_windows(tokenizer, examples, max_len, stride)
    feats = []
    for k in range(len(enc["input_ids"])):
        ex_i = enc["overflow_to_sample_mapping"][k]
        ex = examples[ex_i]
        seq = enc["sequence_ids"][k]
        a_s = ex["answers"]["answer_start"][0]
        s, e = char_to_token_span(enc["offset_mapping"][k], seq, a_s, a_s + len(ex["answers"]["text"][0]))
        feats.append({"input_ids": enc["input_ids"][k], "start": s, "end": e, "example": ex_i,
                      "offsets": [o if seq[i] == 1 else None for i, o in enumerate(enc["offset_mapping"][k])]})
    return feats


def batches(feats, bs, shuffle=False, seed=SEED, pad_id=0):
    idx = np.arange(len(feats))
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    for s in range(0, len(idx), bs):
        b = idx[s:s + bs]
        L = max(len(feats[i]["input_ids"]) for i in b)
        x = torch.full((len(b), L), pad_id); m = torch.zeros(len(b), L, dtype=torch.long)
        for j, i in enumerate(b):
            n = len(feats[i]["input_ids"]); x[j, :n] = torch.tensor(feats[i]["input_ids"]); m[j, :n] = 1
        ys = torch.tensor([feats[i]["start"] for i in b]); ye = torch.tensor([feats[i]["end"] for i in b])
        yield b, x, m, ys, ye


def qa_loss(start_logits, end_logits, ys, ye):
    """L = (L_start + L_end) / 2, mỗi phần là cross-entropy trên các vị trí token."""
    return (F.cross_entropy(start_logits, ys) + F.cross_entropy(end_logits, ye)) / 2


# ---------------------------------------------------------------- train / predict
@torch.no_grad()
def predict_logits(model, feats, bs=64):
    """→ (list start_logits, list end_logits, loss trung bình) cho từng feature. Batch ra NaN được tính lại."""
    model.eval()
    S, E = [None] * len(feats), [None] * len(feats)
    tot = n = 0
    for b, x, m, ys, ye in batches(feats, bs):
        for _ in range(5):
            with torch.autocast(DEVICE.type, dtype=torch.bfloat16):
                out = model(input_ids=x.to(DEVICE), attention_mask=m.to(DEVICE))
            sl, el = out.start_logits.float(), out.end_logits.float()
            if torch.isfinite(sl).all() and torch.isfinite(el).all():
                break
        sl = sl.masked_fill(m.to(DEVICE) == 0, -1e4); el = el.masked_fill(m.to(DEVICE) == 0, -1e4)
        tot += qa_loss(sl, el, ys.to(DEVICE), ye.to(DEVICE)).item() * len(b); n += len(b)
        sl, el = sl.cpu().numpy(), el.cpu().numpy()
        for j, i in enumerate(b):
            L = len(feats[i]["input_ids"])
            S[i], E[i] = sl[j, :L], el[j, :L]
    return S, E, tot / n


def train_qa(model, train_feats, val_feats, val_examples=None, epochs=2, bs=16, lr=3e-5, warmup=0.1,
             evals_per_epoch=4, ckpt=None, seed=SEED, log=print):
    """AdamW + warmup tuyến tính, bf16 autocast, clip 1.0. Ghi train loss (trung bình từng đoạn) và validation loss
    evals_per_epoch lần mỗi epoch (learning curve); cuối mỗi epoch tính EM / F1 trên validation, giữ epoch có F1 cao nhất.
    ckpt: đường dẫn (không đuôi) để lưu checkpoint theo epoch và tiếp tục nếu GPU lỗi giữa chừng."""
    from transformers import get_linear_schedule_with_warmup
    model.to(DEVICE)
    if ckpt and os.path.exists(ckpt + "_final.pt"):
        ck = torch.load(ckpt + "_final.pt", weights_only=False)
        model.load_state_dict(ck["model"]); log(f"  nạp lại model đã train từ {ckpt}_final.pt")
        return ck["info"]
    set_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    per_epoch = math.ceil(len(train_feats) / bs)
    sched = get_linear_schedule_with_warmup(opt, int(warmup * epochs * per_epoch), epochs * per_epoch)
    curve, ep_hist, best, skipped, t_train, start_ep = [], [], (-1, None, 0), 0, 0.0, 1
    if ckpt and os.path.exists(ckpt + "_epoch.pt"):
        ck = torch.load(ckpt + "_epoch.pt", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        curve, ep_hist, best, skipped, t_train, start_ep = ck["curve"], ck["ep_hist"], ck["best"], ck["skipped"], ck["t_train"], ck["epoch"] + 1
        log(f"  tiếp tục từ epoch {start_ep}")
    marks = {int(round(per_epoch * (k + 1) / evals_per_epoch)) for k in range(evals_per_epoch)}
    for ep in range(start_ep, epochs + 1):
        model.train(); t0 = time.time(); run = n = 0
        for step, (_, x, m, ys, ye) in enumerate(batches(train_feats, bs, shuffle=True, seed=seed + ep), 1):
            x, m, ys, ye = x.to(DEVICE), m.to(DEVICE), ys.to(DEVICE), ye.to(DEVICE)
            with torch.autocast(DEVICE.type, dtype=torch.bfloat16):
                out = model(input_ids=x, attention_mask=m)
            loss = qa_loss(out.start_logits.float(), out.end_logits.float(), ys, ye)
            opt.zero_grad(set_to_none=True); loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not (torch.isfinite(loss) and torch.isfinite(gn)):      # GPU đôi khi ra NaN (LAB 07–09) → bỏ bước
                skipped += 1; sched.step(); continue
            opt.step(); sched.step(); run += loss.item(); n += 1
            if step in marks:
                torch.cuda.synchronize(); t_train += time.time() - t0
                _, _, vl = predict_logits(model, val_feats)
                curve.append({"epoch": ep - 1 + step / per_epoch, "train_loss": run / max(n, 1), "val_loss": vl})
                log(f"  epoch {ep - 1 + step / per_epoch:.2f}: train loss {run / max(n, 1):.4f} | val loss {vl:.4f}")
                run = n = 0; model.train(); t0 = time.time()
        res = {"epoch": ep}
        if val_examples is not None:
            S, E, _ = predict_logits(model, val_feats)
            res.update(evaluate(postprocess(val_examples, val_feats, S, E), val_examples))
            log(f"  epoch {ep}: val EM {res['em']:.2f} | val F1 {res['f1']:.2f}")
        ep_hist.append(res)
        score = res.get("f1", ep)
        if score > best[0]:
            best = (score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, ep)
        if ckpt:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "curve": curve,
                        "ep_hist": ep_hist, "best": best, "skipped": skipped, "t_train": t_train, "epoch": ep}, ckpt + "_epoch.pt")
    model.load_state_dict(best[1])
    info = {"curve": curve, "epochs": ep_hist, "best_epoch": best[2], "train_time": t_train, "skipped": skipped,
            "n_train_features": len(train_feats)}
    if ckpt:
        torch.save({"model": best[1], "info": info}, ckpt + "_final.pt"); os.remove(ckpt + "_epoch.pt")
    return info


def postprocess(examples, feats, start_logits, end_logits, n_best=20, max_answer_len=30):
    """Với mỗi example: xét mọi window, n_best vị trí start × n_best vị trí end, giữ cặp hợp lệ (cả hai trong context,
    start ≤ end, ≤ max_answer_len token), điểm = start_logit + end_logit; cắt text gốc theo offset.
    → dict id → {"text", "char_start", "char_end", "score", "window"}."""
    by_ex = collections.defaultdict(list)
    for k, f in enumerate(feats):
        by_ex[f["example"]].append(k)
    preds = {}
    for ei, ex in enumerate(examples):
        best = {"text": "", "char_start": -1, "char_end": -1, "score": -1e9, "window": -1}
        for k in by_ex[ei]:
            off, sl, el = feats[k]["offsets"], start_logits[k], end_logits[k]
            for s in np.argsort(sl)[-n_best:][::-1]:
                if off[s] is None:
                    continue
                for e in np.argsort(el)[-n_best:][::-1]:
                    if off[e] is None or e < s or e - s + 1 > max_answer_len:
                        continue
                    sc = sl[s] + el[e]
                    if sc > best["score"]:
                        best = {"text": ex["context"][off[s][0]:off[e][1]], "char_start": off[s][0], "char_end": off[e][1],
                                "score": float(sc), "window": int(k - by_ex[ei][0])}
        preds[ex["id"]] = best
    return preds


# ---------------------------------------------------------------- EM / F1 (cách chấm của SQuAD v1.1)
def normalize_answer(s):
    """chữ thường → bỏ dấu câu → bỏ mạo từ a / an / the → gộp khoảng trắng."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred, gold):
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1(pred, gold):
    """Token-level F1 giữa hai túi token (đã chuẩn hóa): P = overlap / |pred|, R = overlap / |gold|."""
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    common = collections.Counter(p) & collections.Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    P, R = same / len(p), same / len(g)
    return 2 * P * R / (P + R)


def score_one(pred_text, golds):
    """SQuAD có thể có nhiều đáp án gold: lấy max trên các gold."""
    return max(exact_match(pred_text, g) for g in golds), max(f1(pred_text, g) for g in golds)


def evaluate(preds, examples):
    em = f = 0.0
    for ex in examples:
        a, b = score_one(preds[ex["id"]]["text"], ex["answers"]["text"])
        em += a; f += b
    return {"em": 100 * em / len(examples), "f1": 100 * f / len(examples)}


# ---------------------------------------------------------------- baseline
STOP = set("a an the of in on at to for from by with and or is are was were be been being do does did what which who whom "
           "whose when where why how that this these those it its as into than then there their they he she his her "
           "has have had not no can could would should will may might many much".split())
_TOK = re.compile(r"\w+(?:[-'’.]\w+)*|[^\w\s]")


def _sentences(context):
    """Tách câu đơn giản (dấu . ! ? rồi khoảng trắng + chữ hoa / số / ngoặc), giữ vị trí ký tự."""
    out, start = [], 0
    for m in re.finditer(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])", context):
        out.append((start, m.start())); start = m.end()
    out.append((start, len(context)))
    return out


def baseline_answer(question, context):
    """1) chọn câu có nhiều từ nội dung trùng với câu hỏi nhất (keyword overlap);
    2) trong câu đó, ứng viên là các cụm liên tiếp không chứa từ của câu hỏi (cắt tại từ câu hỏi và dấu phẩy / chấm);
    3) chấm ứng viên: hợp kiểu câu hỏi (when / how many → có số; who / where → có chữ hoa) + gần các từ trùng; độ dài ≤ 6 từ.
    → (text, char_start, char_end)."""
    q_words = [w.lower() for w in _TOK.findall(question)]
    q_content = {w for w in q_words if w not in STOP and w.isalnum()}
    wh = next((w for w in q_words if w in {"when", "who", "where", "what", "which", "how", "why", "whom", "whose"}), "what")
    numeric = wh == "when" or "year" in q_words or ("how" in q_words and ({"many", "much", "long", "old", "far"} & set(q_words)))
    best_sent, best_ov = (0, len(context)), -1
    for a, b in _sentences(context):
        ov = len(q_content & {w.lower() for w in _TOK.findall(context[a:b])})
        if ov > best_ov:
            best_sent, best_ov = (a, b), ov
    a, b = best_sent
    toks = [(m.start() + a, m.end() + a, m.group()) for m in _TOK.finditer(context[a:b])]
    q_all = {w for w in q_words if w.isalnum()}
    hits = [i for i, t in enumerate(toks) if t[2].lower() in q_content]
    cands, cur = [], []
    for i, t in enumerate(toks):
        if t[2].lower() in q_all or t[2] in {",", ".", ";", ":", "(", ")", "\""}:
            if cur: cands.append(cur); cur = []
        else:
            cur.append(i)
    if cur: cands.append(cur)
    best, best_sc = None, -1e9
    for c in cands:
        while c and toks[c[0]][2].lower() in STOP:          # bỏ từ chức năng ở đầu / cuối cụm
            c = c[1:]
        while c and toks[c[-1]][2].lower() in STOP:
            c = c[:-1]
        if not c:
            continue
        c = c[:6]
        words = [toks[i][2] for i in c]
        sc = 0.0
        if numeric:
            sc += 2.0 * any(any(ch.isdigit() for ch in w) for w in words)
        elif wh in {"who", "where", "whom", "whose"}:
            sc += 2.0 * any(w[0].isupper() for w in words)
        else:
            sc += 0.5 * any(w[0].isupper() for w in words)
        if hits:
            sc -= 0.1 * min(abs(i - h) for i in (c[0], c[-1]) for h in hits)
        if sc > best_sc:
            best, best_sc = c, sc
    if best is None:
        return context[a:b].strip(), a, b
    s, e = toks[best[0]][0], toks[best[-1]][1]
    return context[s:e], s, e


def _test_alignment():
    """Các trường hợp đề yêu cầu (mục 16) trên offset giả lập: context "The capital of France is Paris." ."""
    offs = [(0, 0), (0, 3), (4, 11), (12, 14), (15, 21), (22, 24), (25, 30), (30, 31), (0, 0)]
    seq = [None, 1, 1, 1, 1, 1, 1, 1, None]
    assert char_to_token_span(offs, seq, 25, 30) == (6, 6)                  # "Paris"
    assert char_to_token_span(offs, seq, 4, 21) == (2, 4)                   # "capital of France"
    assert char_to_token_span(offs, seq, 27, 30) == (6, 6)                  # bắt đầu giữa token "Pa|ris"
    assert char_to_token_span(offs, seq, 4, 8) == (2, 2)                    # kết thúc giữa token "capi|tal"
    cut = [(0, 0), (0, 3), (4, 11), (12, 14), (0, 0)]                       # window chỉ chứa "The capital of"
    assert char_to_token_span(cut, [None, 1, 1, 1, None], 25, 30) == (0, 0)  # đáp án nằm ngoài window → [CLS]
    assert char_to_token_span(cut, [None, 1, 1, 1, None], 4, 21) == (0, 0)   # đáp án bị cắt dở → [CLS]
    assert f1("France", "Paris France") == 2 * 1 * 0.5 / 1.5 and exact_match("the Paris.", "Paris") == 1.0


if __name__ == "__main__":
    _test_alignment()
    print("char_to_token_span + EM/F1: test pass")
    ctx = "The capital of France is Paris. It has 2.1 million inhabitants."
    for q in ["What is the capital of France?", "How many people live in Paris?"]:
        print(q, "→ baseline:", baseline_answer(q, ctx))
