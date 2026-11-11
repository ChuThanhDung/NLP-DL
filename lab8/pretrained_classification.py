"""LAB 08 — Pretrained Transformer (DistilBERT) cho phân loại sentiment IMDb.

Gồm:
- dữ liệu: IMDb chia giống LAB 05–07, Yelp / Amazon cho domain shift;
- tokenize một lần, cắt / pad theo max_length khi tạo batch;
- PretrainedClassifier: encoder pretrained → [CLS] → dropout → Linear (head task-specific);
- ScratchTransformer: Transformer nhỏ khởi tạo ngẫu nhiên (cấu hình LAB 07), dùng cùng tokenizer;
- train_model (fine-tuning / from scratch) và frozen mode (cache feature [CLS] / mean, chỉ train head);
- predict_proba, metrics.
"""
import math
import random
import re
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

MODEL_NAME = "distilbert-base-uncased"
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_DTYPE = torch.bfloat16


def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ---------------------------------------------------------------- dữ liệu
_BR = re.compile(r"<br\s*/?>", re.I)


def clean(t):
    """Chỉ bỏ thẻ <br /> và ký tự '\\n' dạng chữ; tokenizer của model tự lo phần còn lại."""
    return re.sub(r"\s+", " ", _BR.sub(" ", str(t)).replace("\\n", " ")).strip()


def load_imdb(seed=SEED):
    """Train 19 923 / validation 4 981 / test 25 000 — cùng cách chia với LAB 05–07."""
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/imdb")
    train_raw, test = ds["train"].to_pandas(), ds["test"].to_pandas()
    train_raw = train_raw.drop_duplicates("text").reset_index(drop=True)
    tr, va = train_test_split(train_raw, test_size=0.2, stratify=train_raw.label, random_state=seed)
    out = {}
    for k, d in [("train", tr), ("val", va), ("test", test)]:
        d = d.reset_index(drop=True)
        out[k] = pd.DataFrame({"text": d.text.map(clean), "label": d.label.values})
    return out


DOMAINS = {
    "yelp": ("fancyzhx/yelp_polarity", "plain_text/test-00000-of-00001.parquet"),
    "amazon": ("fancyzhx/amazon_polarity", "amazon_polarity/test-00000-of-00001.parquet"),
}


def load_domain(name, n=2000, seed=SEED):
    """Lấy n mẫu cân bằng nhãn từ tập test của domain khác (chỉ tải đúng file test)."""
    from datasets import load_dataset
    repo, f = DOMAINS[name]
    d = load_dataset("parquet", data_files=f"hf://datasets/{repo}/{f}", split="train").to_pandas()
    if "content" in d:                                  # Amazon: ghép title + content
        d["text"] = d.title.fillna("") + ". " + d.content.fillna("")
    d = d.groupby("label").sample(n // 2, random_state=seed)
    d = d.sample(frac=1, random_state=seed).reset_index(drop=True)
    return pd.DataFrame({"text": d.text.map(clean), "label": d.label.values})


def get_tokenizer(name=MODEL_NAME):
    return AutoTokenizer.from_pretrained(name)


def encode(tokenizer, texts, max_len=512):
    """Tokenize một lần (có [CLS] … [SEP]); cắt ngắn hơn được làm khi tạo batch."""
    return tokenizer(list(texts), truncation=True, max_length=max_len)["input_ids"]


def make_batches(ids, labels, max_len, bs, shuffle=False, seed=SEED):
    """Sinh (input_ids, attention_mask, labels). Cắt về max_len và giữ [SEP] cuối; pad tới câu dài nhất của batch."""
    idx = np.arange(len(ids))
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    else:                                               # khi dự đoán: gom câu cùng độ dài cho nhanh
        idx = np.argsort([min(len(x), max_len) for x in ids], kind="stable")
    for s in range(0, len(idx), bs):
        b = idx[s:s + bs]
        seqs = [ids[i] if len(ids[i]) <= max_len else ids[i][:max_len - 1] + ids[i][-1:] for i in b]
        L = max(map(len, seqs))
        x = torch.zeros(len(b), L, dtype=torch.long)
        m = torch.zeros(len(b), L, dtype=torch.long)
        for j, q in enumerate(seqs):
            x[j, :len(q)] = torch.tensor(q)
            m[j, :len(q)] = 1
        yield b, x, m, (torch.tensor(np.asarray(labels)[b]) if labels is not None else None)


# ---------------------------------------------------------------- model
class PretrainedClassifier(nn.Module):
    """Pretrained encoder → vector [CLS] (vị trí 0) → dropout → Linear → logits (2 lớp).

    Head (dropout + Linear) là phần task-specific, khởi tạo ngẫu nhiên; encoder lấy trọng số pretrained.
    """

    def __init__(self, name=MODEL_NAME, n_classes=2, dropout=0.1, freeze_encoder=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_classes)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def cls_vector(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # (B, T, H)
        return h[:, 0]                                                                         # (B, H)

    def forward(self, input_ids, attention_mask):
        return self.head(self.dropout(self.cls_vector(input_ids, attention_mask)))


class ScratchTransformer(nn.Module):
    """Transformer khởi tạo ngẫu nhiên, cấu hình LAB 07: d_model 128, 4 head, d_ff 256, 1 layer.

    Dùng cùng tokenizer với model pretrained (vocab 30 522), sinusoidal PE; pooling "mean" (trên token thật, như LAB 07) hoặc "cls".
    """

    def __init__(self, vocab_size, d_model=128, n_heads=4, d_ff=256, n_layers=1, max_len=512,
                 dropout=0.3, block_dropout=0.1, n_classes=2, pad_id=0, pool="mean", scale_emb=False):
        super().__init__()
        self.pool = pool
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, block_dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, n_classes)
        self.scale = math.sqrt(d_model) if scale_emb else 1.0

    def forward(self, input_ids, attention_mask):
        x = self.emb(input_ids) * self.scale + self.pe[: input_ids.size(1)]
        h = self.encoder(x, src_key_padding_mask=attention_mask == 0)
        if self.pool == "cls":
            v = h[:, 0]
        else:
            m = attention_mask.unsqueeze(-1).to(h.dtype)
            v = (h * m).sum(1) / m.sum(1)
        return self.head(self.dropout(v))


def count_params(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


# ---------------------------------------------------------------- train / predict
@torch.no_grad()
def predict_proba(model, ids, max_len, bs=64):
    """P(positive) cho từng mẫu (giữ thứ tự gốc). Batch ra NaN (lỗi GPU chập chờn, xem LAB 07) được tính lại."""
    model.eval()
    out = np.zeros(len(ids), dtype=np.float32)
    for b, x, m, _ in make_batches(ids, None, max_len, bs):
        x, m = x.to(DEVICE), m.to(DEVICE)
        for _ in range(5):
            with torch.autocast(DEVICE.type, dtype=AMP_DTYPE):
                logits = model(x, m).float()
            if torch.isfinite(logits).all():
                break
        out[b] = torch.softmax(logits, -1)[:, 1].cpu().numpy()
    return out


def evaluate_loss(model, ids, labels, max_len, bs=64):
    p = np.clip(predict_proba(model, ids, max_len, bs), 1e-7, 1 - 1e-7)
    y = np.asarray(labels)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), p


def train_model(model, train_ids, train_y, val_ids, val_y, max_len=256, epochs=2, bs=16, lr=2e-5,
                weight_decay=0.01, warmup=0.06, clip=1.0, seed=SEED, verbose=True):
    """AdamW + linear warmup/decay, bf16 autocast. Giữ trọng số của epoch có validation loss thấp nhất.

    Trả về history (list dict theo epoch) và info (thời gian train, bộ nhớ đỉnh, số bước bị bỏ).
    """
    set_seed(seed)
    base_mem = torch.cuda.memory_allocated() if DEVICE.type == "cuda" else 0   # bộ nhớ của model khác đang giữ trên GPU
    model.to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    steps = epochs * math.ceil(len(train_ids) / bs)
    sched = get_linear_schedule_with_warmup(opt, int(warmup * steps), steps)
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    hist, best, skipped, train_time = [], (float("inf"), None), 0, 0.0
    for ep in range(1, epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for _, x, m, yb in make_batches(train_ids, train_y, max_len, bs, shuffle=True, seed=seed + ep):
            x, m, yb = x.to(DEVICE), m.to(DEVICE), yb.to(DEVICE)
            with torch.autocast(DEVICE.type, dtype=AMP_DTYPE):
                loss = F.cross_entropy(model(x, m).float(), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, clip)
            if not (torch.isfinite(loss) and torch.isfinite(gn)):   # bước lỗi (NaN) → bỏ qua, như LAB 07
                skipped += 1
                sched.step()
                continue
            opt.step(); sched.step()
            tot += loss.item() * len(yb); n += len(yb)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        train_time += time.time() - t0
        vl, vp = evaluate_loss(model, val_ids, val_y, max_len)
        va = accuracy_score(val_y, vp > 0.5)
        hist.append({"epoch": ep, "train_loss": tot / max(n, 1), "val_loss": vl, "val_acc": va})
        if verbose:
            print(f"  epoch {ep}: train loss {tot / max(n, 1):.4f} | val loss {vl:.4f} | val acc {va:.4f} | {time.time() - t0:.0f}s")
        if vl < best[0]:
            best = (vl, {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}, ep)
    model.load_state_dict(best[1])
    info = {"train_time": train_time, "best_epoch": best[2], "skipped_steps": skipped,
            "peak_mem_mb": (torch.cuda.max_memory_allocated() - base_mem) / 2**20 if DEVICE.type == "cuda" else float("nan")}
    return hist, info


# ---------------------------------------------------------------- frozen mode
@torch.no_grad()
def extract_features(encoder, ids, max_len, bs=64):
    """Chạy encoder đóng băng một lần. Trả về (vector [CLS], mean của token thật), mỗi cái (N, H)."""
    encoder.to(DEVICE).eval()
    H = encoder.config.hidden_size
    cls, mean = torch.zeros(len(ids), H), torch.zeros(len(ids), H)
    for b, x, m, _ in make_batches(ids, None, max_len, bs):
        x, m = x.to(DEVICE), m.to(DEVICE)
        for _ in range(5):
            with torch.autocast(DEVICE.type, dtype=AMP_DTYPE):
                h = encoder(input_ids=x, attention_mask=m).last_hidden_state.float()
            if torch.isfinite(h).all():
                break
        mf = m.unsqueeze(-1).float()
        cls[b] = h[:, 0].cpu()
        mean[b] = ((h * mf).sum(1) / mf.sum(1)).cpu()
    return cls, mean


def train_head(feats_tr, y_tr, feats_va, y_va, n_classes=2, dropout=0.1, epochs=30, bs=64, lr=1e-3, seed=SEED):
    """Encoder đóng băng → chỉ train head (dropout + Linear) trên feature đã cache. Chọn epoch theo val loss."""
    set_seed(seed)
    head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feats_tr.size(1), n_classes)).to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    Xtr, Ytr = feats_tr.to(DEVICE), torch.tensor(np.asarray(y_tr)).to(DEVICE)
    Xva, Yva = feats_va.to(DEVICE), torch.tensor(np.asarray(y_va)).to(DEVICE)
    best, hist = (float("inf"), None), []
    g = torch.Generator(device="cpu").manual_seed(seed)
    for ep in range(1, epochs + 1):
        head.train()
        perm = torch.randperm(len(Xtr), generator=g).to(DEVICE)
        for s in range(0, len(Xtr), bs):
            i = perm[s:s + bs]
            loss = F.cross_entropy(head(Xtr[i]), Ytr[i])
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            vl = F.cross_entropy(head(Xva), Yva).item()
            va = (head(Xva).argmax(-1) == Yva).float().mean().item()
        hist.append({"epoch": ep, "val_loss": vl, "val_acc": va})
        if vl < best[0]:
            best = (vl, {k: v.clone() for k, v in head.state_dict().items()})
    head.load_state_dict(best[1])
    return head, hist


@torch.no_grad()
def head_proba(head, feats):
    head.eval()
    return torch.softmax(head(feats.to(DEVICE)), -1)[:, 1].cpu().numpy()


# ---------------------------------------------------------------- metrics
def metrics(y, p, thr=0.5):
    pred = (np.asarray(p) > thr).astype(int)
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred)}


if __name__ == "__main__":
    tok = get_tokenizer()
    enc = tok(["The cats are playing.", "unbelievable"], padding=True, truncation=True, return_tensors="pt")
    print(list(enc.keys()))
    for row in enc["input_ids"]:
        print(tok.convert_ids_to_tokens(row))
    m = PretrainedClassifier()
    print("pretrained params:", count_params(m))
    print("scratch params:", count_params(ScratchTransformer(tok.vocab_size)))
    with torch.no_grad():
        print("logits:", m(enc["input_ids"], enc["attention_mask"]))
