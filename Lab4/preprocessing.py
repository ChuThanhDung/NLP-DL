"""Tiện ích tiền xử lý và tải dữ liệu cho LAB 04 (IMDb sentiment classification)."""
import re
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42

_HTML_BR = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)
_NON_WORD = re.compile(r"[^a-z0-9'\s]")
_SPACES = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, bỏ thẻ <br />, bỏ ký tự đặc biệt (giữ dấu nháy đơn như don't)."""
    text = _HTML_BR.sub(" ", str(text).lower())
    text = _NON_WORD.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def tokenize(text: str):
    """Tách token đơn giản trên văn bản đã clean (dùng cho Word2Vec)."""
    return clean_text(text).split()


def load_imdb():
    """Tải IMDb từ Hugging Face. Trả về (train_df, test_df) với cột text, label."""
    from datasets import load_dataset

    ds = load_dataset("stanfordnlp/imdb")
    train_df = ds["train"].to_pandas()
    test_df = ds["test"].to_pandas()
    return train_df, test_df


def make_splits(train_df: pd.DataFrame, val_size: float = 0.2, seed: int = SEED):
    """Tách train gốc thành train/validation (stratified).

    Test giữ nguyên là tập test chính thức của IMDb; KHÔNG dùng để fit bất kỳ
    thành phần nào (TF-IDF, scaler, Word2Vec) hay để chọn siêu tham số.
    """
    tr, va = train_test_split(
        train_df, test_size=val_size, stratify=train_df["label"], random_state=seed
    )
    return tr.reset_index(drop=True), va.reset_index(drop=True)


def average_embedding(tokens, wv, dim: int) -> np.ndarray:
    """Average Word2Vec: trung bình vector các từ có trong vocab; rỗng -> vector 0."""
    vecs = [wv[t] for t in tokens if t in wv.key_to_index]
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vecs, axis=0)


def docs_to_matrix(token_lists, wv, dim: int) -> np.ndarray:
    return np.vstack([average_embedding(t, wv, dim) for t in token_lists])


class Timer:
    """Context manager đo thời gian (giây)."""

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.t0
