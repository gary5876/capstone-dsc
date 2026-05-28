"""텍스트 cell 모델 학습 함수 — ADR-016 / ADR-017 §3-2 사전등록 5종.

분류:
- LogReg + TF-IDF (sanity baseline)
- TextCNN (random init embedding + 3-kernel conv)
- DistilBERT / BERT / RoBERTa (HuggingFace finetune)

회귀:
- Ridge + TF-IDF
- XGBoost + TF-IDF
- TextCNN-Reg (regression head)
- DistilBERT-Reg / BERT-Reg (HuggingFace finetune)

각 함수 시그니처: train_*(tr_texts, tr_y, te_texts, te_y, ...) -> metric (float).

분류 metric = accuracy / 회귀 metric = R² (음수 clip to 0).

별도 모듈로 분리한 이유: 노트북에 inline 정의 시 sanity 검증과 운영 코드가
*다른 곳에 있게 됨*. 단일 source of truth 위해 본 모듈로 통일하고 노트북은
import만.

heavy dependency (torch / transformers / xgboost)는 함수 내부 lazy import —
필요 없는 모델만 쓰는 경우 import 실패해도 무관.
"""
from __future__ import annotations

import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score


# =================================================================
# baseline (분류·회귀 공용)
# =================================================================

def train_logreg_tfidf(tr_t, tr_y, te_t, te_y,
                       max_features=20000, ngram_range=(1, 2),
                       max_iter=2000, random_state=42):
    """LogReg + TF-IDF (분류 sanity baseline). returns: accuracy."""
    vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    Xtr = vec.fit_transform(tr_t)
    Xte = vec.transform(te_t)
    clf = LogisticRegression(max_iter=max_iter, n_jobs=-1,
                             random_state=random_state).fit(Xtr, tr_y)
    return float(accuracy_score(te_y, clf.predict(Xte)))


def train_ridge_tfidf(tr_t, tr_y, te_t, te_y,
                      max_features=20000, ngram_range=(1, 2),
                      alpha=1.0, random_state=42):
    """Ridge + TF-IDF (회귀 sanity baseline). returns: R² (음수 clip 0)."""
    vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    Xtr = vec.fit_transform(tr_t)
    Xte = vec.transform(te_t)
    clf = Ridge(alpha=alpha, random_state=random_state).fit(Xtr, tr_y)
    return float(max(0.0, r2_score(te_y, clf.predict(Xte))))


def train_xgb_tfidf(tr_t, tr_y, te_t, te_y,
                    max_features=20000, ngram_range=(1, 2),
                    max_depth=6, n_estimators=500, learning_rate=0.05,
                    random_state=42):
    """XGBoost + TF-IDF (회귀). returns: R² (음수 clip 0)."""
    from xgboost import XGBRegressor
    vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    Xtr = vec.fit_transform(tr_t)
    Xte = vec.transform(te_t)
    clf = XGBRegressor(max_depth=max_depth, n_estimators=n_estimators,
                       learning_rate=learning_rate, random_state=random_state,
                       n_jobs=-1).fit(Xtr, tr_y)
    return float(max(0.0, r2_score(te_y, clf.predict(Xte))))


# =================================================================
# TextCNN — 분류 + 회귀 head 교체
# =================================================================

class _SimpleTokenizer:
    """whitespace tokenize + 단순 vocab. random init embedding용.

    PAD=0, UNK=1 고정. max_features 빈도 컷오프.
    """
    PAD, UNK = 0, 1

    def __init__(self, max_vocab=20000):
        self.max_vocab = max_vocab
        self.itos = ['[PAD]', '[UNK]']
        self.stoi = {'[PAD]': 0, '[UNK]': 1}

    def fit(self, texts):
        from collections import Counter
        cnt = Counter()
        for t in texts:
            cnt.update(t.split())
        for tok, _ in cnt.most_common(self.max_vocab - 2):
            self.stoi[tok] = len(self.itos)
            self.itos.append(tok)
        return self

    def encode(self, text, max_len):
        ids = [self.stoi.get(tok, self.UNK) for tok in text.split()[:max_len]]
        if len(ids) < max_len:
            ids = ids + [self.PAD] * (max_len - len(ids))
        return ids

    def __len__(self):
        return len(self.itos)


def _build_textcnn(vocab_size, n_class, emb=128, kernels=(3, 4, 5), filters=100,
                   dropout=0.5, regression=False):
    """ADR-016/017 §3-2 사전등록 사양 TextCNN."""
    import torch
    import torch.nn as nn

    class TextCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
            self.convs = nn.ModuleList([
                nn.Conv1d(emb, filters, k, padding=k // 2) for k in kernels])
            self.drop = nn.Dropout(dropout)
            self.fc = nn.Linear(filters * len(kernels), 1 if regression else n_class)
            self.regression = regression

        def forward(self, x):
            x = self.emb(x).transpose(1, 2)
            x = torch.cat([torch.max(torch.relu(c(x)), dim=2).values for c in self.convs], dim=1)
            x = self.drop(x)
            out = self.fc(x)
            return out.squeeze(-1) if self.regression else out

    return TextCNN()


def train_textcnn(tr_t, tr_y, te_t, te_y, regression=False,
                  max_len=256, batch=64, epochs=10, lr=1e-3,
                  emb=128, filters=100, device=None, random_state=42):
    """TextCNN finetune. 분류 → accuracy, 회귀 → R² (음수 clip 0)."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(random_state)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    tok = _SimpleTokenizer().fit(tr_t)

    def to_tensor(texts, ys):
        X = torch.tensor([tok.encode(t, max_len) for t in texts], dtype=torch.long)
        if regression:
            y = torch.tensor(list(ys), dtype=torch.float32)
        else:
            y = torch.tensor(list(ys), dtype=torch.long)
        return X, y

    Xtr, ytr = to_tensor(tr_t, tr_y)
    Xte, yte = to_tensor(te_t, te_y)

    n_class = 1 if regression else int(max(int(max(tr_y)), int(max(te_y))) + 1)
    model = _build_textcnn(len(tok), n_class, emb=emb, filters=filters,
                           regression=regression).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss() if regression else nn.CrossEntropyLoss()

    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch, shuffle=True)
    for _ep in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(Xte), batch):
            xb = Xte[i:i + batch].to(device)
            out = model(xb)
            if regression:
                preds.append(out.cpu().numpy())
            else:
                preds.append(out.argmax(dim=1).cpu().numpy())
        preds = np.concatenate(preds)

    if regression:
        return float(max(0.0, r2_score(yte.numpy(), preds)))
    return float(accuracy_score(yte.numpy(), preds))


# =================================================================
# Transformer — DistilBERT / BERT / RoBERTa 공용
# =================================================================

def train_transformer(model_id, tr_t, tr_y, te_t, te_y, regression=False,
                      max_len=256, batch=32, epochs=3, lr=2e-5,
                      weight_decay=0.01, output_dir=None, random_state=42):
    """HuggingFace AutoModelForSequenceClassification finetune (분류/회귀 공용)."""
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer,
        DataCollatorWithPadding, Trainer, TrainingArguments,
    )

    torch.manual_seed(random_state)
    n_label = 1 if regression else int(max(int(max(tr_y)), int(max(te_y))) + 1)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=n_label,
        problem_type='regression' if regression else 'single_label_classification')

    def to_hf(texts, ys):
        return Dataset.from_dict({
            'text': list(texts),
            'labels': [float(y) for y in ys] if regression else [int(y) for y in ys],
        })

    def _tokenize(batch):
        return tok(batch['text'], truncation=True, max_length=max_len)

    ds_tr = to_hf(tr_t, tr_y).map(_tokenize, batched=True, remove_columns=['text'])
    ds_te = to_hf(te_t, te_y).map(_tokenize, batched=True, remove_columns=['text'])

    args = TrainingArguments(
        output_dir=output_dir or f'./_tmp_tx_{os.getpid()}',
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        learning_rate=lr,
        weight_decay=weight_decay,
        logging_strategy='no',
        save_strategy='no',
        report_to='none',
        seed=random_state,
    )

    def _metric(eval_pred):
        preds, labels = eval_pred
        if regression:
            preds = preds.squeeze(-1) if preds.ndim > 1 else preds
            return {'r2': float(max(0.0, r2_score(labels, preds)))}
        return {'accuracy': float(accuracy_score(labels, preds.argmax(axis=-1)))}

    # transformers 4.46+에서 `tokenizer` → `processing_class`로 인자명 변경.
    # 두 버전 호환 위해 시그니처 확인 후 적절히 분기.
    import inspect
    trainer_kwargs = dict(
        model=model, args=args,
        train_dataset=ds_tr, eval_dataset=ds_te,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=_metric,
    )
    trainer_sig = inspect.signature(Trainer.__init__).parameters
    if 'processing_class' in trainer_sig:
        trainer_kwargs['processing_class'] = tok
    else:
        trainer_kwargs['tokenizer'] = tok
    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    metrics = trainer.evaluate()
    return float(metrics.get('eval_r2' if regression else 'eval_accuracy', float('nan')))


# =================================================================
# 사전등록 모델 라인업 — ADR-016/017 §3-2
# =================================================================

CLASSIFICATION_MODELS = {
    'logreg_tfidf':   lambda tr_t, tr_y, te_t, te_y, **kw: train_logreg_tfidf(tr_t, tr_y, te_t, te_y, **kw),
    'textcnn':        lambda tr_t, tr_y, te_t, te_y, **kw: train_textcnn(tr_t, tr_y, te_t, te_y, regression=False, **kw),
    'distilbert':     lambda tr_t, tr_y, te_t, te_y, **kw: train_transformer('distilbert-base-uncased', tr_t, tr_y, te_t, te_y, regression=False, **kw),
    'bert_base':      lambda tr_t, tr_y, te_t, te_y, **kw: train_transformer('bert-base-uncased', tr_t, tr_y, te_t, te_y, regression=False, **kw),
    'roberta_base':   lambda tr_t, tr_y, te_t, te_y, **kw: train_transformer('roberta-base', tr_t, tr_y, te_t, te_y, regression=False, **kw),
}

REGRESSION_MODELS = {
    'ridge_tfidf':    lambda tr_t, tr_y, te_t, te_y, **kw: train_ridge_tfidf(tr_t, tr_y, te_t, te_y, **kw),
    'xgb_tfidf':      lambda tr_t, tr_y, te_t, te_y, **kw: train_xgb_tfidf(tr_t, tr_y, te_t, te_y, **kw),
    'textcnn_reg':    lambda tr_t, tr_y, te_t, te_y, **kw: train_textcnn(tr_t, tr_y, te_t, te_y, regression=True, **kw),
    'distilbert_reg': lambda tr_t, tr_y, te_t, te_y, **kw: train_transformer('distilbert-base-uncased', tr_t, tr_y, te_t, te_y, regression=True, **kw),
    'bert_base_reg':  lambda tr_t, tr_y, te_t, te_y, **kw: train_transformer('bert-base-uncased', tr_t, tr_y, te_t, te_y, regression=True, **kw),
}
