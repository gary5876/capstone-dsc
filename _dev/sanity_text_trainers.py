"""dsc_framework.text_trainers 함수 sanity check.

작은 가짜 corpus로 분류/회귀 baseline + TextCNN + Transformer (DistilBERT)
실제 호출 → 반환 metric float인지 확인. transformers는 CPU로 forward 1 epoch
정도라 분당 단위. 한 번에 다 돌리진 않고 critical 한 path만.
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from dsc_framework.text_trainers import (
    train_logreg_tfidf, train_ridge_tfidf, train_textcnn,
    train_transformer, train_xgb_tfidf,
)


def make_classification_corpus(n=80, seed=42):
    rng = np.random.RandomState(seed)
    vocab = {
        0: ['economy', 'market', 'stock', 'bank', 'finance', 'currency', 'invest'],
        1: ['sport', 'team', 'goal', 'player', 'match', 'season', 'league'],
        2: ['movie', 'actor', 'director', 'scene', 'film', 'plot', 'cinema'],
        3: ['science', 'research', 'experiment', 'theory', 'method', 'data'],
    }
    common = ['the', 'and', 'of', 'in', 'a', 'is', 'to', 'with']
    texts, labels = [], []
    for i in range(n):
        c = i % 4
        n_topic = rng.randint(8, 15)
        n_common = rng.randint(5, 10)
        words = list(rng.choice(vocab[c], size=n_topic)) + list(rng.choice(common, size=n_common))
        rng.shuffle(words)
        texts.append(' '.join(words))
        labels.append(c)
    return texts, labels


def split(texts, labels, ratio=0.7, seed=42):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(texts))
    cut = int(len(texts) * ratio)
    tr_idx, te_idx = idx[:cut], idx[cut:]
    return ([texts[i] for i in tr_idx], [labels[i] for i in tr_idx],
            [texts[i] for i in te_idx], [labels[i] for i in te_idx])


def main():
    print('=' * 70)
    print('TEXT TRAINERS SANITY (small corpus, CPU)')
    print('=' * 70)

    texts, labels = make_classification_corpus(n=80)
    tr_t, tr_y, te_t, te_y = split(texts, labels)
    print(f'corpus n={len(texts)}, train={len(tr_t)}, test={len(te_t)}')

    t0 = time.time()
    acc = train_logreg_tfidf(tr_t, tr_y, te_t, te_y)
    print(f'[{time.time()-t0:5.1f}s] logreg_tfidf  accuracy={acc:.3f}')
    assert 0.0 <= acc <= 1.0

    t0 = time.time()
    acc = train_textcnn(tr_t, tr_y, te_t, te_y, regression=False,
                        epochs=3, batch=16, max_len=64)
    print(f'[{time.time()-t0:5.1f}s] textcnn       accuracy={acc:.3f}')
    assert 0.0 <= acc <= 1.0

    # 회귀 — label을 float으로
    tr_yf = [float(y) for y in tr_y]
    te_yf = [float(y) for y in te_y]

    t0 = time.time()
    r2 = train_ridge_tfidf(tr_t, tr_yf, te_t, te_yf)
    print(f'[{time.time()-t0:5.1f}s] ridge_tfidf   R²={r2:.3f}')
    assert 0.0 <= r2 <= 1.0

    t0 = time.time()
    r2 = train_xgb_tfidf(tr_t, tr_yf, te_t, te_yf, n_estimators=50)
    print(f'[{time.time()-t0:5.1f}s] xgb_tfidf     R²={r2:.3f}')
    assert 0.0 <= r2 <= 1.0

    t0 = time.time()
    r2 = train_textcnn(tr_t, tr_yf, te_t, te_yf, regression=True,
                       epochs=3, batch=16, max_len=64)
    print(f'[{time.time()-t0:5.1f}s] textcnn_reg   R²={r2:.3f}')
    assert 0.0 <= r2 <= 1.0

    # Transformer는 CPU에서 시간 큼 — 1 epoch, max_len 32, batch 4로 forward 위주 검증
    t0 = time.time()
    acc = train_transformer('distilbert-base-uncased', tr_t, tr_y, te_t, te_y,
                            regression=False, epochs=1, batch=4, max_len=32)
    print(f'[{time.time()-t0:5.1f}s] distilbert    accuracy={acc:.3f}')
    assert 0.0 <= acc <= 1.0

    t0 = time.time()
    r2 = train_transformer('distilbert-base-uncased', tr_t, tr_yf, te_t, te_yf,
                           regression=True, epochs=1, batch=4, max_len=32)
    print(f'[{time.time()-t0:5.1f}s] distilbert_reg R²={r2:.3f}')
    assert 0.0 <= r2 <= 1.0

    print('\nsanity ok.')


if __name__ == '__main__':
    main()
