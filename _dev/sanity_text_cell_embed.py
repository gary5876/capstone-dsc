"""Embedding-active CPU sanity for text cell — DistilBERT mean-pool 경로 검증.

`sanity_text_cell.py`(use_embeddings=False)에서 못 검증한 3개 메트릭:
- feature_correlation
- label_consistency
- feature_informativeness

및 회귀 트랙의 target_smoothness, feature_informativeness_reg.

작은 corpus(n=40, sample_cap=20)로 CPU 5분 이내 실행 목표.
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

from dsc_framework.text_cell import compute_dsc_text
from dsc_framework.text_cell_regression import compute_dsc_text_regression


def make_corpus(n=40, seed=42):
    rng = np.random.RandomState(seed)
    vocab = {
        0: ['economy', 'market', 'stock', 'bank', 'finance'],
        1: ['sport', 'team', 'goal', 'player', 'match'],
        2: ['movie', 'actor', 'director', 'scene', 'film'],
        3: ['science', 'research', 'experiment', 'theory', 'method'],
    }
    common = ['the', 'and', 'of', 'in', 'a', 'is', 'to', 'with']
    texts, labels = [], []
    for i in range(n):
        c = i % 4
        length = rng.randint(15, 30)
        words = list(rng.choice(vocab[c], size=length // 2)) \
              + list(rng.choice(common, size=length - length // 2))
        rng.shuffle(words)
        texts.append(' '.join(words))
        labels.append(c)
    return texts, labels


def main():
    print('=' * 70)
    print('EMBEDDING-ACTIVE SANITY (DistilBERT mean-pool, CPU, sample_cap=20)')
    print('=' * 70)
    t0 = time.time()

    texts, labels = make_corpus(n=40)
    print(f'corpus: n={len(texts)} texts × {len(set(labels))} classes')

    r = compute_dsc_text(texts, labels, use_embeddings=True,
                         sample_cap=20, random_state=42)
    print(f'\nclassification clean (embeddings=True): DSC={r["score"]:.2f} ({r["grade"]})')
    print(f'  feature_correlation     {r["feature_correlation"]:.4f}')
    print(f'  label_consistency       {r["label_consistency"]:.4f}')
    print(f'  feature_informativeness {r["feature_informativeness"]:.4f}')
    assert r['label_consistency'] > 0.1, (
        f'label_consistency too low ({r["label_consistency"]}) — '
        'DistilBERT embedding이 4-class 토픽을 분리하지 못함. 통합 실패.')
    assert r['feature_informativeness'] > 0.05, 'MI 낮음, embedding-label 신호 부족'

    # 회귀 트랙 — target = label을 float화
    targets = [float(l) for l in labels]
    r2 = compute_dsc_text_regression(texts, targets, use_embeddings=True,
                                     sample_cap=20, random_state=42)
    print(f'\nregression clean (embeddings=True): DSC={r2["score"]:.2f} ({r2["grade"]})')
    print(f'  feature_correlation         {r2["feature_correlation"]:.4f}')
    print(f'  target_smoothness           {r2["target_smoothness"]:.4f}')
    print(f'  feature_informativeness_reg {r2["feature_informativeness_reg"]:.4f}')
    assert r2['target_smoothness'] > 0.0, 'target_smoothness 0이면 임베딩 공간 무효'

    # label_swap 50% → label_consistency 큰 폭 하락 기대
    from dsc_framework.text_polluters import LabelSwapTextPolluter
    pol = LabelSwapTextPolluter(0.5)
    t_p, l_p = pol.pollute(texts, labels)
    r_lab = compute_dsc_text(t_p, l_p, use_embeddings=True,
                              sample_cap=20, random_state=42)
    drop = r['label_consistency'] - r_lab['label_consistency']
    print(f'\nlabel_swap(0.5):')
    print(f'  label_consistency       {r["label_consistency"]:.4f} → {r_lab["label_consistency"]:.4f} (Δ {drop:+.4f})')
    assert drop > 0.0, 'label_swap이 label_consistency를 떨어뜨리지 못함 — 정의식 오작동'

    elapsed = time.time() - t0
    print(f'\nelapsed: {elapsed:.1f}s — sanity ok.')


if __name__ == '__main__':
    main()
