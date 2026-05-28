"""CPU-only sanity check for text cell.

ADR-016/017 메트릭·polluter가 가짜 corpus에서 동작하는지 + clean → polluted로
DSC 점수가 단조 감소하는지 확인. GPU/HuggingFace 의존 메트릭은
use_embeddings=False로 폴백.

실행:
    python _dev/sanity_text_cell.py
"""
from __future__ import annotations

import os
import sys

# dsc/ 루트를 PYTHONPATH에 추가 (dq4ai, dsc_framework 둘 다 접근)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from dsc_framework.text_cell import compute_dsc_text
from dsc_framework.text_cell_regression import compute_dsc_text_regression
from dsc_framework.text_polluters import (
    ClassBalanceTextPolluter,
    CompletenessTextPolluter,
    LabelSwapTextPolluter,
    NoiseInjectionTextPolluter,
    TargetDistributionSkewTextPolluter,
    TargetNoiseTextPolluter,
    WordShufflePolluter,
)


def make_corpus(n=100, seed=42):
    """가짜 토픽 분류 corpus. 4개 클래스, 각 클래스마다 distinctive 어휘."""
    rng = np.random.RandomState(seed)
    vocab = {
        0: ['economy', 'market', 'stock', 'bank', 'finance', 'currency', 'invest', 'trade'],
        1: ['sport', 'team', 'goal', 'player', 'match', 'season', 'league', 'champion'],
        2: ['movie', 'actor', 'director', 'scene', 'film', 'plot', 'character', 'cinema'],
        3: ['science', 'research', 'experiment', 'data', 'theory', 'hypothesis', 'method', 'result'],
    }
    common = ['the', 'and', 'of', 'in', 'a', 'is', 'to', 'with']
    texts, labels = [], []
    for i in range(n):
        c = i % 4
        length = rng.randint(15, 40)
        words = list(rng.choice(vocab[c], size=length // 2)) \
              + list(rng.choice(common, size=length - length // 2))
        rng.shuffle(words)
        texts.append(' '.join(words))
        labels.append(c)
    return texts, labels


def check_classification():
    print('=' * 70)
    print('TEXT × CLASSIFICATION CELL — sanity (use_embeddings=False)')
    print('=' * 70)

    texts, labels = make_corpus(n=120)
    base = compute_dsc_text(texts, labels, use_embeddings=False)
    print(f"clean DSC: {base['score']:.2f} ({base['grade']})")
    for k in ('completeness_text', 'uniqueness', 'validity', 'consistency',
              'outlier_ratio', 'class_balance', 'sample_quality_text'):
        print(f"  {k:25s} {base[k]:.4f}")

    polluters = [
        ('completeness_text(0.5)',     CompletenessTextPolluter(0.5)),
        ('noise_injection_text(0.3)',  NoiseInjectionTextPolluter(0.3)),
        ('word_shuffle(0.5)',          WordShufflePolluter(0.5)),
        ('class_balance(0.5)',         ClassBalanceTextPolluter(0.5)),
        ('label_swap(0.3)',            LabelSwapTextPolluter(0.3)),
    ]
    print()
    print('polluter             ΔDSC      relevant metric drop')
    print('-' * 70)
    for name, pol in polluters:
        try:
            t_p, l_p = pol.pollute(texts, labels)
        except Exception as e:
            print(f"{name:25s} ERROR: {e}")
            continue
        out = compute_dsc_text(t_p, l_p, use_embeddings=False)
        delta = out['score'] - base['score']
        # 관련 메트릭 1개 추출
        key_map = {
            'completeness_text(0.5)':    'completeness_text',
            'noise_injection_text(0.3)': 'sample_quality_text',
            'word_shuffle(0.5)':         'uniqueness',  # 셔플은 통계량엔 영향 적음
            'class_balance(0.5)':        'class_balance',
            'label_swap(0.3)':           'class_balance',  # label_swap은 본 메트릭이 아님
        }
        rk = key_map.get(name)
        rdrop = base[rk] - out[rk] if rk else 0.0
        print(f"{name:25s} {delta:+7.2f}    {rk}: {base[rk]:.3f} → {out[rk]:.3f} (Δ {rdrop:+.3f})")


def check_regression():
    print()
    print('=' * 70)
    print('TEXT × REGRESSION CELL — sanity (use_embeddings=False)')
    print('=' * 70)

    rng = np.random.RandomState(0)
    texts, _ = make_corpus(n=120)
    # 별점 1~5 균일 분포
    targets = rng.uniform(1.0, 5.0, size=len(texts)).tolist()

    base = compute_dsc_text_regression(texts, targets, use_embeddings=False)
    print(f"clean DSC: {base['score']:.2f} ({base['grade']})")
    for k in ('completeness_text', 'uniqueness', 'validity', 'consistency',
              'outlier_ratio', 'target_distribution_quality', 'sample_quality_text'):
        print(f"  {k:30s} {base[k]:.4f}")

    polluters = [
        ('target_distribution_skew(0.7)', TargetDistributionSkewTextPolluter(0.7)),
        ('target_noise(0.5)',             TargetNoiseTextPolluter(0.5)),
        ('completeness_text(0.5)',        CompletenessTextPolluter(0.5)),
    ]
    print()
    print('polluter                       ΔDSC      relevant metric drop')
    print('-' * 70)
    for name, pol in polluters:
        try:
            t_p, tg_p = pol.pollute(texts, targets)
        except Exception as e:
            print(f"{name:30s} ERROR: {e}")
            continue
        out = compute_dsc_text_regression(t_p, tg_p, use_embeddings=False)
        delta = out['score'] - base['score']
        key_map = {
            'target_distribution_skew(0.7)': 'target_distribution_quality',
            'target_noise(0.5)':             'target_distribution_quality',
            'completeness_text(0.5)':        'completeness_text',
        }
        rk = key_map.get(name)
        rdrop = base[rk] - out[rk] if rk else 0.0
        print(f"{name:30s} {delta:+7.2f}    {rk}: {base[rk]:.3f} → {out[rk]:.3f} (Δ {rdrop:+.3f})")


def check_router_detection():
    print()
    print('=' * 70)
    print('ROUTER / data_type_detection — sanity')
    print('=' * 70)

    from dsc_framework import compute_dsc, detect_data_type

    texts, labels = make_corpus(n=30)
    dt = detect_data_type(texts)
    print(f"detect_data_type(list[str])          → {dt!r}")
    assert dt == 'text', f"expected 'text', got {dt!r}"

    dt = detect_data_type((texts, labels))
    print(f"detect_data_type((texts, labels))    → {dt!r}")
    assert dt == 'text'

    res = compute_dsc(texts=texts, labels=labels, data_type='text', use_embeddings=False)
    print(f"compute_dsc(texts=..., labels=...)   → data_type={res['data_type']}, "
          f"task={res['task']}, score={res['score']}")
    assert res['data_type'] == 'text' and res['task'] == 'classification'

    res = compute_dsc(texts=texts, targets=[1.0] * 30, data_type='text', task='regression',
                      use_embeddings=False)
    print(f"compute_dsc(texts=..., targets=...)  → data_type={res['data_type']}, "
          f"task={res['task']}, score={res['score']}")
    assert res['data_type'] == 'text' and res['task'] == 'regression'
    print('all router/detection assertions passed.')


if __name__ == '__main__':
    check_classification()
    check_regression()
    check_router_detection()
    print()
    print('sanity ok.')
