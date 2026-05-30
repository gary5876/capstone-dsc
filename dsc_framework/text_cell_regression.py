"""DSC v5 텍스트 cell 회귀 트랙 — text × regression.

ADR-017 사전등록. ADR-016 분류 cell과 메트릭 7개 공유, 3개 재정의:
- class_balance              → target_distribution_quality (10-bin entropy)
- label_consistency          → target_smoothness            (k-NN 이웃 target std 보수)
- feature_informativeness    → feature_informativeness_reg  (mutual_info_regression)

fallback 가중치 합 1.00:
- completeness_text          : 0.15
- uniqueness                 : 0.10
- validity                   : 0.05
- consistency                : 0.05
- outlier_ratio              : 0.05
- target_distribution_quality: 0.10  (재정의)
- feature_correlation        : 0.05
- target_smoothness          : 0.20  (재정의)
- feature_informativeness_reg: 0.10  (재정의)
- sample_quality_text        : 0.15

embedding 추출/공유 메트릭은 text_cell 모듈을 그대로 import.
"""
from __future__ import annotations

from collections import Counter
from math import log

import numpy as np

from .shared_metrics import to_grade

from .text_cell import (
    DEFAULT_WEIGHTS_TEXT,
    _calc_feature_correlation_from_feats,
    _ensure_label_array,
    _ensure_text_list,
    _extract_features,
    calc_completeness_text,
    calc_consistency,
    calc_outlier_ratio,
    calc_sample_quality_text,
    calc_uniqueness,
    calc_validity,
)


# =================================================================
# 6'. target_distribution_quality — 10-bin entropy
# =================================================================

def calc_target_distribution_quality(targets, n_bins=10):
    """target 값을 equal-width n_bins binning → normalized Shannon entropy.

    균일 분포 = 1.0, 편향 = 0에 가까움. tabular regression cell·dq4ai의
    TargetDistributionSkewPolluter.compute_quality_measure와 동일 공식.
    """
    arr = np.asarray(targets, dtype=float)
    if len(arr) == 0:
        return 0.0
    ref_min, ref_max = float(arr.min()), float(arr.max())
    if ref_max == ref_min:
        return 0.0
    bin_edges = np.linspace(ref_min, ref_max, n_bins + 1)
    bin_edges[-1] = bin_edges[-1] + 1e-9
    counts, _ = np.histogram(arr, bins=bin_edges)
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    nonzero = probs[probs > 0]
    if len(nonzero) <= 1:
        return 0.0
    ent = float(-(nonzero * np.log(nonzero)).sum())
    max_ent = log(n_bins)
    return float(ent / max_ent) if max_ent > 0 else 0.0


# =================================================================
# 8'. target_smoothness — k-NN embedding 이웃 target std 보수
# =================================================================

def _calc_target_smoothness_from_feats(feats, targets, k=5):
    """유사 텍스트끼리 target 유사 = high smoothness. tabular regression cell과 동일 공식."""
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    targets = np.asarray(targets, dtype=float)
    if len(feats) < k + 1:
        return 1.0
    feats_std = StandardScaler().fit_transform(feats)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(feats_std)
    _, idx = nn.kneighbors(feats_std)
    neighbor_targets = targets[idx[:, 1:]]
    local_std = neighbor_targets.std(axis=1).mean()
    target_std = targets.std()
    if target_std == 0:
        return 1.0
    return float(1.0 - np.clip(local_std / target_std, 0.0, 1.0))


def calc_target_smoothness(texts, targets, k=5, sample_cap=1000, random_state=1):
    if len(texts) < k + 1:
        return 1.0
    feats, sample_idx = _extract_features(texts, sample_cap=sample_cap, random_state=random_state)
    t = np.asarray(targets, dtype=float)[sample_idx]
    return _calc_target_smoothness_from_feats(feats, t, k=k)


# =================================================================
# 9'. feature_informativeness_reg — mutual_info_regression / log(n_bins)
# =================================================================

def _calc_feature_informativeness_reg_from_feats(feats, targets, n_bins=10, random_state=1):
    from sklearn.feature_selection import mutual_info_regression

    try:
        mi = mutual_info_regression(feats, targets, discrete_features=False,
                                    random_state=random_state)
    except Exception:
        return 1.0
    h_target = log(n_bins) if n_bins > 1 else 1.0
    return float(np.clip(mi.sum() / h_target, 0.0, 1.0))


def calc_feature_informativeness_reg(texts, targets, sample_cap=1000, random_state=1, n_bins=10):
    if len(texts) < 10:
        return 1.0
    feats, sample_idx = _extract_features(texts, sample_cap=sample_cap, random_state=random_state)
    t = np.asarray(targets, dtype=float)[sample_idx]
    return _calc_feature_informativeness_reg_from_feats(feats, t, n_bins=n_bins,
                                                        random_state=random_state)


# =================================================================
# 가중치 + 통합 진입점
# =================================================================

DEFAULT_WEIGHTS_TEXT_REG = {
    'completeness_text':            DEFAULT_WEIGHTS_TEXT['completeness_text'],
    'uniqueness':                   DEFAULT_WEIGHTS_TEXT['uniqueness'],
    'validity':                     DEFAULT_WEIGHTS_TEXT['validity'],
    'consistency':                  DEFAULT_WEIGHTS_TEXT['consistency'],
    'outlier_ratio':                DEFAULT_WEIGHTS_TEXT['outlier_ratio'],
    'target_distribution_quality':  DEFAULT_WEIGHTS_TEXT['class_balance'],
    'feature_correlation':          DEFAULT_WEIGHTS_TEXT['feature_correlation'],
    'target_smoothness':            DEFAULT_WEIGHTS_TEXT['label_consistency'],
    'feature_informativeness_reg':  DEFAULT_WEIGHTS_TEXT['feature_informativeness'],
    'sample_quality_text':          DEFAULT_WEIGHTS_TEXT['sample_quality_text'],
}


def compute_dsc_text_regression(texts, targets, weights=None,
                                use_embeddings=True,
                                sample_cap=1000, random_state=1,
                                target_n_bins=10):
    """DSC text regression cell 점수 (0~100) + 등급 + 지표별.

    Args:
        texts: list[str] / pd.Series / np.ndarray
        targets: list/np.ndarray of float (ordinal도 float 캐스팅)
        weights: dict (None → DEFAULT_WEIGHTS_TEXT_REG, ADR-015 fallback)
        use_embeddings: False면 embedding 의존 3개 메트릭 1.0 처리
        target_n_bins: target_distribution_quality binning 수 (ADR-017 default 10)

    Pre-registered: ADR-017. 운영 시 weights는 LLM weight generator 출력 사용.
    """
    w = weights or DEFAULT_WEIGHTS_TEXT_REG
    texts = _ensure_text_list(texts)
    t_arr = np.asarray(targets, dtype=float)

    metrics = {
        'completeness_text':           calc_completeness_text(texts, sample_cap=sample_cap, random_state=random_state),
        'uniqueness':                  calc_uniqueness(texts, sample_cap=sample_cap, random_state=random_state),
        'validity':                    calc_validity(texts, sample_cap=sample_cap, random_state=random_state),
        'consistency':                 calc_consistency(texts, sample_cap=sample_cap, random_state=random_state),
        'outlier_ratio':               calc_outlier_ratio(texts, sample_cap=sample_cap, random_state=random_state),
        'target_distribution_quality': calc_target_distribution_quality(t_arr, n_bins=target_n_bins),
        'sample_quality_text':         calc_sample_quality_text(texts, sample_cap=sample_cap, random_state=random_state),
    }
    if use_embeddings and len(texts) >= 10:
        feats, sample_idx = _extract_features(texts, sample_cap=sample_cap, random_state=random_state)
        t_sample = t_arr[sample_idx]
        metrics['feature_correlation'] = _calc_feature_correlation_from_feats(feats)
        metrics['target_smoothness'] = (
            _calc_target_smoothness_from_feats(feats, t_sample, k=5)
            if len(feats) >= 6 else 1.0
        )
        metrics['feature_informativeness_reg'] = _calc_feature_informativeness_reg_from_feats(
            feats, t_sample, n_bins=target_n_bins, random_state=random_state)
    else:
        metrics['feature_correlation'] = 1.0
        metrics['target_smoothness'] = 1.0
        metrics['feature_informativeness_reg'] = 1.0

    score = sum(metrics[k] * w[k] for k in w) * 100
    rounded = {k: round(v, 4) for k, v in metrics.items()}
    return {'score': round(score, 2), 'grade': to_grade(score),
            **rounded, 'metrics': rounded}
