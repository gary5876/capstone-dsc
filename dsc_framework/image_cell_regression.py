"""DSC v5 이미지 cell 회귀 트랙 — image × regression.

text_cell_regression(ADR-017)이 text_cell(ADR-016)을 미러링한 것과 동일한 패턴으로,
image_cell(ADR-014)을 회귀 트랙으로 미러링. 메트릭 7개 공유, 3개 재정의:
- class_balance              → target_distribution_quality (10-bin entropy)
- label_consistency          → target_smoothness            (k-NN 이웃 target std 보수)
- feature_informativeness    → feature_informativeness_reg  (mutual_info_regression)

fallback 가중치 합 1.00 (운영 가중치는 셀별 데이터 기반 선정으로 대체):
- completeness_image         : 0.15
- uniqueness                 : 0.10
- validity                   : 0.05
- consistency                : 0.05
- outlier_ratio              : 0.05
- target_distribution_quality: 0.10  (재정의)
- feature_correlation        : 0.05
- target_smoothness          : 0.20  (재정의)
- feature_informativeness_reg: 0.10  (재정의)
- sample_quality_image       : 0.15

embedding 추출(ResNet18)/공유 메트릭은 image_cell 모듈을 그대로 import.
"""
from __future__ import annotations

from math import log

import numpy as np

from .shared_metrics import to_grade

from .image_cell import (
    DEFAULT_WEIGHTS_IMAGE,
    _calc_feature_correlation_from_feats,
    _extract_features,
    calc_completeness_image,
    calc_consistency,
    calc_outlier_ratio,
    calc_sample_quality_image,
    calc_signal_integrity,
    calc_uniqueness,
    calc_validity,
)


# =================================================================
# 6'. target_distribution_quality — 10-bin entropy
# =================================================================

def calc_target_distribution_quality(targets, n_bins=10):
    """target 값을 equal-width n_bins binning → normalized Shannon entropy.

    균일 분포 = 1.0, 편향 = 0에 가까움. regression cell·dq4ai의
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
    """유사 이미지끼리 target 유사 = high smoothness. regression cell과 동일 공식."""
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


def calc_target_smoothness(images, targets, k=5, sample_cap=2000, random_state=1):
    if len(images) < k + 1:
        return 1.0
    feats, sample_idx = _extract_features(images, sample_cap=sample_cap, random_state=random_state)
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


def calc_feature_informativeness_reg(images, targets, sample_cap=2000, random_state=1, n_bins=10):
    if len(images) < 10:
        return 1.0
    feats, sample_idx = _extract_features(images, sample_cap=sample_cap, random_state=random_state)
    t = np.asarray(targets, dtype=float)[sample_idx]
    return _calc_feature_informativeness_reg_from_feats(feats, t, n_bins=n_bins,
                                                        random_state=random_state)


# =================================================================
# 가중치 + 통합 진입점
# =================================================================

DEFAULT_WEIGHTS_IMAGE_REG = {
    'completeness_image':           DEFAULT_WEIGHTS_IMAGE['completeness_image'],
    'uniqueness':                   DEFAULT_WEIGHTS_IMAGE['uniqueness'],
    'validity':                     DEFAULT_WEIGHTS_IMAGE['validity'],
    'consistency':                  DEFAULT_WEIGHTS_IMAGE['consistency'],
    'outlier_ratio':                DEFAULT_WEIGHTS_IMAGE['outlier_ratio'],
    'target_distribution_quality':  DEFAULT_WEIGHTS_IMAGE['class_balance'],
    'feature_correlation':          DEFAULT_WEIGHTS_IMAGE['feature_correlation'],
    'target_smoothness':            DEFAULT_WEIGHTS_IMAGE['label_consistency'],
    'feature_informativeness_reg':  DEFAULT_WEIGHTS_IMAGE['feature_informativeness'],
    'sample_quality_image':         DEFAULT_WEIGHTS_IMAGE['sample_quality_image'],
    'signal_integrity':             DEFAULT_WEIGHTS_IMAGE['signal_integrity'],
}


def compute_dsc_image_regression(images, targets, weights=None,
                                 use_embeddings=True,
                                 sample_cap=2000, random_state=1,
                                 target_n_bins=10,
                                 precomputed_feats=None):
    """DSC image regression cell 점수 (0~100) + 등급 + 지표별.

    Args:
        images: list of np.ndarray/PIL.Image/torch.Tensor
        targets: list/np.ndarray of float (연속/순서형 target, float 캐스팅)
        weights: dict (None → DEFAULT_WEIGHTS_IMAGE_REG fallback)
        use_embeddings: False면 ResNet 의존 3개 메트릭 1.0 처리 (debug용)
        target_n_bins: target_distribution_quality binning 수 (default 10)

    Pre-registered: image cell 회귀 트랙. 운영 가중치는 셀별 데이터 기반 선정 사용.
    """
    w = weights or DEFAULT_WEIGHTS_IMAGE_REG
    t_arr = np.asarray(targets, dtype=float)

    metrics = {
        'completeness_image':          calc_completeness_image(images, sample_cap=sample_cap, random_state=random_state),
        'uniqueness':                  calc_uniqueness(images, sample_cap=sample_cap, random_state=random_state),
        'validity':                    calc_validity(images, sample_cap=sample_cap, random_state=random_state),
        'consistency':                 calc_consistency(images, sample_cap=sample_cap, random_state=random_state),
        'outlier_ratio':               calc_outlier_ratio(images, sample_cap=sample_cap, random_state=random_state),
        'target_distribution_quality': calc_target_distribution_quality(t_arr, n_bins=target_n_bins),
        'sample_quality_image':        calc_sample_quality_image(images, sample_cap=sample_cap, random_state=random_state),
        'signal_integrity':            calc_signal_integrity(images, sample_cap=sample_cap, random_state=random_state),
    }
    if use_embeddings and len(images) >= 10:
        # ResNet18 feature를 1번만 추출, 3개 embedding 메트릭이 공유.
        # precomputed_feats=(feats, sample_idx) 주어지면 재사용 (probe와 임베딩 공유 → 중복 추출 제거).
        if precomputed_feats is not None:
            feats, sample_idx = precomputed_feats
        else:
            feats, sample_idx = _extract_features(images, sample_cap=sample_cap, random_state=random_state)
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
