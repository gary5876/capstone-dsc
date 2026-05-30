"""DSC v5 회귀 cell.

ADR-011 강한 버전 task-conditional framework의 regression instance.
- target_distribution_quality: class_balance 대체 (10-bin entropy)
- target_smoothness: label_consistency 대체 (k-NN target 거리 기반)
- feature_informativeness_reg: mutual_info_regression 사용
- 보조 지표 compute_dsc_degradation (ADR-012, baseline-relative)

공통 6지표는 shared_metrics에서 import (분류 cell과 동일 정의식 보장).
"""
from math import log

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .shared_metrics import (
    calc_completeness, calc_consistency, calc_feature_correlation,
    calc_outlier_ratio, calc_uniqueness, calc_validity, to_grade,
)


def calc_target_distribution_quality(df, target_col, n_bins=10,
                                     reference_df=None):
    """Normalized Shannon entropy of binned target distribution.

    Replaces class_balance for regression. 1.0 = uniform across n_bins,
    0.0 = fully concentrated. Uses reference_df target range for bin edges
    if provided, ensuring scores are comparable across pollution levels.
    """
    if target_col not in df.columns:
        return 1.0
    target = pd.to_numeric(df[target_col], errors='coerce').dropna().values
    if len(target) == 0:
        return 1.0

    if reference_df is not None and target_col in reference_df.columns:
        ref_target = pd.to_numeric(reference_df[target_col],
                                   errors='coerce').dropna()
        if len(ref_target) > 0:
            ref_min, ref_max = float(ref_target.min()), float(ref_target.max())
        else:
            ref_min, ref_max = float(target.min()), float(target.max())
    else:
        ref_min, ref_max = float(target.min()), float(target.max())

    if ref_max == ref_min:
        return 0.0

    bin_edges = np.linspace(ref_min, ref_max, n_bins + 1)
    bin_edges[-1] = bin_edges[-1] + 1e-9
    counts, _ = np.histogram(target, bins=bin_edges)
    total = counts.sum()
    if total == 0:
        return 0.0

    probs = counts / total
    nonzero = probs[probs > 0]
    entropy = float(-np.sum(nonzero * np.log(nonzero)))
    max_entropy = log(n_bins)
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def calc_target_smoothness(df, target_col, numerical_cols, k=5,
                           sample_cap=2000, random_state=1):
    """k-NN target smoothness — regression analogue of label_consistency.

    For each sample, computes the absolute deviation between its target
    and the mean target of its k nearest neighbors (in standardized feature
    space, with target also standardized). Lower deviation = smoother
    target surface = higher quality.

    Score = 1 / (1 + mean_normalized_deviation).

    Reproducible with random_state for sample_cap subsampling.
    """
    num_cols = [c for c in numerical_cols if c in df.columns]
    if not num_cols or target_col not in df.columns:
        return 1.0
    work = df[num_cols + [target_col]].dropna()
    work = work.drop_duplicates(subset=num_cols + [target_col]).reset_index(drop=True)
    if len(work) < k + 1:
        return 1.0
    if sample_cap and len(work) > sample_cap:
        work = work.sample(n=sample_cap, random_state=random_state).reset_index(drop=True)

    X = work[num_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
    y = pd.to_numeric(work[target_col], errors='coerce').fillna(0).values

    y_std = y.std()
    if y_std == 0:
        return 1.0
    y_norm = (y - y.mean()) / y_std

    X_std = StandardScaler().fit_transform(X)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_std)
    _, idx = nn.kneighbors(X_std)
    neighbor_targets = y_norm[idx[:, 1:]]
    neighbor_mean = neighbor_targets.mean(axis=1)
    deviations = np.abs(y_norm - neighbor_mean)
    mean_dev = float(deviations.mean())

    return float(1.0 / (1.0 + mean_dev))


def calc_feature_informativeness_reg(df, target_col, numerical_cols,
                                     categorical_cols, sample_cap=2000,
                                     random_state=1):
    """Mutual information between features and continuous target.

    Replaces feature_informativeness (mutual_info_classif) with
    mutual_info_regression. Sum of MI across features, normalized by
    target's binned entropy as proxy for upper bound.
    """
    if target_col not in df.columns:
        return 1.0
    num_cols = [c for c in numerical_cols if c in df.columns]
    cat_cols = [c for c in categorical_cols if c in df.columns]
    if not num_cols and not cat_cols:
        return 1.0
    keep_cols = num_cols + cat_cols + [target_col]
    work = df[keep_cols].dropna(subset=[target_col]).copy()
    work = work.drop_duplicates().reset_index(drop=True)
    if sample_cap and len(work) > sample_cap:
        work = work.sample(n=sample_cap, random_state=random_state).reset_index(drop=True)

    y = pd.to_numeric(work[target_col], errors='coerce').fillna(0).values

    pieces, discrete_mask = [], []
    for col in num_cols:
        s = pd.to_numeric(work[col], errors='coerce').fillna(0)
        pieces.append(s.values.reshape(-1, 1))
        discrete_mask.append(False)
    for col in cat_cols:
        s = LabelEncoder().fit_transform(work[col].astype(str).fillna('NA'))
        pieces.append(s.reshape(-1, 1))
        discrete_mask.append(True)
    X = np.hstack(pieces)

    try:
        mi = mutual_info_regression(X, y, discrete_features=discrete_mask,
                                    random_state=random_state)
    except Exception:
        return 1.0

    if y.max() == y.min():
        return 1.0
    bin_edges = np.linspace(y.min(), y.max() + 1e-9, 11)
    counts, _ = np.histogram(y, bins=bin_edges)
    total = counts.sum()
    if total == 0:
        return 1.0
    probs = counts / total
    nonzero = probs[probs > 0]
    h_y = float(-(nonzero * np.log(nonzero)).sum())
    if h_y <= 0:
        return 1.0

    return float(np.clip(mi.sum() / h_y, 0.0, 1.0))


DEFAULT_WEIGHTS_REGRESSION = {
    'completeness':                 0.20,
    'uniqueness':                   0.15,
    'validity':                     0.05,
    'consistency':                  0.10,
    'outlier_ratio':                0.05,
    'target_distribution_quality':  0.10,
    'feature_correlation':          0.05,
    'target_smoothness':            0.20,
    'feature_informativeness_reg':  0.10,
}


def compute_dsc_regression(df, target_col, numerical_cols, categorical_cols,
                           weights=None,
                           placeholder_numerical=-1,
                           placeholder_categorical='empty',
                           reference_df=None,
                           n_bins_target=10):
    """Compute DSC regression score and grade.

    Returns dict with score (0~100), grade (A/B/C/D), and per-metric scores.
    Pre-registered (ADR-011, 마스터플랜 sect 3-2).
    """
    w = weights or DEFAULT_WEIGHTS_REGRESSION
    metrics = {
        'completeness': calc_completeness(
            df, target_col, placeholder_numerical, placeholder_categorical),
        'uniqueness': calc_uniqueness(df, target_col),
        'validity': calc_validity(df, target_col, numerical_cols, categorical_cols),
        'consistency': calc_consistency(
            df, target_col, categorical_cols,
            reference_df=reference_df,
            placeholder_categorical=placeholder_categorical),
        'outlier_ratio': calc_outlier_ratio(
            df, target_col, numerical_cols, reference_df=reference_df),
        'target_distribution_quality': calc_target_distribution_quality(
            df, target_col, n_bins=n_bins_target, reference_df=reference_df),
        'feature_correlation': calc_feature_correlation(df, target_col, numerical_cols),
        'target_smoothness': calc_target_smoothness(df, target_col, numerical_cols),
        'feature_informativeness_reg': calc_feature_informativeness_reg(
            df, target_col, numerical_cols, categorical_cols),
    }
    score = sum(metrics[k] * w[k] for k in w) * 100
    rounded = {k: round(v, 4) for k, v in metrics.items()}
    return {'score': round(score, 2), 'grade': to_grade(score),
            **rounded, 'metrics': rounded}


def compute_dsc_degradation(polluted_dsc, clean_dsc, weights=None):
    """Baseline-relative degradation index (DSC v5 auxiliary metric).

    For each metric m: m_deg = max(0, 1 - polluted[m] / clean[m]).
    Overall = weighted sum of per-metric degradations.

    Returns:
        overall_degradation:    [0, 1], 0 = no loss, 1 = total loss (clipped)
        overall_degradation_signed: raw value, can be negative if metric improved
        preservation_score:     (1 - overall_degradation) * 100, comparable to DSC
        per-metric *_deg:       clipped per-metric degradations
        per-metric *_deg_signed: raw per-metric values

    The clipped form is the primary signal; signed form is included for
    diagnostic transparency (e.g., target_smoothness can spuriously rise
    when too few rows survive after dropna — reported as 0 in clipped form
    but visible in signed form).

    Both polluted_dsc and clean_dsc must be the dicts returned by
    compute_dsc_regression (containing per-metric scores).
    """
    w = weights or DEFAULT_WEIGHTS_REGRESSION
    per_metric_raw = {}
    per_metric_clipped = {}
    for m in w.keys():
        if m not in clean_dsc or m not in polluted_dsc:
            per_metric_raw[m] = 0.0
            per_metric_clipped[m] = 0.0
            continue
        clean_v = float(clean_dsc[m])
        polluted_v = float(polluted_dsc[m])
        if clean_v <= 0:
            per_metric_raw[m] = 0.0
            per_metric_clipped[m] = 0.0
        else:
            raw = 1.0 - (polluted_v / clean_v)
            per_metric_raw[m] = raw
            per_metric_clipped[m] = max(0.0, raw)

    overall_raw = sum(w[m] * per_metric_raw[m] for m in w.keys())
    overall_clipped = sum(w[m] * per_metric_clipped[m] for m in w.keys())
    preservation_score = (1.0 - overall_clipped) * 100

    return {
        'overall_degradation':        round(overall_clipped, 4),
        'overall_degradation_signed': round(overall_raw, 4),
        'preservation_score':         round(preservation_score, 2),
        **{f'{m}_deg':        round(v, 4) for m, v in per_metric_clipped.items()},
        **{f'{m}_deg_signed': round(v, 4) for m, v in per_metric_raw.items()},
    }
