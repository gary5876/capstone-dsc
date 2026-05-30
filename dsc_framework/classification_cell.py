"""DSC v5 분류 cell.

v4 마스터플랜의 사전등록 정의식·가중치를 그대로 사용 (r=0.598 보존).
- ADR-009: value_accuracy 제거, label_consistency·feature_informativeness 신설
- ADR-011: task-conditional framework 강한 버전, classification cell instance
"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .shared_metrics import (
    calc_completeness, calc_consistency, calc_feature_correlation,
    calc_outlier_ratio, calc_uniqueness, calc_validity, to_grade,
)


def calc_class_balance(df, target_col):
    """클래스 균형 — 최소 비율 / 이상 비율."""
    counts = df[target_col].value_counts()
    n_classes = len(counts)
    if n_classes <= 1:
        return 1.0
    min_ratio = counts.min() / counts.sum()
    ideal_ratio = 1.0 / n_classes
    return min(min_ratio / ideal_ratio, 1.0)


def calc_label_consistency(df, target_col, numerical_cols, k=5,
                           sample_cap=2000, random_state=1):
    """k-NN 라벨 일관성 — chance level 보정.
    각 샘플의 k 최근접 이웃 라벨 중 자기 라벨과 같은 비율 → chance 보정.
    duplicate 행은 사전 제거 (UniquenessPolluter 복제 영향 무력화).
    수치형 컬럼만 사용. 수치형 없으면 1.0 (지표 비활성)."""
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
    y = work[target_col].astype(str).values
    X_std = StandardScaler().fit_transform(X)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_std)
    _, idx = nn.kneighbors(X_std)
    raw = (y[idx[:, 1:]] == y[:, None]).mean()
    class_props = pd.Series(y).value_counts(normalize=True).values
    chance = float((class_props ** 2).sum())
    if chance >= 1.0:
        return 1.0
    return float(np.clip((raw - chance) / (1.0 - chance), 0.0, 1.0))


def calc_feature_informativeness(df, target_col, numerical_cols, categorical_cols,
                                 sample_cap=2000, random_state=1):
    """피처-라벨 mutual information 합 / H(Y). 0~1 범위.
    duplicate 사전 제거. 수치형/범주형 모두 사용."""
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
    y = LabelEncoder().fit_transform(work[target_col].astype(str))
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
        mi = mutual_info_classif(X, y, discrete_features=discrete_mask,
                                 random_state=random_state)
    except Exception:
        return 1.0
    class_props = np.bincount(y) / len(y)
    class_props = class_props[class_props > 0]
    h_y = float(-(class_props * np.log(class_props)).sum())
    if h_y <= 0:
        return 1.0
    return float(np.clip(mi.sum() / h_y, 0.0, 1.0))


DEFAULT_WEIGHTS_CLASSIFICATION = {
    'completeness':           0.20,
    'uniqueness':             0.15,
    'validity':               0.05,
    'consistency':            0.10,
    'outlier_ratio':          0.05,
    'class_balance':          0.10,
    'feature_correlation':    0.05,
    'label_consistency':      0.20,
    'feature_informativeness': 0.10,
}


def compute_dsc_classification(df, target_col, numerical_cols, categorical_cols,
                                weights=None,
                                placeholder_numerical=-1,
                                placeholder_categorical='empty',
                                reference_df=None):
    """DSC 분류 cell 점수(0~100) + 등급 + 지표별 점수.

    Pre-registered (ADR-009, ADR-011, v4 마스터플랜). r=0.598 (Pearson).
    """
    w = weights or DEFAULT_WEIGHTS_CLASSIFICATION
    metrics = {
        'completeness':           calc_completeness(df, target_col, placeholder_numerical, placeholder_categorical),
        'uniqueness':             calc_uniqueness(df, target_col),
        'validity':               calc_validity(df, target_col, numerical_cols, categorical_cols),
        'consistency':            calc_consistency(df, target_col, categorical_cols, reference_df=reference_df, placeholder_categorical=placeholder_categorical),
        'outlier_ratio':          calc_outlier_ratio(df, target_col, numerical_cols, reference_df=reference_df),
        'class_balance':          calc_class_balance(df, target_col),
        'feature_correlation':    calc_feature_correlation(df, target_col, numerical_cols),
        'label_consistency':      calc_label_consistency(df, target_col, numerical_cols),
        'feature_informativeness': calc_feature_informativeness(df, target_col, numerical_cols, categorical_cols),
    }
    score = sum(metrics[k] * w[k] for k in w) * 100
    rounded = {k: round(v, 4) for k, v in metrics.items()}
    return {'score': round(score, 2), 'grade': to_grade(score),
            **rounded, 'metrics': rounded}
