"""두 cell이 공유하는 6개 데이터 품질 메트릭.

분류 cell의 사전등록 정의식(v4 마스터플랜, r=0.598)을 단일 출처로 보존.
회귀 cell도 이 모듈을 import하여 동일한 정의식을 사용한다.
"""
import re

import numpy as np
import pandas as pd


def calc_completeness(df, target_col, placeholder_numerical=-1,
                      placeholder_categorical='empty'):
    """결측치 + placeholder 비율. 1=완전, 0=전부 결측."""
    feature_df = df.drop(columns=[target_col], errors='ignore')
    total_cells = feature_df.shape[0] * feature_df.shape[1]
    if total_cells == 0:
        return 1.0
    missing_count = feature_df.isnull().sum().sum()
    if placeholder_numerical is not None:
        for col in feature_df.select_dtypes(include=[np.number]).columns:
            missing_count += (feature_df[col] == placeholder_numerical).sum()
    if placeholder_categorical is not None:
        for col in feature_df.select_dtypes(include=['object', 'category']).columns:
            ph = (placeholder_categorical.get(col, 'empty')
                  if isinstance(placeholder_categorical, dict)
                  else placeholder_categorical)
            missing_count += (feature_df[col].astype(str) == str(ph)).sum()
    return 1.0 - (missing_count / total_cells)


def calc_uniqueness(df, target_col):
    """중복 행 비율. 1=전부 유일."""
    n = len(df)
    if n <= 1:
        return 1.0
    return 1.0 - (df.duplicated().sum() / n)


def calc_validity(df, target_col, numerical_cols, categorical_cols):
    """타입/형식 유효성."""
    scores = []
    for col in numerical_cols:
        if col not in df.columns:
            continue
        converted = pd.to_numeric(df[col], errors='coerce')
        total = len(df[col].dropna())
        scores.append(converted.notna().sum() / total if total > 0 else 1.0)
    for col in categorical_cols:
        if col not in df.columns:
            continue
        s = df[col].dropna().astype(str)
        if len(s) == 0:
            scores.append(1.0)
            continue
        valid = s.apply(lambda x: 0 < len(x.strip()) < 200)
        scores.append(valid.mean())
    return float(np.mean(scores)) if scores else 1.0


def calc_consistency(df, target_col, categorical_cols, reference_df=None,
                     placeholder_categorical='empty'):
    """카테고리 표현 일관성 — reference 있으면 새 표현 비율의 보수, placeholder 제외."""
    if not categorical_cols:
        return 1.0
    scores = []
    for col in categorical_cols:
        if col not in df.columns:
            continue
        cur_vals = df[col].dropna().astype(str)
        if len(cur_vals) == 0:
            scores.append(1.0); continue
        ph = None
        if placeholder_categorical is not None:
            ph = (str(placeholder_categorical.get(col, 'empty'))
                  if isinstance(placeholder_categorical, dict)
                  else str(placeholder_categorical))
            cur_vals = cur_vals[cur_vals != ph]
        if len(cur_vals) == 0:
            scores.append(1.0); continue
        if reference_df is not None and col in reference_df.columns:
            ref_vals = reference_df[col].dropna().astype(str)
            if ph is not None:
                ref_vals = ref_vals[ref_vals != ph]
            ref_set = set(ref_vals.unique())
            new_row_ratio = (~cur_vals.isin(ref_set)).mean()
            scores.append(1.0 - float(new_row_ratio))
        else:
            has_suffix = cur_vals.apply(lambda x: bool(re.search(r'-\d+$', x)))
            scores.append(1.0 - float(has_suffix.mean()))
    return float(np.mean(scores)) if scores else 1.0


def calc_outlier_ratio(df, target_col, numerical_cols, reference_df=None):
    """IQR 기반 outlier가 아닌 비율. reference_df의 IQR을 고정 기준으로 사용."""
    if not numerical_cols:
        return 1.0
    scores = []
    for col in numerical_cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(s) < 4:
            scores.append(1.0); continue
        if reference_df is not None and col in reference_df.columns:
            ref = pd.to_numeric(reference_df[col], errors='coerce').dropna()
            if len(ref) >= 4:
                q1, q3 = ref.quantile(0.25), ref.quantile(0.75)
            else:
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
        else:
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            scores.append(1.0); continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_count = ((s < lower) | (s > upper)).sum()
        scores.append(1.0 - outlier_count / len(s))
    return float(np.mean(scores)) if scores else 1.0


def to_grade(score):
    """0~100 점수 → 등급. 사전등록 임계(90/75/60), 모든 cell 공유 단일 출처."""
    if score >= 90:
        return 'A'
    if score >= 75:
        return 'B'
    if score >= 60:
        return 'C'
    return 'D'


def calc_feature_correlation(df, target_col, numerical_cols, threshold=0.95):
    """고상관(>threshold) 피처 쌍이 없는 비율."""
    cols = [c for c in numerical_cols if c in df.columns]
    if len(cols) < 2:
        return 1.0
    num_df = df[cols].apply(pd.to_numeric, errors='coerce')
    corr = num_df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    total_pairs = upper.size - upper.isna().sum().sum()
    if total_pairs == 0:
        return 1.0
    high_corr_pairs = (upper > threshold).sum().sum()
    return 1.0 - (high_corr_pairs / total_pairs)
