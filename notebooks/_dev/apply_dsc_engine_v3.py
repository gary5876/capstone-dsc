"""DSC 엔진 v3 적용 — P1(outlier 역반응) + P2(value 미감지) 해결.

v3 변경점:
- calc_outlier_ratio에 reference_df 인자 추가 (베이스라인 IQR 고정 → 노이즈 자기참조 회피)
- calc_value_accuracy 신설 (KS·TVD 기반 분포 drift 측정)
- 가중치 재배분: outlier_ratio 0.10→0.05, validity 0.20→0.10, consistency 0.15→0.10,
  uniqueness 0.20→0.15, completeness 0.25→0.20, value_accuracy 0.30 신설
- compute_dsc에 reference_df 인자 추가
- 노트북 01·02 호출부에 reference_df 전달 추가
"""
import json

BASE = 'G:/내 드라이브/capstone/dsc/notebooks'

NEW_DSC_ENGINE = '''# ============================================================
# 2-1. DSC 스코어링 엔진 정의 (v3 — outlier reference 고정 + value_accuracy 신설)
# ============================================================
import numpy as np
import re
from scipy import stats as sp_stats


def calc_completeness(df, target_col, placeholder_numerical=-1, placeholder_categorical='empty'):
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


def calc_consistency(df, target_col, categorical_cols):
    """범주형 표현 일관성 — 오염 시그니처 행 비율 측정."""
    if not categorical_cols:
        return 1.0
    scores = []
    for col in categorical_cols:
        if col not in df.columns:
            continue
        s = df[col].dropna().astype(str)
        if len(s) == 0:
            scores.append(1.0)
            continue
        has_suffix = s.apply(lambda x: bool(re.search(r'-\\d+$', x)))
        scores.append(1.0 - has_suffix.mean())
    return float(np.mean(scores)) if scores else 1.0


def calc_outlier_ratio(df, target_col, numerical_cols, reference_df=None):
    """IQR 기반 outlier가 아닌 비율.
    v3: reference_df의 IQR을 고정 기준으로 사용 → 노이즈가 IQR을 부풀려 outlier가
    줄어드는 자기참조 함정 회피."""
    if not numerical_cols:
        return 1.0
    scores = []
    for col in numerical_cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(s) < 4:
            scores.append(1.0)
            continue
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
            scores.append(1.0)
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_count = ((s < lower) | (s > upper)).sum()
        scores.append(1.0 - outlier_count / len(s))
    return float(np.mean(scores)) if scores else 1.0


def calc_class_balance(df, target_col):
    """클래스 균형 — 최소 비율 / 이상 비율."""
    counts = df[target_col].value_counts()
    n_classes = len(counts)
    if n_classes <= 1:
        return 1.0
    min_ratio = counts.min() / counts.sum()
    ideal_ratio = 1.0 / n_classes
    return min(min_ratio / ideal_ratio, 1.0)


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


def calc_value_accuracy(df, target_col, numerical_cols, categorical_cols, reference_df=None):
    """값 정확성 — reference 분포 대비 drift.
    수치형: 1 - KS statistic. 범주형: 1 - Total Variation Distance.
    reference 없으면 1.0 (지표 비활성)."""
    if reference_df is None:
        return 1.0
    scores = []
    for col in numerical_cols:
        if col not in df.columns or col not in reference_df.columns:
            continue
        ref = pd.to_numeric(reference_df[col], errors='coerce').dropna()
        cur = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(ref) < 5 or len(cur) < 5:
            scores.append(1.0)
            continue
        ks_stat, _ = sp_stats.ks_2samp(ref, cur)
        scores.append(1.0 - float(ks_stat))
    for col in categorical_cols:
        if col not in df.columns or col not in reference_df.columns:
            continue
        ref_counts = reference_df[col].astype(str).value_counts(normalize=True)
        cur_counts = df[col].astype(str).value_counts(normalize=True)
        all_cats = ref_counts.index.union(cur_counts.index)
        ref_p = ref_counts.reindex(all_cats, fill_value=0)
        cur_p = cur_counts.reindex(all_cats, fill_value=0)
        tvd = 0.5 * (ref_p - cur_p).abs().sum()
        scores.append(1.0 - float(tvd))
    return float(np.mean(scores)) if scores else 1.0


DEFAULT_WEIGHTS = {
    'completeness': 0.20, 'uniqueness': 0.15, 'validity': 0.10,
    'consistency': 0.10, 'outlier_ratio': 0.05,
    'class_balance': 0.05, 'feature_correlation': 0.05,
    'value_accuracy': 0.30,
}


def compute_dsc(df, target_col, numerical_cols, categorical_cols,
                weights=None,
                placeholder_numerical=-1,
                placeholder_categorical='empty',
                reference_df=None):
    """DSC 점수(0~100) + 등급 + 지표별 점수.
    v3: value_accuracy 신설, outlier_ratio reference 고정, 가중치 재배분.
    reference_df: 베이스라인 데이터(보통 clean df). 베이스라인 자체 측정 시 자기 자신을 넘김."""
    w = weights or DEFAULT_WEIGHTS
    metrics = {
        'completeness':        calc_completeness(df, target_col, placeholder_numerical, placeholder_categorical),
        'uniqueness':          calc_uniqueness(df, target_col),
        'validity':            calc_validity(df, target_col, numerical_cols, categorical_cols),
        'consistency':         calc_consistency(df, target_col, categorical_cols),
        'outlier_ratio':       calc_outlier_ratio(df, target_col, numerical_cols, reference_df=reference_df),
        'class_balance':       calc_class_balance(df, target_col),
        'feature_correlation': calc_feature_correlation(df, target_col, numerical_cols),
        'value_accuracy':      calc_value_accuracy(df, target_col, numerical_cols, categorical_cols, reference_df=reference_df),
    }
    score = sum(metrics[k] * w[k] for k in w) * 100
    if score >= 90:   grade = 'A'
    elif score >= 75: grade = 'B'
    elif score >= 60: grade = 'C'
    else:             grade = 'D'
    return {'score': round(score, 2), 'grade': grade,
            **{k: round(v, 4) for k, v in metrics.items()}}


print('DSC 스코어링 엔진 v3 정의 완료')
print('  변경: value_accuracy 신설, outlier_ratio reference 고정, 가중치 재배분')'''


def replace_engine_cell(nb, cell_idx):
    nb['cells'][cell_idx]['source'] = [NEW_DSC_ENGINE]
    print(f'  cell {cell_idx} DSC 엔진 v3 교체')


def patch_call_in_cell(nb, cell_idx, ref_var_name):
    """cell 안의 compute_dsc 호출 끝에 reference_df= 인자 추가."""
    src = ''.join(nb['cells'][cell_idx]['source'])
    # 기존 호출 패턴: placeholder_categorical=...,\n        )  또는  )\n
    # 두 가지 형태 모두 처리
    new_arg = f'        reference_df={ref_var_name},\n'
    # 들여쓰기 4칸과 8칸 두 가지 호출이 있을 수 있음
    patterns = [
        ("placeholder_categorical=meta.get('placeholder_categorical', 'empty'),\n    )",
         f"placeholder_categorical=meta.get('placeholder_categorical', 'empty'),\n    reference_df={ref_var_name},\n    )"),
        ("placeholder_categorical=meta.get('placeholder_categorical', 'empty'),\n        )",
         f"placeholder_categorical=meta.get('placeholder_categorical', 'empty'),\n        reference_df={ref_var_name},\n        )"),
        ("placeholder_categorical=meta.get('placeholder_categorical', 'empty'),\n            )",
         f"placeholder_categorical=meta.get('placeholder_categorical', 'empty'),\n            reference_df={ref_var_name},\n            )"),
    ]
    if 'reference_df' in src:
        print(f'  cell {cell_idx} 이미 reference_df 포함 → 패치 건너뜀')
        return
    patched = False
    for old, new in patterns:
        if old in src:
            src = src.replace(old, new)
            patched = True
            break
    if not patched:
        print(f'  cell {cell_idx} 호출 패턴 매칭 실패 — 수동 확인 필요')
        return
    nb['cells'][cell_idx]['source'] = [src]
    print(f'  cell {cell_idx} compute_dsc 호출에 reference_df={ref_var_name} 추가')


# ============================================================
# 노트북 01 패치
# ============================================================
print('=== 노트북 01 패치 ===')
path = f'{BASE}/01_setup_and_baseline.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)
replace_engine_cell(nb, 11)
patch_call_in_cell(nb, 13, ref_var_name='df')  # 베이스라인은 자기 자신
with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('  저장 완료')

# ============================================================
# 노트북 02 패치
# ============================================================
print('=== 노트북 02 패치 ===')
path = f'{BASE}/02_pollution_and_dsc.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)
replace_engine_cell(nb, 8)
patch_call_in_cell(nb, 9, ref_var_name='df_clean')  # 폴루션 데이터는 clean이 reference
with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print('  저장 완료')

print()
print('=== 적용 완료 ===')
