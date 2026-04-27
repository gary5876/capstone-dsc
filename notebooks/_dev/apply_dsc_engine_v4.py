"""DSC 엔진 v4 적용 — F3(value_accuracy 정의 모순) 해결.

v4 변경점:
- value_accuracy 제거
- label_consistency 신설 (k-NN, chance level 보정, dedup 전처리)
- feature_informativeness 신설 (mutual_info_classif 합/H(Y) 정규화, dedup 전처리)
- 가중치 재배분: value_accuracy 0.30 → label_consistency 0.20 + feature_informativeness 0.10

검증 통과 기준 (verify_v4_metrics.py 4/4 통과):
- baseline에서 LC > 0 (chance 초과), FI > 0.1
- feature_accuracy/completeness 0.75 → LC, FI 둘 다 음의 신호
- uniqueness 0.75 → 두 지표 거의 변화 없음 (|Δ| ≤ 0.05) — drop_duplicates 효과
"""
import json

BASE = 'G:/내 드라이브/capstone/dsc/notebooks'

NEW_DSC_ENGINE = '''# ============================================================
# 2-1. DSC 스코어링 엔진 정의 (v4 — value_accuracy 제거, 절대품질 지표 신설)
# ============================================================
import numpy as np
import re
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import mutual_info_classif


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


def calc_consistency(df, target_col, categorical_cols, reference_df=None,
                     placeholder_categorical='empty'):
    """카테고리 표현 일관성 — reference 있을 때 새 표현 비율의 보수, placeholder 제외."""
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
            has_suffix = cur_vals.apply(lambda x: bool(re.search(r'-\\d+$', x)))
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
        mi = mutual_info_classif(X, y, discrete_features=discrete_mask, random_state=random_state)
    except Exception:
        return 1.0
    class_props = np.bincount(y) / len(y)
    class_props = class_props[class_props > 0]
    h_y = float(-(class_props * np.log(class_props)).sum())
    if h_y <= 0:
        return 1.0
    return float(np.clip(mi.sum() / h_y, 0.0, 1.0))


DEFAULT_WEIGHTS = {
    'completeness': 0.20, 'uniqueness': 0.15, 'validity': 0.05,
    'consistency': 0.10, 'outlier_ratio': 0.05,
    'class_balance': 0.10, 'feature_correlation': 0.05,
    'label_consistency': 0.20, 'feature_informativeness': 0.10,
}


def compute_dsc(df, target_col, numerical_cols, categorical_cols,
                weights=None,
                placeholder_numerical=-1,
                placeholder_categorical='empty',
                reference_df=None):
    """DSC 점수(0~100) + 등급 + 지표별 점수.
    v4: value_accuracy 제거, label_consistency·feature_informativeness 신설.
    reference_df: outlier·consistency의 reference anchor (없어도 70% 작동)."""
    w = weights or DEFAULT_WEIGHTS
    metrics = {
        'completeness':            calc_completeness(df, target_col, placeholder_numerical, placeholder_categorical),
        'uniqueness':              calc_uniqueness(df, target_col),
        'validity':                calc_validity(df, target_col, numerical_cols, categorical_cols),
        'consistency':             calc_consistency(df, target_col, categorical_cols, reference_df=reference_df, placeholder_categorical=placeholder_categorical),
        'outlier_ratio':           calc_outlier_ratio(df, target_col, numerical_cols, reference_df=reference_df),
        'class_balance':           calc_class_balance(df, target_col),
        'feature_correlation':     calc_feature_correlation(df, target_col, numerical_cols),
        'label_consistency':       calc_label_consistency(df, target_col, numerical_cols),
        'feature_informativeness': calc_feature_informativeness(df, target_col, numerical_cols, categorical_cols),
    }
    score = sum(metrics[k] * w[k] for k in w) * 100
    if score >= 90:   grade = 'A'
    elif score >= 75: grade = 'B'
    elif score >= 60: grade = 'C'
    else:             grade = 'D'
    return {'score': round(score, 2), 'grade': grade,
            **{k: round(v, 4) for k, v in metrics.items()}}


print('DSC 스코어링 엔진 v4 정의 완료')
print('  변경: value_accuracy 제거 → label_consistency(0.20) + feature_informativeness(0.10) 신설')'''


def replace_engine_cell(nb_path, cell_idx):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    nb['cells'][cell_idx]['source'] = [NEW_DSC_ENGINE]
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f'  {nb_path.split("/")[-1]} cell {cell_idx} 교체 완료')


print('=== 노트북 01 패치 ===')
replace_engine_cell(f'{BASE}/01_setup_and_baseline.ipynb', 11)

print('=== 노트북 02 패치 ===')
replace_engine_cell(f'{BASE}/02_pollution_and_dsc.ipynb', 8)

print()
print('=== v4 적용 완료 ===')
print('compute_dsc 호출부 (cell 13 in nb01, cell 9 in nb02)는 reference_df 인자 그대로 유지.')
print('value_accuracy 컬럼이 dsc_scores.csv에 더 이상 없게 됨. 04 노트북은 자동 적응 (컬럼 누락 검사 없음).')
