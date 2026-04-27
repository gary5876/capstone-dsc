"""기존 train_polluted/ 데이터에 v4 엔진을 적용해 새 DSC 점수 시뮬레이션.

목적: 노트북 재실행 전에 v4 → r 변화량 미리 측정.
입력: data/train_polluted/{ds}/{polluter}_{level}/train_data.csv (모든 87건)
       data/raw/{ds}.csv (reference로 train clean을 만들기 위함)
       results/model_performance.csv (학습 결과 그대로 사용)
출력: notebooks/_dev/simulated_v4_scores.csv (DSC 점수 87건)
       notebooks/_dev/simulated_v4_merged.csv (병합 435건)
       표준출력: r 변화 요약
"""
from __future__ import annotations
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import mutual_info_classif


BASE = Path('G:/내 드라이브/capstone/dsc')
RAW = BASE / 'data' / 'raw'
TRAIN = BASE / 'data' / 'train_polluted'
RESULTS = BASE / 'results'
DEV = BASE / 'notebooks' / '_dev'

DATASETS = {
    'TelcoCustomerChurn': {
        'target': 'Churn',
        'numerical_cols': ['tenure', 'MonthlyCharges', 'TotalCharges'],
        'categorical_cols': [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents',
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod',
        ],
    },
    'SouthGermanCredit': {
        'target': 'credit_risk',
        'numerical_cols': ['duration', 'amount', 'age'],
        'categorical_cols': [
            'status', 'credit_history', 'purpose', 'savings',
            'employment_duration', 'installment_rate', 'personal_status_sex',
            'other_debtors', 'present_residence', 'property',
            'other_installment_plans', 'housing', 'number_credits',
            'job', 'people_liable', 'telephone', 'foreign_worker',
        ],
    },
    'letter': {
        'target': 'lettr',
        'numerical_cols': [
            'x-box', 'y-box', 'width', 'high', 'onpix', 'x-bar',
            'y-bar', 'x2bar', 'y2bar', 'xybar', 'x2ybr', 'xy2br',
            'x-ege', 'xegvy', 'y-ege', 'yegvx',
        ],
        'categorical_cols': [],
    },
}

# ============================================================
# v4 엔진 (apply_dsc_engine_v4.py와 동일 — 외부 의존 없음, copy)
# ============================================================
def calc_completeness(df, target_col, placeholder_numerical=-1, placeholder_categorical='empty'):
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
            ph = placeholder_categorical
            missing_count += (feature_df[col].astype(str) == str(ph)).sum()
    return 1.0 - (missing_count / total_cells)


def calc_uniqueness(df, target_col):
    n = len(df)
    if n <= 1:
        return 1.0
    return 1.0 - (df.duplicated().sum() / n)


def calc_validity(df, target_col, numerical_cols, categorical_cols):
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
            scores.append(1.0); continue
        valid = s.apply(lambda x: 0 < len(x.strip()) < 200)
        scores.append(valid.mean())
    return float(np.mean(scores)) if scores else 1.0


def calc_consistency(df, target_col, categorical_cols, reference_df=None,
                     placeholder_categorical='empty'):
    if not categorical_cols:
        return 1.0
    scores = []
    for col in categorical_cols:
        if col not in df.columns:
            continue
        cur_vals = df[col].dropna().astype(str)
        if len(cur_vals) == 0:
            scores.append(1.0); continue
        ph = str(placeholder_categorical) if placeholder_categorical is not None else None
        if ph is not None:
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
    counts = df[target_col].value_counts()
    n_classes = len(counts)
    if n_classes <= 1:
        return 1.0
    min_ratio = counts.min() / counts.sum()
    ideal_ratio = 1.0 / n_classes
    return min(min_ratio / ideal_ratio, 1.0)


def calc_feature_correlation(df, target_col, numerical_cols, threshold=0.95):
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
                weights=None, placeholder_numerical=-1,
                placeholder_categorical='empty', reference_df=None):
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


# ============================================================
# 시뮬레이션 본체
# ============================================================
def make_train_clean(ds_name, meta):
    """노트북 02 cell 9와 같은 split을 재현해 train_clean(reference_df) 생성."""
    df = pd.read_csv(RAW / f'{ds_name}.csv')
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    y_enc = LabelEncoder().fit_transform(df[meta['target']].astype(str))
    train_idx, _ = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=1, stratify=y_enc
    )
    return df.iloc[train_idx].reset_index(drop=True)


def main():
    rows = []
    for ds_name, meta in DATASETS.items():
        print(f'\n=== {ds_name} ===')
        train_clean = make_train_clean(ds_name, meta)
        ds_dir = TRAIN / ds_name
        if not ds_dir.exists():
            print(f'  데이터 없음: {ds_dir}')
            continue
        polluter_dirs = sorted(ds_dir.iterdir())
        for pdir in polluter_dirs:
            name = pdir.name  # 'completeness_25' or 'none_0'
            if name == 'none_0':
                polluter, level = 'none', 0.0
            else:
                m = re.match(r'(.+)_(\d+)$', name)
                if not m:
                    continue
                polluter = m.group(1)
                level = int(m.group(2)) / 100.0
            data_path = pdir / 'train_data.csv'
            if not data_path.exists():
                continue
            df = pd.read_csv(data_path)
            if 'TotalCharges' in df.columns:
                df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
            try:
                result = compute_dsc(
                    df, target_col=meta['target'],
                    numerical_cols=meta['numerical_cols'],
                    categorical_cols=meta['categorical_cols'],
                    reference_df=train_clean,
                )
                rows.append({'dataset': ds_name, 'polluter': polluter, 'level': level, **result})
                print(f'  {polluter:20s} L{int(level*100):>3}  DSC={result["score"]:6.2f} ({result["grade"]})  '
                      f'LC={result["label_consistency"]:.3f}  FI={result["feature_informativeness"]:.3f}')
            except Exception as e:
                print(f'  {polluter}_{level} ERROR: {e}')

    df_dsc = pd.DataFrame(rows)
    df_dsc.to_csv(DEV / 'simulated_v4_scores.csv', index=False)
    print(f'\nDSC 시뮬 저장: {len(df_dsc)}건 → simulated_v4_scores.csv')

    # 병합
    mp = pd.read_csv(RESULTS / 'model_performance.csv')
    merged = df_dsc.merge(mp, on=['dataset', 'polluter', 'level'])
    merged.to_csv(DEV / 'simulated_v4_merged.csv', index=False)
    print(f'병합 결과: {len(merged)}건 → simulated_v4_merged.csv')

    # 핵심 통계
    r, p = sp_stats.pearsonr(merged.score, merged.f1_macro)
    rho, prho = sp_stats.spearmanr(merged.score, merged.f1_macro)
    print(f'\n전체 Pearson r = {r:.4f} (p={p:.2e}), Spearman ρ = {rho:.4f} (p={prho:.2e})')
    print(f'r² = {r**2:.4f} (분산 설명력)')

    # 등급별
    print('\n등급별 평균 F1:')
    for g, sub in merged.groupby('grade'):
        print(f'  {g}: 평균={sub.f1_macro.mean():.4f}, n={len(sub)}')

    # polluter별
    print('\nPolluter × Dataset 별 r:')
    for pol in sorted(merged.polluter.unique()):
        if pol == 'none':
            continue
        for ds in sorted(merged.dataset.unique()):
            sub = merged[(merged.polluter == pol) & (merged.dataset == ds)]
            if len(sub) >= 10:
                rr, _ = sp_stats.pearsonr(sub.score, sub.f1_macro)
                print(f'  {pol:20s} × {ds:20s}  r={rr:+.3f}  n={len(sub)}')


if __name__ == '__main__':
    main()
