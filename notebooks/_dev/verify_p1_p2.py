"""P1+P2 새 DSC 지표 단위 검증 스크립트.

진단 보고서 §2 P1: outlier_ratio가 가우시안 노이즈에 역반응 (letter feature_accuracy 0.75 → outlier_ratio 0.967→0.992 상승)
진단 보고서 §2 P2: validity가 타입만 검사 → feature_accuracy 오염 0% 감지

검증 기준:
- letter feature_accuracy 0.75에서 새 value_accuracy가 베이스라인 대비 유의미하게 하락해야 함
- letter feature_accuracy 0.75에서 새 outlier_ratio(reference 기반)가 더 이상 상승하지 않아야 함
- completeness/uniqueness/consistent_repr 폴루션에서는 value_accuracy가 크게 하락하지 않아야 함 (false positive 방지)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

BASE = Path('G:/내 드라이브/capstone/dsc')
RAW = BASE / 'data' / 'raw'
POLLUTED = BASE / 'data' / 'polluted'

DATASETS = {
    'TelcoCustomerChurn': {
        'path': RAW / 'TelcoCustomerChurn.csv',
        'target': 'Churn',
        'numerical_cols': ['tenure', 'MonthlyCharges', 'TotalCharges'],
        'categorical_cols': [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents',
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod'
        ],
    },
    'SouthGermanCredit': {
        'path': RAW / 'SouthGermanCredit.csv',
        'target': 'credit_risk',
        'numerical_cols': ['duration', 'amount', 'age'],
        'categorical_cols': [
            'status', 'credit_history', 'purpose', 'savings',
            'employment_duration', 'installment_rate', 'personal_status_sex',
            'other_debtors', 'present_residence', 'property',
            'other_installment_plans', 'housing', 'number_credits',
            'job', 'people_liable', 'telephone', 'foreign_worker'
        ],
    },
    'letter': {
        'path': RAW / 'letter.csv',
        'target': 'lettr',
        'numerical_cols': [
            'x-box', 'y-box', 'width', 'high', 'onpix', 'x-bar',
            'y-bar', 'x2bar', 'y2bar', 'xybar', 'x2ybr', 'xy2br',
            'x-ege', 'xegvy', 'y-ege', 'yegvx'
        ],
        'categorical_cols': [],
    },
}


def calc_outlier_ratio_v3(df, target_col, numerical_cols, reference_df=None):
    """IQR 기반 outlier가 아닌 비율.
    v3: reference_df의 IQR을 고정 기준으로 사용해 노이즈 자기참조 회피.
    """
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


def calc_value_accuracy(df, target_col, numerical_cols, categorical_cols, reference_df=None):
    """값 정확성 — reference 분포 대비 drift 측정.
    수치형: 1 - KS statistic (양 분포의 누적분포 최대 거리).
    범주형: 1 - Total Variation Distance (카테고리 확률 분포 차이의 절반).
    reference 없으면 1.0 (지표 비활성).
    """
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


# ============================================================
# 검증 — 베이스라인 vs 각 폴루션 강도에서 두 지표 변화
# ============================================================
print('=' * 80)
print('P1 (outlier 역반응) + P2 (value 미감지) 새 지표 검증')
print('=' * 80)

scenarios = [
    ('letter', 'feature_accuracy', [0.1, 0.25, 0.5, 0.75]),
    ('letter', 'completeness', [0.1, 0.5, 0.75]),
    ('letter', 'uniqueness', [0.1, 0.5, 0.75]),
    ('SouthGermanCredit', 'feature_accuracy', [0.1, 0.5, 0.75]),
    ('SouthGermanCredit', 'completeness', [0.1, 0.75]),
    ('SouthGermanCredit', 'consistent_repr', [0.1, 0.75]),
    ('TelcoCustomerChurn', 'feature_accuracy', [0.1, 0.5, 0.75]),
    ('TelcoCustomerChurn', 'completeness', [0.1, 0.75]),
    ('TelcoCustomerChurn', 'consistent_repr', [0.1, 0.75]),
]

for ds_name, polluter, levels in scenarios:
    meta = DATASETS[ds_name]
    target = meta['target']
    num_cols = meta['numerical_cols']
    cat_cols = meta['categorical_cols']

    df_clean = pd.read_csv(meta['path'])
    if 'TotalCharges' in df_clean.columns:
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce').fillna(0)

    print()
    print(f'--- {ds_name} / {polluter} ---')
    base_outlier_v3 = calc_outlier_ratio_v3(df_clean, target, num_cols, reference_df=df_clean)
    base_value = calc_value_accuracy(df_clean, target, num_cols, cat_cols, reference_df=df_clean)
    print(f'  baseline           outlier_v3={base_outlier_v3:.4f}  value_accuracy={base_value:.4f}')

    for lvl in levels:
        polluted_path = POLLUTED / ds_name / f'{polluter}_{int(lvl*100)}' / 'data.csv'
        if not polluted_path.exists():
            print(f'  level={lvl:.2f}  (폴루션 파일 없음)')
            continue
        df_p = pd.read_csv(polluted_path)
        if 'TotalCharges' in df_p.columns:
            df_p['TotalCharges'] = pd.to_numeric(df_p['TotalCharges'], errors='coerce').fillna(0)
        out_v3 = calc_outlier_ratio_v3(df_p, target, num_cols, reference_df=df_clean)
        val = calc_value_accuracy(df_p, target, num_cols, cat_cols, reference_df=df_clean)
        d_out = out_v3 - base_outlier_v3
        d_val = val - base_value
        print(f'  level={lvl:.2f}  outlier_v3={out_v3:.4f}({d_out:+.4f})  value_accuracy={val:.4f}({d_val:+.4f})')

print()
print('=' * 80)
print('검증 기준 — feature_accuracy에서 value_accuracy가 강도와 음의 상관, outlier_v3는 더 이상 상승 안 함')
print('=' * 80)
