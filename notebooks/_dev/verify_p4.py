"""P4 가중치 재배분 단위 검증.

변경: class_balance 0.05 → 0.10, validity 0.10 → 0.05
근거: validity가 P2에서 사실상 무력화 (타입만 검사 → feature_accuracy 무감지)
      → 그 자리를 class_balance 강화에 양보.
검증 기준:
- letter class_balance 0.75 케이스의 ΔDSC가 v3 대비 1.5~2배로 늘어나야 함
- baseline 점수에 큰 변화 없어야 함 (validity는 baseline에서 1.0이므로 0.05만 줄어듦)
"""
import json
import pandas as pd
import numpy as np
import re
from scipy import stats as sp_stats

# 노트북 02 cell 8에서 v3 엔진 코드 추출
nb = json.load(open('G:/내 드라이브/capstone/dsc/notebooks/02_pollution_and_dsc.ipynb', encoding='utf-8'))
engine_src = ''.join(nb['cells'][8]['source'])
ns = {'pd': pd, 'np': np, 're': re, 'sp_stats': sp_stats}
exec(engine_src, ns)

compute_dsc = ns['compute_dsc']

WEIGHTS_V31 = {
    'completeness': 0.20, 'uniqueness': 0.15, 'validity': 0.05,
    'consistency': 0.10, 'outlier_ratio': 0.05,
    'class_balance': 0.10, 'feature_correlation': 0.05,
    'value_accuracy': 0.30,
}
assert abs(sum(WEIGHTS_V31.values()) - 1.0) < 1e-9, '가중치 합 != 1'

DATASETS = {
    'TelcoCustomerChurn': {
        'path': 'G:/내 드라이브/capstone/dsc/data/raw/TelcoCustomerChurn.csv',
        'target': 'Churn',
        'numerical_cols': ['tenure','MonthlyCharges','TotalCharges'],
        'categorical_cols': ['gender','SeniorCitizen','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod'],
    },
    'SouthGermanCredit': {
        'path': 'G:/내 드라이브/capstone/dsc/data/raw/SouthGermanCredit.csv',
        'target': 'credit_risk',
        'numerical_cols': ['duration','amount','age'],
        'categorical_cols': ['status','credit_history','purpose','savings','employment_duration','installment_rate','personal_status_sex','other_debtors','present_residence','property','other_installment_plans','housing','number_credits','job','people_liable','telephone','foreign_worker'],
    },
    'letter': {
        'path': 'G:/내 드라이브/capstone/dsc/data/raw/letter.csv',
        'target': 'lettr',
        'numerical_cols': ['x-box','y-box','width','high','onpix','x-bar','y-bar','x2bar','y2bar','xybar','x2ybr','xy2br','x-ege','xegvy','y-ege','yegvx'],
        'categorical_cols': [],
    },
}

print(f'{"dataset":22s} {"polluter":18s} {"lvl":>5s} {"DSC v3":>10s} {"DSC v3.1":>10s} {"Δ":>8s}')
print('=' * 70)

scenarios = [
    ('letter', 'class_balance', 0.75),
    ('letter', 'class_balance', 0.5),
    ('letter', 'class_balance', 0.25),
    ('SouthGermanCredit', 'class_balance', 0.75),
    ('TelcoCustomerChurn', 'class_balance', 0.75),
    ('letter', 'feature_accuracy', 0.75),  # 비교: 다른 폴루션도 영향 보는지
    ('letter', 'completeness', 0.75),
    ('letter', 'none', 0.0),
    ('SouthGermanCredit', 'none', 0.0),
    ('TelcoCustomerChurn', 'none', 0.0),
]

for ds, pol, lvl in scenarios:
    meta = DATASETS[ds]
    df_clean = pd.read_csv(meta['path'])
    if 'TotalCharges' in df_clean.columns:
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce').fillna(0)
    if pol == 'none':
        df_target = df_clean
    else:
        path = f'G:/내 드라이브/capstone/dsc/data/polluted/{ds}/{pol}_{int(lvl*100)}/data.csv'
        df_target = pd.read_csv(path)
        if 'TotalCharges' in df_target.columns:
            df_target['TotalCharges'] = pd.to_numeric(df_target['TotalCharges'], errors='coerce').fillna(0)

    res_v3 = compute_dsc(df_target, target_col=meta['target'],
                         numerical_cols=meta['numerical_cols'],
                         categorical_cols=meta['categorical_cols'],
                         reference_df=df_clean)
    res_v31 = compute_dsc(df_target, target_col=meta['target'],
                          numerical_cols=meta['numerical_cols'],
                          categorical_cols=meta['categorical_cols'],
                          reference_df=df_clean,
                          weights=WEIGHTS_V31)
    print(f'{ds:22s} {pol:18s} {lvl:>5.2f} {res_v3["score"]:>10.2f} {res_v31["score"]:>10.2f} {res_v31["score"]-res_v3["score"]:>+8.2f}')
