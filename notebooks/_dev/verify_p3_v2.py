"""P3 consistency 재설계 v2 — 새 카테고리 행 비율 + placeholder 제외."""
import pandas as pd
import numpy as np
import re


def calc_consistency_v32(df, target_col, categorical_cols, reference_df=None,
                          placeholder_categorical='empty'):
    """카테고리 표현 일관성.
    reference 있을 때: reference에 없던 새 표현이 차지하는 행 비율의 보수(1-비율).
                       placeholder는 결측 신호이므로 제외(completeness와 분리).
    reference 없을 때: 기존 v3 fallback (-숫자 정규식)."""
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


CATCOLS = {
    'TelcoCustomerChurn': ['gender','SeniorCitizen','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod'],
    'SouthGermanCredit': ['status','credit_history','purpose','savings','employment_duration','installment_rate','personal_status_sex','other_debtors','present_residence','property','other_installment_plans','housing','number_credits','job','people_liable','telephone','foreign_worker'],
}

print(f'{"dataset":22s} {"polluter":18s} {"lvl":>5s} {"v3.2":>8s}')
print('=' * 60)

scenarios = [
    ('TelcoCustomerChurn', 'consistent_repr', 0.1),
    ('TelcoCustomerChurn', 'consistent_repr', 0.25),
    ('TelcoCustomerChurn', 'consistent_repr', 0.5),
    ('TelcoCustomerChurn', 'consistent_repr', 0.75),
    ('SouthGermanCredit', 'consistent_repr', 0.1),
    ('SouthGermanCredit', 'consistent_repr', 0.25),
    ('SouthGermanCredit', 'consistent_repr', 0.5),
    ('SouthGermanCredit', 'consistent_repr', 0.75),
    ('TelcoCustomerChurn', 'feature_accuracy', 0.75),  # false positive 체크
    ('TelcoCustomerChurn', 'completeness', 0.75),  # placeholder false positive 체크
    ('TelcoCustomerChurn', 'uniqueness', 0.75),
    ('SouthGermanCredit', 'completeness', 0.75),
]

for ds, pol, lvl in scenarios:
    catcols = CATCOLS[ds]
    df_clean = pd.read_csv(f'G:/내 드라이브/capstone/dsc/data/raw/{ds}.csv')
    df_p = pd.read_csv(f'G:/내 드라이브/capstone/dsc/data/polluted/{ds}/{pol}_{int(lvl*100)}/data.csv')
    target = 'Churn' if ds=='TelcoCustomerChurn' else 'credit_risk'
    s = calc_consistency_v32(df_p, target, catcols, reference_df=df_clean)
    print(f'{ds:22s} {pol:18s} {lvl:>5.2f} {s:>8.4f}')

print()
print('=== baseline (자기 자신 reference) — 1.0 이어야 함 ===')
for ds in ['TelcoCustomerChurn', 'SouthGermanCredit']:
    df_clean = pd.read_csv(f'G:/내 드라이브/capstone/dsc/data/raw/{ds}.csv')
    target = 'Churn' if ds=='TelcoCustomerChurn' else 'credit_risk'
    s = calc_consistency_v32(df_clean, target, CATCOLS[ds], reference_df=df_clean)
    print(f'  {ds}: {s:.4f}')

print()
print('=== 일반화 테스트: 대문자 변환 ===')
df_clean = pd.read_csv('G:/내 드라이브/capstone/dsc/data/raw/TelcoCustomerChurn.csv')
df_dirty = df_clean.copy()
mask = df_dirty.sample(frac=0.3, random_state=42).index
df_dirty.loc[mask, 'gender'] = df_dirty.loc[mask, 'gender'].apply(lambda x: x.upper())
s = calc_consistency_v32(df_dirty, 'Churn', CATCOLS['TelcoCustomerChurn'], reference_df=df_clean)
print(f'  gender 30% MALE/FEMALE 변환: {s:.4f}')
print('  16개 컬럼 평균이라 영향 작음. gender 단독은 0.7 정도 (1-0.3)')
