"""P3 consistency 재설계 검증.

기존(v3) 방식: 정규식 r'-\\d+$' 매칭 → polluter 시그니처에만 작동 (toy metric)
신(v3.2) 방식: reference 대비 카디널리티 비율 (ref_n / cur_n) → 일반화된 카디널리티 drift

검증 기준:
- consistent_repr 폴루션에서 신 metric이 강도와 음의 상관 (기존만큼)
- 다른 폴루션에서 신 metric이 안정적 (false positive 없음)
- 일반화: '-숫자' 외 임의 카테고리 추가도 감지하는지
"""
import json
import pandas as pd
import numpy as np
import re

# 기존 v3 로직
def calc_consistency_v3(df, target_col, categorical_cols):
    if not categorical_cols:
        return 1.0
    scores = []
    for col in categorical_cols:
        if col not in df.columns:
            continue
        s = df[col].dropna().astype(str)
        if len(s) == 0:
            scores.append(1.0); continue
        has_suffix = s.apply(lambda x: bool(re.search(r'-\d+$', x)))
        scores.append(1.0 - has_suffix.mean())
    return float(np.mean(scores)) if scores else 1.0


# 신 v3.2 로직 — reference 카디널리티 기반
def calc_consistency_v32(df, target_col, categorical_cols, reference_df=None):
    if not categorical_cols:
        return 1.0
    scores = []
    for col in categorical_cols:
        if col not in df.columns:
            continue
        cur_n = df[col].dropna().astype(str).nunique()
        if cur_n == 0:
            scores.append(1.0); continue
        if reference_df is not None and col in reference_df.columns:
            ref_n = reference_df[col].dropna().astype(str).nunique()
            if cur_n <= ref_n:
                scores.append(1.0)
            else:
                scores.append(ref_n / cur_n)
        else:
            # reference 없으면 기존 v3 fallback
            s = df[col].dropna().astype(str)
            has_suffix = s.apply(lambda x: bool(re.search(r'-\d+$', x)))
            scores.append(1.0 - has_suffix.mean())
    return float(np.mean(scores)) if scores else 1.0


CATCOLS = {
    'TelcoCustomerChurn': ['gender','SeniorCitizen','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod'],
    'SouthGermanCredit': ['status','credit_history','purpose','savings','employment_duration','installment_rate','personal_status_sex','other_debtors','present_residence','property','other_installment_plans','housing','number_credits','job','people_liable','telephone','foreign_worker'],
}

print(f'{"dataset":22s} {"polluter":18s} {"lvl":>5s} {"v3":>8s} {"v3.2":>8s} {"Δ":>8s}')
print('=' * 70)

scenarios = [
    ('TelcoCustomerChurn', 'consistent_repr', 0.1),
    ('TelcoCustomerChurn', 'consistent_repr', 0.5),
    ('TelcoCustomerChurn', 'consistent_repr', 0.75),
    ('SouthGermanCredit', 'consistent_repr', 0.1),
    ('SouthGermanCredit', 'consistent_repr', 0.5),
    ('SouthGermanCredit', 'consistent_repr', 0.75),
    ('TelcoCustomerChurn', 'feature_accuracy', 0.75),
    ('TelcoCustomerChurn', 'completeness', 0.75),
    ('TelcoCustomerChurn', 'uniqueness', 0.75),
    ('SouthGermanCredit', 'feature_accuracy', 0.75),
]

for ds, pol, lvl in scenarios:
    catcols = CATCOLS[ds]
    df_clean = pd.read_csv(f'G:/내 드라이브/capstone/dsc/data/raw/{ds}.csv')
    df_p = pd.read_csv(f'G:/내 드라이브/capstone/dsc/data/polluted/{ds}/{pol}_{int(lvl*100)}/data.csv')
    target = 'Churn' if ds=='TelcoCustomerChurn' else 'credit_risk'
    s3 = calc_consistency_v3(df_p, target, catcols)
    s32 = calc_consistency_v32(df_p, target, catcols, reference_df=df_clean)
    print(f'{ds:22s} {pol:18s} {lvl:>5.2f} {s3:>8.4f} {s32:>8.4f} {s32-s3:>+8.4f}')


# 일반화 테스트: 정규식 패턴이 아닌 단순 카테고리 추가
print()
print('=== 일반화 테스트: -숫자 패턴 외 임의 카테고리 추가 ===')
df_clean = pd.read_csv('G:/내 드라이브/capstone/dsc/data/raw/TelcoCustomerChurn.csv')
df_dirty = df_clean.copy()
# gender 컬럼에 'unknown' 카테고리를 30% 행에 주입 (대소문자 변형, 동의어 등)
mask = df_dirty.sample(frac=0.3, random_state=42).index
df_dirty.loc[mask, 'gender'] = df_dirty.loc[mask, 'gender'].apply(lambda x: x.upper())  # MALE/FEMALE 추가
target = 'Churn'
catcols = CATCOLS['TelcoCustomerChurn']
s3 = calc_consistency_v3(df_dirty, target, catcols)
s32 = calc_consistency_v32(df_dirty, target, catcols, reference_df=df_clean)
print(f'gender 30%를 대문자화 (Male→MALE):  v3={s3:.4f}  v3.2={s32:.4f}')
print('  v3는 -숫자 패턴이 없어 1.0(못잡음), v3.2는 카디널리티 증가로 감지')
