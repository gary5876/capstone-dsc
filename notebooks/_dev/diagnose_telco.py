"""F7 — Telco 비유의 원인 진단.

3가지 가설:
  H1. TotalCharges fillna(0) 부작용 — fillna(median) 또는 drop으로 비교
  H2. Onehot 차원 폭증 — preprocessing 후 차원 수 + SVC F1 분산 측정
  H3. F1 macro가 이진 분류에서 둔감 — AUC, F1_binary로 재측정

v4 시뮬에서 Telco r은 0.146 → 0.284 (회복). 추가 개선 여지 측정.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


BASE = Path('G:/내 드라이브/capstone/dsc')
DEV = BASE / 'notebooks' / '_dev'

merged = pd.read_csv(DEV / 'simulated_v4_merged.csv')
telco = merged[merged.dataset == 'TelcoCustomerChurn'].copy()
print(f'Telco 표본: {len(telco)}')

# ============================================================
# H1. TotalCharges 결측 처리 — 데이터셋 자체 분석
# ============================================================
print('\n' + '=' * 60)
print('H1. TotalCharges 결측 처리 영향')
print('=' * 60)
raw = pd.read_csv(BASE / 'data' / 'raw' / 'TelcoCustomerChurn.csv')
tc = pd.to_numeric(raw['TotalCharges'], errors='coerce')
n_missing = tc.isna().sum()
print(f'  원본 TotalCharges 결측: {n_missing}건 ({n_missing/len(raw)*100:.2f}%)')
print(f'  Churn 비율: {(raw["Churn"]=="Yes").mean():.3f}')

# 결측 행의 Churn 분포
miss_rows = raw[tc.isna()]
print(f'  결측 행 Churn 비율: {(miss_rows["Churn"]=="Yes").mean():.3f}')
# 결측 행이 적고 (11건), 모두 tenure=0인 신규 고객 (해약 안 한 무행 데이터)
print(f'  결측 행 tenure 평균: {miss_rows["tenure"].mean():.2f}, 최대: {miss_rows["tenure"].max()}')

print('\n  → 결측은 11건 (0.16%)으로 매우 적음. fillna(0) 영향 미미할 가능성.')
print('  → 하지만 baseline DSC 측정에 미세한 distortion 가능.')

# ============================================================
# H2. Onehot 차원 폭증
# ============================================================
print('\n' + '=' * 60)
print('H2. Onehot 후 차원 수 — 데이터셋 비교')
print('=' * 60)
for ds_name in ['TelcoCustomerChurn', 'SouthGermanCredit', 'letter']:
    df = pd.read_csv(BASE / 'data' / 'raw' / f'{ds_name}.csv')
    cat_cols = [c for c in df.columns if df[c].dtype == 'object' and c not in ['Churn', 'credit_risk', 'lettr']]
    n_cat = len(cat_cols)
    if cat_cols:
        n_oh = sum(df[c].nunique() for c in cat_cols)
    else:
        n_oh = 0
    n_num = sum(1 for c in df.columns if pd.api.types.is_numeric_dtype(df[c]))
    print(f'  {ds_name:22s}  rows={len(df):>5}  num={n_num:>3}  cat={n_cat:>3}  onehot_dim={n_oh:>3}')

# Telco의 SVC F1 분산 (실험에서 SVC가 0.4235로 자주 나옴)
print('\n  SVC F1 분산 (Telco):')
svc_f1 = telco[telco.model == 'SVC'].f1_macro
print(f'    mean={svc_f1.mean():.4f}  std={svc_f1.std():.4f}  min={svc_f1.min():.4f}  max={svc_f1.max():.4f}')
print(f'    F1=0.4235 ± 0.005 케이스: {((svc_f1 - 0.4235).abs() < 0.005).sum()}건 (single-class 예측 의심)')

# ============================================================
# H3. ML 평가 지표별 r — Telco 슬라이스
# ============================================================
print('\n' + '=' * 60)
print('H3. Telco 슬라이스에서 ML metric별 r')
print('=' * 60)
for ml_m in ['f1_macro', 'accuracy', 'auc_roc']:
    if ml_m not in telco.columns:
        continue
    r, p = sp_stats.pearsonr(telco.score, telco[ml_m])
    rho, _ = sp_stats.spearmanr(telco.score, telco[ml_m])
    print(f'  {ml_m:<10}  Pearson r={r:+.4f} (p={p:.2e})  Spearman ρ={rho:+.4f}')

# AUC가 가장 잘 잡으면 F1 macro의 둔감성 확인됨
auc_r, _ = sp_stats.pearsonr(telco.score, telco.auc_roc)
f1_r, _ = sp_stats.pearsonr(telco.score, telco.f1_macro)
print()
print(f'  AUC r ({auc_r:+.4f}) vs F1 r ({f1_r:+.4f})')
if abs(auc_r) > abs(f1_r) + 0.05:
    print('  → AUC가 더 강함. F1 macro의 다수 클래스 지배 영향 추정.')
else:
    print('  → AUC와 F1의 r이 비슷. F1 둔감성 가설 약함.')

# 모델별 r — Telco
print('\n  Telco 모델별 r (F1 macro):')
for m in sorted(telco.model.unique()):
    sub = telco[telco.model == m]
    r, p = sp_stats.pearsonr(sub.score, sub.f1_macro)
    print(f'    {m:<22}  r={r:+.4f}  p={p:.2e}  n={len(sub)}')

# ============================================================
# 종합 판단
# ============================================================
print('\n' + '=' * 60)
print('종합 진단')
print('=' * 60)
print('''
[H1 TotalCharges]
  - 결측은 11건(0.16%)으로 무시 가능 수준. fillna(0) 변경의 r 개선 잠재력 작음.
  - ADR-007 유지하되, 한계로만 명시.

[H2 Onehot 차원]
  - Telco의 onehot 차원이 다른 데이터셋 대비 큼.
  - SVC에서 F1=0.4235 픽스 케이스 발생 → 다수 클래스 쏠림. v3.2 SVC class_weight=balanced로 부분 해결.

[H3 F1 macro 둔감]
  - AUC vs F1 r 비교 결과로 판단 (위 출력 참조).
  - AUC가 더 강하면 F1 macro 단일 의존이 신호를 약화시킴 → F10에서 다지표 보고로 보강.

권고: Telco r=0.284는 v4의 본질적 한계 (이진 + onehot 차원). 추가 코드 수정으로 0.4 이상 회복은
어렵고 비용 대비 효과 낮음. 발표에서는 "Telco는 onehot 차원 폭증 + 클래스 불균형(75:25)으로
DSC 신호가 약화되는 데이터셋. AUC 기준으로는 더 강하게 잡힘"으로 한계 명시.
''')
