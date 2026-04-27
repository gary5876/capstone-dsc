"""F2/F4 — 주장 강도(r²) + 가법성 한계(min-aggregation) 분석.

F2: 8개 metric을 RandomForestRegressor에 입력 → F1 예측 R² 측정.
    선형 결합으로는 r²=0.37인데, 비선형 결합 시 R²이 얼마나 올라가는지.
    R² ≥ 0.5이면 "DSC 차원들은 정보가 풍부하나 선형 결합의 한계"로 방어 가능.

F4: DSC = Σ wᵢ sᵢ (가법) vs DSC_min = min(metrics) 두 모델 비교.
    어느 쪽이 ML F1과 더 잘 상관하는지.

입력: simulated_v4_merged.csv (v4 시뮬 결과, 435건 × 9 metric + F1)
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score


BASE = Path('G:/내 드라이브/capstone/dsc')
DEV = BASE / 'notebooks' / '_dev'

METRICS = [
    'completeness', 'uniqueness', 'validity', 'consistency',
    'outlier_ratio', 'class_balance', 'feature_correlation',
    'label_consistency', 'feature_informativeness',
]


def main():
    merged = pd.read_csv(DEV / 'simulated_v4_merged.csv')
    print(f'표본 수: {len(merged)}')

    # --- F2: 비선형 R² ---
    print('\n' + '=' * 80)
    print('F2 — 비선형 결합 R²')
    print('=' * 80)
    X = merged[METRICS].values
    y = merged.f1_macro.values

    # 선형 r² (비교 baseline)
    r_lin, _ = sp_stats.pearsonr(merged.score, y)
    print(f'  선형 (DSC = Σwᵢsᵢ) Pearson r = {r_lin:.4f}, r² = {r_lin**2:.4f}')

    # 비선형 R² — RF 5-fold CV
    rf = RandomForestRegressor(n_estimators=200, random_state=1, n_jobs=-1)
    cv = KFold(n_splits=5, shuffle=True, random_state=1)
    r2_scores = cross_val_score(rf, X, y, cv=cv, scoring='r2')
    print(f'  비선형 (RF 5-fold CV) R² = {r2_scores.mean():.4f} ± {r2_scores.std():.4f}')
    print(f'  fold별 R²: {[f"{x:.3f}" for x in r2_scores]}')

    # --- F4: min-aggregation 비교 ---
    print('\n' + '=' * 80)
    print('F4 — min-aggregation vs 가법 모델')
    print('=' * 80)
    merged['dsc_min'] = merged[METRICS].min(axis=1) * 100
    r_add, p_add = sp_stats.pearsonr(merged.score, y)
    r_min, p_min = sp_stats.pearsonr(merged.dsc_min, y)
    print(f'  가법 (현재) DSC = Σwᵢsᵢ:  r = {r_add:+.4f}  p = {p_add:.2e}  r² = {r_add**2:.4f}')
    print(f'  min(metrics):              r = {r_min:+.4f}  p = {p_min:.2e}  r² = {r_min**2:.4f}')
    if abs(r_add) > abs(r_min):
        print(f'  → 가법 모델이 +{abs(r_add)-abs(r_min):.4f} 더 강함. 가법 가정이 본 데이터에서는 유효.')
    else:
        print(f'  → min 모델이 +{abs(r_min)-abs(r_add):.4f} 더 강함. 최약 차원이 결정. 가법 한계 분명.')

    # 모델 클래스별로도
    print('\nmodel별 가법 vs min 비교:')
    print(f'  {"model":<22} {"r_add":>8} {"r_min":>8}  better')
    for m in sorted(merged.model.unique()):
        sub = merged[merged.model == m]
        rA, _ = sp_stats.pearsonr(sub.score, sub.f1_macro)
        rM, _ = sp_stats.pearsonr(sub.dsc_min, sub.f1_macro)
        winner = 'add' if abs(rA) > abs(rM) else 'min'
        print(f'  {m:<22} {rA:+.4f} {rM:+.4f}  {winner}')


if __name__ == '__main__':
    main()
