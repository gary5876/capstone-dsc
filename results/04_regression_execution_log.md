# 노트북 04 회귀 버전 실행 로그 — Scoreboard

- **실행 시각**: 2026-06-03 13:29:53
- **DSC 데이터**: 81건
- **모델 성능**: 324건
- **merged**: 324건

## 1. 통계 요약

- **Pearson_r**: 0.5663
- **Spearman_rho**: 0.569
- **r_squared**: 0.3207
- **nonlinear_RF_5fold_R2**: 0.5381
- **preservation_r**: 0.4698
- **polluter_holdout_pass**: 5/5
- **all_models_positive_r**: True

## 2. 가설 검증

| 가설 | 결과 |
|---|---|
| H1 r ≥ 0.4 | ✅ PASS |
| H2 ρ ≥ 0.4 | ✅ PASS |
| H3 비선형 우위 | ✅ PASS |
| H4 모든 모델 양의 r | ✅ PASS |
| H5 polluter hold-out 4/5 | ✅ PASS |

**종합**: 5/5 PASS

## 3. 모델별 r

| 모델 | r | n |
|---|---|---|
| LinearRegression | +0.6788 | 81 |
| MLPReg | +0.6275 | 81 |
| RandomForestReg | +0.5460 | 81 |
| XGBoostReg | +0.6456 | 81 |

## 4. Polluter hold-out

| Polluter (제외) | r | 결과 |
|---|---|---|
| completeness | +0.4992 | ✅ |
| consistent_repr | +0.5663 | ✅ |
| feature_accuracy | +0.6336 | ✅ |
| target_distribution_skew | +0.5624 | ✅ |
| uniqueness | +0.6060 | ✅ |
