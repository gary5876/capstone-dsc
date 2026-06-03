# 노트북 01 회귀 버전 실행 로그

- **실행 시각**: 2026-06-03 09:29:15
- **BASE 경로**: /content/drive/MyDrive/capstone/dsc
- **DSC framework**: dsc_framework.regression_cell (v5 사전등록)

## 1. 데이터셋

| 데이터셋 | 행 | 열(after drop) | 타겟 | nunique | dtype |
|---|---|---|---|---|---|
| CaliforniaHousing | 20,640 | 9 | MedHouseVal | 3842 | float64 |
| BikeSharing | 17,379 | 13 | cnt | 869 | int64 |
| WineQuality | 6,497 | 12 | quality | 7 | int64 |

## 2. 베이스라인 DSC 점수 (회귀 cell)

| 데이터셋 | DSC Score | 등급 | completeness | uniqueness | validity | consistency | outlier_ratio | target_distribution_quality | feature_correlation | target_smoothness | feature_informativeness_reg |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CaliforniaHousing | 89.74 | B | 1.0 | 1.0 | 1.0 | 1.0 | 0.9726 | 0.9113 | 1.0 | 0.7171 | 0.6419 |
| BikeSharing | 88.05 | B | 1.0 | 0.9999 | 1.0 | 1.0 | 0.997 | 0.7285 | 0.9524 | 0.6757 | 0.7502 |
| WineQuality | 79.39 | B | 1.0 | 0.8185 | 1.0 | 1.0 | 0.9722 | 0.5532 | 1.0 | 0.6007 | 0.4704 |

## 3. 베이스라인 모델 R²

| 데이터셋 | 모델 | R² | R²_clipped |
|---|---|---|---|
| CaliforniaHousing | LinearRegression | 0.5966 | 0.5966 |
| CaliforniaHousing | RandomForestReg | 0.8065 | 0.8065 |
| CaliforniaHousing | XGBoostReg | 0.8327 | 0.8327 |
| CaliforniaHousing | SVR | 0.733 | 0.733 |
| CaliforniaHousing | MLPReg | 0.7397 | 0.7397 |
| BikeSharing | LinearRegression | 0.4073 | 0.4073 |
| BikeSharing | RandomForestReg | 0.945 | 0.945 |
| BikeSharing | XGBoostReg | 0.9496 | 0.9496 |
| BikeSharing | SVR | 0.4205 | 0.4205 |
| BikeSharing | MLPReg | 0.9532 | 0.9532 |
| WineQuality | LinearRegression | 0.2736 | 0.2736 |
| WineQuality | RandomForestReg | 0.4905 | 0.4905 |
| WineQuality | XGBoostReg | 0.4122 | 0.4122 |
| WineQuality | SVR | 0.3872 | 0.3872 |
| WineQuality | MLPReg | 0.0967 | 0.0967 |

## 4. 산출물

- `/content/drive/MyDrive/capstone/dsc/results/dsc_scores_regression.csv` — 베이스라인 DSC 점수 3건
- `/content/drive/MyDrive/capstone/dsc/results/model_performance_regression.csv` — 베이스라인 모델 R² 15건

---
*이 로그는 노트북 01 회귀 버전 실행 시 자동 생성됨*