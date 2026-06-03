# 노트북 03 회귀 버전 실행 로그

- **실행 시각**: 2026-06-03 13:23:39
- **총 학습**: 324건
- **에러**: 0건

## 1. 모델 성능 요약 (R² mean by dataset × model)

| model            |   BikeSharing |   CaliforniaHousing |   WineQuality |
|:-----------------|--------------:|--------------------:|--------------:|
| LinearRegression |        0.3098 |              0.417  |        0.1472 |
| MLPReg           |        0.7372 |              0.6068 |        0.1696 |
| RandomForestReg  |        0.7695 |              0.6425 |        0.2992 |
| XGBoostReg       |        0.7518 |              0.6282 |        0.2214 |

## 2. 폴루터별 평균 R² 하락

| 데이터셋 | baseline R² | completeness Δ | uniqueness Δ | feature_accuracy Δ | consistent_repr Δ | target_distribution_skew Δ |
|---|---|---|---|---|---|---|
| CaliforniaHousing | 0.7565 | -0.3211 | -0.0050 | -0.4000 | - | -0.0358 |
| BikeSharing | 0.8105 | -0.3941 | -0.0009 | -0.3982 | -0.0257 | -0.0515 |
| WineQuality | 0.3853 | -0.2573 | -0.0340 | -0.2702 | - | -0.1718 |