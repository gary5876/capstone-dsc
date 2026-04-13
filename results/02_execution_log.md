# 노트북 02 실행 로그: 오염 생성 & DSC 점수 계산

- **실행 시각**: 2026-04-12 13:54:46
- **총 실험**: 59건 (베이스라인 포함)
- **소요 시간**: 111초
- **에러**: 0건

## 1. 오염 설정

- **데이터셋**: ['TelcoCustomerChurn', 'SouthGermanCredit', 'letter']
- **오염 강도**: [0.1, 0.25, 0.5, 0.75]
- **Polluter**: CompletenessPolluter, UniquenessPolluter, FeatureAccuracyPolluter, ConsistentRepresentationPolluter, ClassBalancePolluter

## 2. DSC 점수 결과

| 데이터셋 | 오염 유형 | 강도 | DSC Score | 등급 | completeness | uniqueness | validity | consistency | outlier_ratio | class_balance | feature_corr |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TelcoCustomerChurn | none | 0.0 | 97.65 | A | 1.0 | 1.0 | 0.9999 | 1.0 | 1.0 | 0.5307 | 1.0 |
| SouthGermanCredit | none | 0.0 | 97.45 | A | 1.0 | 1.0 | 1.0 | 1.0 | 0.945 | 0.6 | 1.0 |
| letter | none | 0.0 | 98.11 | A | 1.0 | 0.9334 | 1.0 | 1.0 | 0.9671 | 0.9542 | 1.0 |
| TelcoCustomerChurn | completeness | 0.1 | 95.17 | A | 0.905 | 1.0 | 1.0 | 0.9938 | 0.9985 | 0.5307 | 1.0 |
| TelcoCustomerChurn | uniqueness | 0.1 | 90.99 | A | 1.0 | 0.6666 | 1.0 | 1.0 | 1.0 | 0.5308 | 1.0 |
| TelcoCustomerChurn | feature_accuracy | 0.1 | 97.65 | A | 1.0 | 1.0 | 1.0 | 1.0 | 0.9997 | 0.5307 | 1.0 |
| TelcoCustomerChurn | consistent_repr | 0.1 | 96.25 | A | 1.0 | 1.0 | 1.0 | 0.9063 | 1.0 | 0.5307 | 1.0 |
| TelcoCustomerChurn | class_balance | 0.1 | 99.48 | A | 1.0 | 1.0 | 1.0 | 1.0 | 0.9981 | 0.9005 | 1.0 |
| TelcoCustomerChurn | completeness | 0.25 | 91.34 | A | 0.7626 | 1.0 | 1.0 | 0.9844 | 0.9854 | 0.5307 | 1.0 |
| TelcoCustomerChurn | uniqueness | 0.25 | 87.65 | B | 1.0 | 0.5 | 1.0 | 1.0 | 1.0 | 0.5307 | 1.0 |
| TelcoCustomerChurn | feature_accuracy | 0.25 | 97.65 | A | 1.0 | 1.0 | 1.0 | 1.0 | 0.9993 | 0.5307 | 1.0 |
| TelcoCustomerChurn | consistent_repr | 0.25 | 94.14 | A | 1.0 | 1.0 | 1.0 | 0.7657 | 1.0 | 0.5307 | 1.0 |
| TelcoCustomerChurn | class_balance | 0.25 | 98.75 | A | 1.0 | 1.0 | 1.0 | 1.0 | 0.9996 | 0.7501 | 1.0 |
| TelcoCustomerChurn | completeness | 0.5 | 84.79 | B | 0.5251 | 1.0 | 1.0 | 0.9688 | 0.948 | 0.5307 | 1.0 |
| TelcoCustomerChurn | uniqueness | 0.5 | 84.32 | B | 1.0 | 0.3333 | 1.0 | 1.0 | 1.0 | 0.5307 | 1.0 |
| TelcoCustomerChurn | feature_accuracy | 0.5 | 97.62 | A | 1.0 | 1.0 | 1.0 | 1.0 | 0.9969 | 0.5307 | 1.0 |
| TelcoCustomerChurn | consistent_repr | 0.5 | 90.62 | A | 1.0 | 1.0 | 1.0 | 0.5313 | 1.0 | 0.5307 | 1.0 |
| TelcoCustomerChurn | class_balance | 0.5 | 97.5 | A | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.5003 | 1.0 |
| TelcoCustomerChurn | completeness | 0.75 | 76.34 | B | 0.2875 | 0.9783 | 1.0 | 0.9531 | 0.7631 | 0.5307 | 1.0 |
| TelcoCustomerChurn | uniqueness | 0.75 | 82.65 | B | 1.0 | 0.25 | 1.0 | 1.0 | 1.0 | 0.5307 | 1.0 |
| TelcoCustomerChurn | feature_accuracy | 0.75 | 97.61 | A | 1.0 | 1.0 | 1.0 | 1.0 | 0.9956 | 0.5307 | 1.0 |
| TelcoCustomerChurn | consistent_repr | 0.75 | 87.11 | B | 1.0 | 1.0 | 1.0 | 0.2969 | 1.0 | 0.5307 | 1.0 |
| TelcoCustomerChurn | class_balance | 0.75 | 96.25 | A | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.2504 | 1.0 |
| SouthGermanCredit | completeness | 0.1 | 94.6 | A | 0.9 | 1.0 | 1.0 | 1.0 | 0.9097 | 0.6 | 1.0 |
| SouthGermanCredit | uniqueness | 0.1 | 90.76 | A | 1.0 | 0.6667 | 1.0 | 1.0 | 0.9422 | 0.6 | 1.0 |
| SouthGermanCredit | feature_accuracy | 0.1 | 97.54 | A | 1.0 | 1.0 | 1.0 | 1.0 | 0.954 | 0.6 | 1.0 |
| SouthGermanCredit | consistent_repr | 0.1 | 96.3 | A | 1.0 | 1.0 | 1.0 | 0.9235 | 0.945 | 0.6 | 1.0 |
| SouthGermanCredit | class_balance | 0.1 | 99.11 | A | 1.0 | 1.0 | 1.0 | 1.0 | 0.9611 | 0.9 | 1.0 |
| SouthGermanCredit | completeness | 0.25 | 91.54 | A | 0.75 | 1.0 | 1.0 | 1.0 | 0.9787 | 0.6 | 1.0 |
| SouthGermanCredit | uniqueness | 0.25 | 87.41 | B | 1.0 | 0.5 | 1.0 | 1.0 | 0.9405 | 0.6 | 1.0 |
| SouthGermanCredit | feature_accuracy | 0.25 | 97.67 | A | 1.0 | 1.0 | 1.0 | 1.0 | 0.967 | 0.6 | 1.0 |
| SouthGermanCredit | consistent_repr | 0.25 | 94.58 | A | 1.0 | 1.0 | 1.0 | 0.8088 | 0.945 | 0.6 | 1.0 |
| SouthGermanCredit | class_balance | 0.25 | 98.17 | A | 1.0 | 1.0 | 1.0 | 1.0 | 0.9422 | 0.75 | 1.0 |
| SouthGermanCredit | completeness | 0.5 | 85.14 | B | 0.5 | 1.0 | 1.0 | 1.0 | 0.9637 | 0.6 | 1.0 |
| SouthGermanCredit | uniqueness | 0.5 | 84.08 | B | 1.0 | 0.3333 | 1.0 | 1.0 | 0.9416 | 0.6 | 1.0 |
| SouthGermanCredit | feature_accuracy | 0.5 | 97.74 | A | 1.0 | 1.0 | 1.0 | 1.0 | 0.974 | 0.6 | 1.0 |
| SouthGermanCredit | consistent_repr | 0.5 | 91.71 | A | 1.0 | 1.0 | 1.0 | 0.6176 | 0.945 | 0.6 | 1.0 |
| SouthGermanCredit | class_balance | 0.5 | 96.95 | A | 1.0 | 1.0 | 1.0 | 1.0 | 0.945 | 0.5 | 1.0 |
| SouthGermanCredit | completeness | 0.75 | 76.63 | B | 0.25 | 0.994 | 1.0 | 1.0 | 0.75 | 0.6 | 1.0 |
| SouthGermanCredit | uniqueness | 0.75 | 82.48 | B | 1.0 | 0.25 | 1.0 | 1.0 | 0.9483 | 0.6 | 1.0 |
| SouthGermanCredit | feature_accuracy | 0.75 | 97.87 | A | 1.0 | 1.0 | 1.0 | 1.0 | 0.9873 | 0.6 | 1.0 |
| SouthGermanCredit | consistent_repr | 0.75 | 88.85 | B | 1.0 | 1.0 | 1.0 | 0.4265 | 0.945 | 0.6 | 1.0 |
| SouthGermanCredit | class_balance | 0.75 | 95.73 | A | 1.0 | 1.0 | 1.0 | 1.0 | 0.9483 | 0.25 | 1.0 |
| letter | completeness | 0.1 | 96.58 | A | 0.9 | 0.9959 | 1.0 | 1.0 | 0.9393 | 0.9542 | 1.0 |
| letter | uniqueness | 0.1 | 91.64 | A | 1.0 | 0.6666 | 1.0 | 1.0 | 0.9656 | 0.7297 | 1.0 |
| letter | feature_accuracy | 0.1 | 99.54 | A | 1.0 | 1.0 | 1.0 | 1.0 | 0.9772 | 0.9542 | 1.0 |
| letter | class_balance | 0.1 | 98.33 | A | 1.0 | 0.9563 | 1.0 | 1.0 | 0.9654 | 0.9093 | 1.0 |
| letter | completeness | 0.25 | 93.48 | A | 0.75 | 1.0 | 1.0 | 1.0 | 0.9959 | 0.9542 | 1.0 |
| letter | uniqueness | 0.25 | 88.31 | B | 1.0 | 0.5 | 1.0 | 1.0 | 0.966 | 0.7298 | 1.0 |
| letter | feature_accuracy | 0.25 | 99.64 | A | 1.0 | 1.0 | 1.0 | 1.0 | 0.9867 | 0.9542 | 1.0 |
| letter | class_balance | 0.25 | 97.42 | A | 1.0 | 0.9563 | 1.0 | 1.0 | 0.9523 | 0.7549 | 1.0 |
| letter | completeness | 0.5 | 87.25 | B | 0.5 | 1.0 | 1.0 | 1.0 | 0.9976 | 0.9542 | 1.0 |
| letter | uniqueness | 0.5 | 84.97 | B | 1.0 | 0.3333 | 1.0 | 1.0 | 0.9658 | 0.7298 | 1.0 |
| letter | feature_accuracy | 0.5 | 99.68 | A | 1.0 | 1.0 | 1.0 | 1.0 | 0.9906 | 0.9542 | 1.0 |
| letter | class_balance | 0.5 | 96.19 | A | 1.0 | 0.9564 | 1.0 | 1.0 | 0.9517 | 0.5098 | 1.0 |
| letter | completeness | 0.75 | 78.04 | B | 0.25 | 0.9757 | 1.0 | 1.0 | 0.75 | 0.9542 | 1.0 |
| letter | uniqueness | 0.75 | 83.31 | B | 1.0 | 0.25 | 1.0 | 1.0 | 0.9659 | 0.7298 | 1.0 |
| letter | feature_accuracy | 0.75 | 99.69 | A | 1.0 | 1.0 | 1.0 | 1.0 | 0.9918 | 0.9542 | 1.0 |
| letter | class_balance | 0.75 | 94.93 | A | 1.0 | 0.9535 | 1.0 | 1.0 | 0.9533 | 0.2647 | 1.0 |

## 3. 오염 강도별 DSC 변화 요약

### TelcoCustomerChurn

| 오염 유형 | 0%(baseline) | 10% | 25% | 50% | 75% | 최대 하락폭 |
|---|---|---|---|---|---|---|
| class_balance | 97.65 | 99.48 | 98.75 | 97.5 | 96.25 | 1.4 |
| completeness | 97.65 | 95.17 | 91.34 | 84.79 | 76.34 | 21.31 |
| consistent_repr | 97.65 | 96.25 | 94.14 | 90.62 | 87.11 | 10.54 |
| feature_accuracy | 97.65 | 97.65 | 97.65 | 97.62 | 97.61 | 0.04 |
| uniqueness | 97.65 | 90.99 | 87.65 | 84.32 | 82.65 | 15.0 |

### SouthGermanCredit

| 오염 유형 | 0%(baseline) | 10% | 25% | 50% | 75% | 최대 하락폭 |
|---|---|---|---|---|---|---|
| class_balance | 97.45 | 99.11 | 98.17 | 96.95 | 95.73 | 1.72 |
| completeness | 97.45 | 94.6 | 91.54 | 85.14 | 76.63 | 20.82 |
| consistent_repr | 97.45 | 96.3 | 94.58 | 91.71 | 88.85 | 8.6 |
| feature_accuracy | 97.45 | 97.54 | 97.67 | 97.74 | 97.87 | -0.09 |
| uniqueness | 97.45 | 90.76 | 87.41 | 84.08 | 82.48 | 14.97 |

### letter

| 오염 유형 | 0%(baseline) | 10% | 25% | 50% | 75% | 최대 하락폭 |
|---|---|---|---|---|---|---|
| class_balance | 98.11 | 98.33 | 97.42 | 96.19 | 94.93 | 3.18 |
| completeness | 98.11 | 96.58 | 93.48 | 87.25 | 78.04 | 20.07 |
| feature_accuracy | 98.11 | 99.54 | 99.64 | 99.68 | 99.69 | -1.43 |
| uniqueness | 98.11 | 91.64 | 88.31 | 84.97 | 83.31 | 14.8 |

## 5. 자동 감지된 문제점

- ⚠️ **TelcoCustomerChurn/feature_accuracy_75%**: DSC 하락 0.0점 (5점 미만 — 오염 감지 부족)
- ⚠️ **TelcoCustomerChurn/class_balance_75%**: DSC 하락 1.4점 (5점 미만 — 오염 감지 부족)
- ⚠️ **SouthGermanCredit/feature_accuracy_75%**: DSC 하락 -0.4점 (5점 미만 — 오염 감지 부족)
- ⚠️ **SouthGermanCredit/class_balance_75%**: DSC 하락 1.7점 (5점 미만 — 오염 감지 부족)
- ⚠️ **letter/feature_accuracy_75%**: DSC 하락 -1.6점 (5점 미만 — 오염 감지 부족)
- ⚠️ **letter/class_balance_75%**: DSC 하락 3.2점 (5점 미만 — 오염 감지 부족)

## 6. 산출물

- `dsc_scores.csv` — DSC 점수 59건
- `data/polluted/` — 오염 데이터 파일
- `02_execution_log.md` — 이 로그 파일

---
*이 로그는 노트북 02 실행 시 자동 생성됨*