# 노트북 02 실행 로그: 오염 생성 & DSC 점수 계산

- **실행 시각**: 2026-04-25 06:46:42
- **총 실험**: 87건 (베이스라인 포함)
- **소요 시간**: 132초
- **에러**: 0건

## 1. 오염 설정

- **데이터셋**: ['TelcoCustomerChurn', 'SouthGermanCredit', 'letter']
- **오염 강도**: [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
- **Polluter**: CompletenessPolluter, UniquenessPolluter, FeatureAccuracyPolluter, ConsistentRepresentationPolluter, ClassBalancePolluter

## 2. DSC 점수 결과

| 데이터셋 | 오염 유형 | 강도 | DSC Score | 등급 | completeness | uniqueness | validity | consistency | outlier_ratio | class_balance | feature_corr |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TelcoCustomerChurn | none | 0.0 | 95.31 | A | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.5307 | 1.0 |
| TelcoCustomerChurn | completeness | 0.1 | 90.41 | A | 0.9051 | 1.0 | 1.0 | 1.0 | 1.0 | 0.5307 | 1.0 |
| TelcoCustomerChurn | uniqueness | 0.1 | 90.16 | A | 1.0 | 0.6666 | 1.0 | 1.0 | 1.0 | 0.5308 | 1.0 |
| TelcoCustomerChurn | feature_accuracy | 0.1 | 94.28 | A | 1.0 | 1.0 | 1.0 | 1.0 | 0.9999 | 0.5307 | 1.0 |
| TelcoCustomerChurn | consistent_repr | 0.1 | 91.78 | A | 1.0 | 1.0 | 1.0 | 0.9001 | 1.0 | 0.5307 | 1.0 |
| TelcoCustomerChurn | class_balance | 0.1 | 97.67 | A | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.9003 | 1.0 |
| TelcoCustomerChurn | completeness | 0.25 | 83.06 | B | 0.7626 | 1.0 | 1.0 | 1.0 | 1.0 | 0.5307 | 1.0 |
| TelcoCustomerChurn | uniqueness | 0.25 | 87.68 | B | 1.0 | 0.5 | 1.0 | 1.0 | 1.0 | 0.5307 | 1.0 |
| TelcoCustomerChurn | feature_accuracy | 0.25 | 93.0 | A | 1.0 | 1.0 | 1.0 | 1.0 | 0.9992 | 0.5307 | 1.0 |
| TelcoCustomerChurn | consistent_repr | 0.25 | 86.49 | B | 1.0 | 1.0 | 1.0 | 0.7501 | 1.0 | 0.5307 | 1.0 |
| TelcoCustomerChurn | class_balance | 0.25 | 96.71 | A | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.7505 | 1.0 |
| TelcoCustomerChurn | completeness | 0.5 | 70.81 | C | 0.525 | 1.0 | 1.0 | 1.0 | 1.0 | 0.5307 | 1.0 |
| TelcoCustomerChurn | uniqueness | 0.5 | 85.14 | B | 1.0 | 0.3333 | 1.0 | 1.0 | 1.0 | 0.5307 | 1.0 |
| TelcoCustomerChurn | feature_accuracy | 0.5 | 91.15 | A | 1.0 | 1.0 | 1.0 | 1.0 | 0.9945 | 0.5307 | 1.0 |
| TelcoCustomerChurn | consistent_repr | 0.5 | 77.68 | B | 1.0 | 1.0 | 1.0 | 0.5 | 1.0 | 0.5307 | 1.0 |
| TelcoCustomerChurn | class_balance | 0.5 | 94.76 | A | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.5003 | 1.0 |
| TelcoCustomerChurn | completeness | 0.75 | 58.26 | D | 0.2876 | 0.9798 | 1.0 | 1.0 | 1.0 | 0.5307 | 1.0 |
| TelcoCustomerChurn | uniqueness | 0.75 | 83.93 | B | 1.0 | 0.25 | 1.0 | 1.0 | 1.0 | 0.5307 | 1.0 |
| TelcoCustomerChurn | feature_accuracy | 0.75 | 89.04 | B | 1.0 | 1.0 | 1.0 | 1.0 | 0.9718 | 0.5307 | 1.0 |
| TelcoCustomerChurn | consistent_repr | 0.75 | 68.86 | C | 1.0 | 1.0 | 1.0 | 0.2501 | 1.0 | 0.5307 | 1.0 |
| TelcoCustomerChurn | class_balance | 0.75 | 91.42 | A | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.2502 | 1.0 |
| TelcoCustomerChurn | completeness | 0.9 | 45.23 | D | 0.1451 | 0.6013 | 1.0 | 1.0 | 1.0 | 0.5307 | 1.0 |
| TelcoCustomerChurn | uniqueness | 0.9 | 83.2 | B | 1.0 | 0.2 | 1.0 | 1.0 | 1.0 | 0.5307 | 1.0 |
| TelcoCustomerChurn | feature_accuracy | 0.9 | 87.88 | B | 1.0 | 1.0 | 1.0 | 1.0 | 0.9515 | 0.5307 | 1.0 |
| TelcoCustomerChurn | consistent_repr | 0.9 | 63.57 | C | 1.0 | 1.0 | 1.0 | 0.1001 | 1.0 | 0.5307 | 1.0 |
| TelcoCustomerChurn | class_balance | 0.9 | 89.44 | B | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.1003 | 1.0 |
| TelcoCustomerChurn | completeness | 0.95 | 38.49 | D | 0.0976 | 0.3151 | 1.0 | 1.0 | 1.0 | 0.5307 | 1.0 |
| TelcoCustomerChurn | uniqueness | 0.95 | 82.71 | B | 1.0 | 0.1667 | 1.0 | 1.0 | 1.0 | 0.5307 | 1.0 |
| TelcoCustomerChurn | feature_accuracy | 0.95 | 87.45 | B | 1.0 | 1.0 | 1.0 | 1.0 | 0.9441 | 0.5307 | 1.0 |
| TelcoCustomerChurn | consistent_repr | 0.95 | 61.81 | C | 1.0 | 1.0 | 1.0 | 0.0501 | 1.0 | 0.5307 | 1.0 |
| TelcoCustomerChurn | class_balance | 0.95 | 88.76 | B | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0502 | 1.0 |
| SouthGermanCredit | none | 0.0 | 95.7 | A | 1.0 | 1.0 | 1.0 | 1.0 | 0.9408 | 0.6 | 1.0 |
| SouthGermanCredit | completeness | 0.1 | 90.56 | A | 0.9 | 1.0 | 1.0 | 1.0 | 0.9121 | 0.6 | 1.0 |
| SouthGermanCredit | uniqueness | 0.1 | 90.4 | A | 1.0 | 0.6667 | 1.0 | 1.0 | 0.9389 | 0.6 | 1.0 |
| SouthGermanCredit | feature_accuracy | 0.1 | 94.1 | A | 1.0 | 1.0 | 1.0 | 1.0 | 0.9375 | 0.6 | 1.0 |
| SouthGermanCredit | consistent_repr | 0.1 | 92.15 | A | 1.0 | 1.0 | 1.0 | 0.9 | 0.9408 | 0.6 | 1.0 |
| SouthGermanCredit | class_balance | 0.1 | 97.88 | A | 1.0 | 1.0 | 1.0 | 1.0 | 0.9313 | 0.9 | 1.0 |
| SouthGermanCredit | completeness | 0.25 | 82.86 | B | 0.75 | 1.0 | 1.0 | 1.0 | 0.8721 | 0.6 | 1.0 |
| SouthGermanCredit | uniqueness | 0.25 | 87.86 | B | 1.0 | 0.5 | 1.0 | 1.0 | 0.9446 | 0.6 | 1.0 |
| SouthGermanCredit | feature_accuracy | 0.25 | 92.01 | A | 1.0 | 1.0 | 1.0 | 1.0 | 0.93 | 0.6 | 1.0 |
| SouthGermanCredit | consistent_repr | 0.25 | 86.83 | B | 1.0 | 1.0 | 1.0 | 0.75 | 0.9408 | 0.6 | 1.0 |
| SouthGermanCredit | class_balance | 0.25 | 96.57 | A | 1.0 | 1.0 | 1.0 | 1.0 | 0.9368 | 0.75 | 1.0 |
| SouthGermanCredit | completeness | 0.5 | 70.04 | C | 0.5 | 1.0 | 1.0 | 1.0 | 0.8071 | 0.6 | 1.0 |
| SouthGermanCredit | uniqueness | 0.5 | 85.44 | B | 1.0 | 0.3333 | 1.0 | 1.0 | 0.9426 | 0.6 | 1.0 |
| SouthGermanCredit | feature_accuracy | 0.5 | 88.08 | B | 1.0 | 1.0 | 1.0 | 1.0 | 0.8633 | 0.6 | 1.0 |
| SouthGermanCredit | consistent_repr | 0.5 | 77.95 | B | 1.0 | 1.0 | 1.0 | 0.5 | 0.9408 | 0.6 | 1.0 |
| SouthGermanCredit | class_balance | 0.5 | 94.3 | A | 1.0 | 1.0 | 1.0 | 1.0 | 0.9535 | 0.5 | 1.0 |
| SouthGermanCredit | completeness | 0.75 | 57.09 | D | 0.25 | 0.9938 | 1.0 | 1.0 | 0.7367 | 0.6 | 1.0 |
| SouthGermanCredit | uniqueness | 0.75 | 84.19 | B | 1.0 | 0.25 | 1.0 | 1.0 | 0.9396 | 0.6 | 1.0 |
| SouthGermanCredit | feature_accuracy | 0.75 | 84.49 | B | 1.0 | 1.0 | 1.0 | 1.0 | 0.7671 | 0.6 | 1.0 |
| SouthGermanCredit | consistent_repr | 0.75 | 69.08 | C | 1.0 | 1.0 | 1.0 | 0.25 | 0.9408 | 0.6 | 1.0 |
| SouthGermanCredit | class_balance | 0.75 | 91.56 | A | 1.0 | 1.0 | 1.0 | 1.0 | 0.9576 | 0.25 | 1.0 |
| SouthGermanCredit | completeness | 0.9 | 45.25 | D | 0.1 | 0.7188 | 1.0 | 1.0 | 0.6938 | 0.6 | 1.0 |
| SouthGermanCredit | uniqueness | 0.9 | 83.44 | B | 1.0 | 0.2 | 1.0 | 1.0 | 0.9392 | 0.6 | 1.0 |
| SouthGermanCredit | feature_accuracy | 0.9 | 82.25 | B | 1.0 | 1.0 | 1.0 | 1.0 | 0.7188 | 0.6 | 1.0 |
| SouthGermanCredit | consistent_repr | 0.9 | 63.75 | C | 1.0 | 1.0 | 1.0 | 0.1 | 0.9408 | 0.6 | 1.0 |
| SouthGermanCredit | class_balance | 0.9 | 89.87 | B | 1.0 | 1.0 | 1.0 | 1.0 | 0.9576 | 0.1 | 1.0 |
| SouthGermanCredit | completeness | 0.95 | 38.09 | D | 0.05 | 0.4125 | 1.0 | 1.0 | 0.68 | 0.6 | 1.0 |
| SouthGermanCredit | uniqueness | 0.95 | 82.98 | B | 1.0 | 0.1667 | 1.0 | 1.0 | 0.9398 | 0.6 | 1.0 |
| SouthGermanCredit | feature_accuracy | 0.95 | 81.55 | B | 1.0 | 1.0 | 1.0 | 1.0 | 0.6917 | 0.6 | 1.0 |
| SouthGermanCredit | consistent_repr | 0.95 | 61.98 | C | 1.0 | 1.0 | 1.0 | 0.05 | 0.9408 | 0.6 | 1.0 |
| SouthGermanCredit | class_balance | 0.95 | 89.28 | B | 1.0 | 1.0 | 1.0 | 1.0 | 0.9576 | 0.05 | 1.0 |
| letter | none | 0.0 | 98.41 | A | 1.0 | 0.9401 | 1.0 | 1.0 | 0.9543 | 0.9539 | 1.0 |
| letter | completeness | 0.1 | 93.93 | A | 0.9 | 0.996 | 1.0 | 1.0 | 0.89 | 0.9539 | 1.0 |
| letter | uniqueness | 0.1 | 91.59 | A | 1.0 | 0.6665 | 1.0 | 1.0 | 0.9544 | 0.7351 | 1.0 |
| letter | feature_accuracy | 0.1 | 95.44 | A | 1.0 | 1.0 | 1.0 | 1.0 | 0.9396 | 0.9539 | 1.0 |
| letter | class_balance | 0.1 | 98.16 | A | 1.0 | 0.9622 | 1.0 | 1.0 | 0.9549 | 0.9183 | 1.0 |
| letter | completeness | 0.25 | 86.01 | B | 0.75 | 0.9999 | 1.0 | 1.0 | 0.7939 | 0.9539 | 1.0 |
| letter | uniqueness | 0.25 | 89.13 | B | 1.0 | 0.5 | 1.0 | 1.0 | 0.9543 | 0.7346 | 1.0 |
| letter | feature_accuracy | 0.25 | 94.74 | A | 1.0 | 1.0 | 1.0 | 1.0 | 0.895 | 0.9539 | 1.0 |
| letter | class_balance | 0.25 | 96.46 | A | 1.0 | 0.9613 | 1.0 | 1.0 | 0.9545 | 0.7549 | 1.0 |
| letter | completeness | 0.5 | 72.71 | C | 0.5 | 1.0 | 1.0 | 1.0 | 0.6336 | 0.9539 | 1.0 |
| letter | uniqueness | 0.5 | 86.63 | B | 1.0 | 0.3333 | 1.0 | 1.0 | 0.9543 | 0.7346 | 1.0 |
| letter | feature_accuracy | 0.5 | 92.68 | A | 1.0 | 1.0 | 1.0 | 1.0 | 0.7897 | 0.9539 | 1.0 |
| letter | class_balance | 0.5 | 93.86 | A | 1.0 | 0.9613 | 1.0 | 1.0 | 0.9537 | 0.5098 | 1.0 |
| letter | completeness | 0.75 | 59.06 | D | 0.25 | 0.9772 | 1.0 | 1.0 | 0.4728 | 0.9539 | 1.0 |
| letter | uniqueness | 0.75 | 85.36 | B | 1.0 | 0.25 | 1.0 | 1.0 | 0.9541 | 0.7346 | 1.0 |
| letter | feature_accuracy | 0.75 | 90.74 | A | 1.0 | 1.0 | 1.0 | 1.0 | 0.6812 | 0.9539 | 1.0 |
| letter | class_balance | 0.75 | 91.16 | A | 1.0 | 0.9589 | 1.0 | 1.0 | 0.9531 | 0.2647 | 1.0 |
| letter | completeness | 0.9 | 45.61 | D | 0.1 | 0.6126 | 1.0 | 1.0 | 0.3764 | 0.9539 | 1.0 |
| letter | uniqueness | 0.9 | 84.6 | B | 1.0 | 0.2 | 1.0 | 1.0 | 0.9545 | 0.7346 | 1.0 |
| letter | feature_accuracy | 0.9 | 89.77 | B | 1.0 | 1.0 | 1.0 | 1.0 | 0.6231 | 0.9539 | 1.0 |
| letter | class_balance | 0.9 | 89.38 | B | 1.0 | 0.9574 | 1.0 | 1.0 | 0.9529 | 0.1013 | 1.0 |
| letter | completeness | 0.95 | 38.7 | D | 0.05 | 0.3291 | 1.0 | 1.0 | 0.3445 | 0.9539 | 1.0 |
| letter | uniqueness | 0.95 | 84.11 | B | 1.0 | 0.1667 | 1.0 | 1.0 | 0.9546 | 0.7346 | 1.0 |
| letter | feature_accuracy | 0.95 | 89.49 | B | 1.0 | 1.0 | 1.0 | 1.0 | 0.6048 | 0.9539 | 1.0 |
| letter | class_balance | 0.95 | 88.95 | B | 1.0 | 0.9569 | 1.0 | 1.0 | 0.9529 | 0.0621 | 1.0 |

## 3. 오염 강도별 DSC 변화 요약

### TelcoCustomerChurn

| 오염 유형 | 0%(baseline) | 10% | 25% | 50% | 75% | 최대 하락폭 |
|---|---|---|---|---|---|---|
| class_balance | 95.31 | 97.67 | 96.71 | 94.76 | 91.42 | 3.89 |
| completeness | 95.31 | 90.41 | 83.06 | 70.81 | 58.26 | 37.05 |
| consistent_repr | 95.31 | 91.78 | 86.49 | 77.68 | 68.86 | 26.45 |
| feature_accuracy | 95.31 | 94.28 | 93.0 | 91.15 | 89.04 | 6.27 |
| uniqueness | 95.31 | 90.16 | 87.68 | 85.14 | 83.93 | 11.38 |

### SouthGermanCredit

| 오염 유형 | 0%(baseline) | 10% | 25% | 50% | 75% | 최대 하락폭 |
|---|---|---|---|---|---|---|
| class_balance | 95.7 | 97.88 | 96.57 | 94.3 | 91.56 | 4.14 |
| completeness | 95.7 | 90.56 | 82.86 | 70.04 | 57.09 | 38.61 |
| consistent_repr | 95.7 | 92.15 | 86.83 | 77.95 | 69.08 | 26.62 |
| feature_accuracy | 95.7 | 94.1 | 92.01 | 88.08 | 84.49 | 11.21 |
| uniqueness | 95.7 | 90.4 | 87.86 | 85.44 | 84.19 | 11.51 |

### letter

| 오염 유형 | 0%(baseline) | 10% | 25% | 50% | 75% | 최대 하락폭 |
|---|---|---|---|---|---|---|
| class_balance | 98.41 | 98.16 | 96.46 | 93.86 | 91.16 | 7.25 |
| completeness | 98.41 | 93.93 | 86.01 | 72.71 | 59.06 | 39.35 |
| feature_accuracy | 98.41 | 95.44 | 94.74 | 92.68 | 90.74 | 7.67 |
| uniqueness | 98.41 | 91.59 | 89.13 | 86.63 | 85.36 | 13.05 |

## 4. 에러 목록

_에러 없음_

## 5. 자동 감지된 문제점

- ⚠️ **TelcoCustomerChurn/class_balance_75%**: DSC 하락 3.9점 (5점 미만 — 감지 부족)
- ⚠️ **SouthGermanCredit/class_balance_75%**: DSC 하락 4.1점 (5점 미만 — 감지 부족)

## 6. 산출물

- `dsc_scores.csv` — DSC 점수 87건
- `data/polluted/` — 오염 데이터 파일
- `02_execution_log.md` — 이 로그 파일

---
*이 로그는 노트북 02 실행 시 자동 생성됨*