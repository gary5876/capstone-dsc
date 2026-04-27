# 노트북 02 실행 로그: 오염 생성 & DSC 점수 계산

- **실행 시각**: 2026-04-27 07:35:50
- **총 실험**: 87건 (베이스라인 포함)
- **소요 시간**: 214초
- **에러**: 0건

## 1. 오염 설정

- **데이터셋**: ['TelcoCustomerChurn', 'SouthGermanCredit', 'letter']
- **오염 강도**: [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
- **Polluter**: CompletenessPolluter, UniquenessPolluter, FeatureAccuracyPolluter, ConsistentRepresentationPolluter, ClassBalancePolluter

## 2. DSC 점수 결과

| 데이터셋 | 오염 유형 | 강도 | DSC Score | 등급 | completeness | uniqueness | validity | consistency | outlier_ratio | class_balance | feature_corr |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TelcoCustomerChurn | none | 0.0 | 80.75 | B | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.5307 | 1.0 |
| TelcoCustomerChurn | completeness | 0.1 | 78.46 | B | 0.9051 | 1.0 | 1.0 | 1.0 | 1.0 | 0.5307 | 1.0 |
| TelcoCustomerChurn | uniqueness | 0.1 | 75.08 | B | 1.0 | 0.6666 | 1.0 | 1.0 | 1.0 | 0.5308 | 1.0 |
| TelcoCustomerChurn | feature_accuracy | 0.1 | 79.78 | B | 1.0 | 1.0 | 1.0 | 1.0 | 0.9999 | 0.5307 | 1.0 |
| TelcoCustomerChurn | consistent_repr | 0.1 | 79.75 | B | 1.0 | 1.0 | 1.0 | 0.9001 | 1.0 | 0.5307 | 1.0 |
| TelcoCustomerChurn | class_balance | 0.1 | 85.57 | B | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.9003 | 1.0 |
| TelcoCustomerChurn | completeness | 0.25 | 74.55 | C | 0.7626 | 1.0 | 1.0 | 1.0 | 1.0 | 0.5307 | 1.0 |
| TelcoCustomerChurn | uniqueness | 0.25 | 73.16 | C | 1.0 | 0.5 | 1.0 | 1.0 | 1.0 | 0.5307 | 1.0 |
| TelcoCustomerChurn | feature_accuracy | 0.25 | 74.78 | C | 1.0 | 1.0 | 1.0 | 1.0 | 0.9992 | 0.5307 | 1.0 |
| TelcoCustomerChurn | consistent_repr | 0.25 | 78.25 | B | 1.0 | 1.0 | 1.0 | 0.7501 | 1.0 | 0.5307 | 1.0 |
| TelcoCustomerChurn | class_balance | 0.25 | 84.23 | B | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.7505 | 1.0 |
| TelcoCustomerChurn | completeness | 0.5 | 66.91 | C | 0.525 | 1.0 | 1.0 | 1.0 | 1.0 | 0.5307 | 1.0 |
| TelcoCustomerChurn | uniqueness | 0.5 | 71.37 | C | 1.0 | 0.3333 | 1.0 | 1.0 | 1.0 | 0.5307 | 1.0 |
| TelcoCustomerChurn | feature_accuracy | 0.5 | 70.4 | C | 1.0 | 1.0 | 1.0 | 1.0 | 0.9945 | 0.5307 | 1.0 |
| TelcoCustomerChurn | consistent_repr | 0.5 | 75.75 | B | 1.0 | 1.0 | 1.0 | 0.5 | 1.0 | 0.5307 | 1.0 |
| TelcoCustomerChurn | class_balance | 0.5 | 80.5 | B | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.5003 | 1.0 |
| TelcoCustomerChurn | completeness | 0.75 | 57.22 | D | 0.2876 | 0.9798 | 1.0 | 1.0 | 1.0 | 0.5307 | 1.0 |
| TelcoCustomerChurn | uniqueness | 0.75 | 69.53 | C | 1.0 | 0.25 | 1.0 | 1.0 | 1.0 | 0.5307 | 1.0 |
| TelcoCustomerChurn | feature_accuracy | 0.75 | 68.55 | C | 1.0 | 1.0 | 1.0 | 1.0 | 0.9718 | 0.5307 | 1.0 |
| TelcoCustomerChurn | consistent_repr | 0.75 | 73.25 | C | 1.0 | 1.0 | 1.0 | 0.2501 | 1.0 | 0.5307 | 1.0 |
| TelcoCustomerChurn | class_balance | 0.75 | 76.39 | B | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.2502 | 1.0 |
| TelcoCustomerChurn | completeness | 0.9 | 45.9 | D | 0.1451 | 0.6013 | 1.0 | 1.0 | 1.0 | 0.5307 | 1.0 |
| TelcoCustomerChurn | uniqueness | 0.9 | 69.2 | C | 1.0 | 0.2 | 1.0 | 1.0 | 1.0 | 0.5307 | 1.0 |
| TelcoCustomerChurn | feature_accuracy | 0.9 | 69.07 | C | 1.0 | 1.0 | 1.0 | 1.0 | 0.9515 | 0.5307 | 1.0 |
| TelcoCustomerChurn | consistent_repr | 0.9 | 71.75 | C | 1.0 | 1.0 | 1.0 | 0.1001 | 1.0 | 0.5307 | 1.0 |
| TelcoCustomerChurn | class_balance | 0.9 | 72.37 | C | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.1003 | 1.0 |
| TelcoCustomerChurn | completeness | 0.95 | 39.86 | D | 0.0976 | 0.3151 | 1.0 | 1.0 | 1.0 | 0.5307 | 1.0 |
| TelcoCustomerChurn | uniqueness | 0.95 | 67.72 | C | 1.0 | 0.1667 | 1.0 | 1.0 | 1.0 | 0.5307 | 1.0 |
| TelcoCustomerChurn | feature_accuracy | 0.95 | 69.29 | C | 1.0 | 1.0 | 1.0 | 1.0 | 0.9441 | 0.5307 | 1.0 |
| TelcoCustomerChurn | consistent_repr | 0.95 | 71.25 | C | 1.0 | 1.0 | 1.0 | 0.0501 | 1.0 | 0.5307 | 1.0 |
| TelcoCustomerChurn | class_balance | 0.95 | 70.76 | C | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0502 | 1.0 |
| SouthGermanCredit | none | 0.0 | 71.44 | C | 1.0 | 1.0 | 1.0 | 1.0 | 0.9408 | 0.6 | 1.0 |
| SouthGermanCredit | completeness | 0.1 | 68.76 | C | 0.9 | 1.0 | 1.0 | 1.0 | 0.9121 | 0.6 | 1.0 |
| SouthGermanCredit | uniqueness | 0.1 | 65.97 | C | 1.0 | 0.6667 | 1.0 | 1.0 | 0.9389 | 0.6 | 1.0 |
| SouthGermanCredit | feature_accuracy | 0.1 | 70.09 | C | 1.0 | 1.0 | 1.0 | 1.0 | 0.9375 | 0.6 | 1.0 |
| SouthGermanCredit | consistent_repr | 0.1 | 71.16 | C | 1.0 | 1.0 | 1.0 | 0.9 | 0.9408 | 0.6 | 1.0 |
| SouthGermanCredit | class_balance | 0.1 | 75.09 | B | 1.0 | 1.0 | 1.0 | 1.0 | 0.9313 | 0.9 | 1.0 |
| SouthGermanCredit | completeness | 0.25 | 64.74 | C | 0.75 | 1.0 | 1.0 | 1.0 | 0.8721 | 0.6 | 1.0 |
| SouthGermanCredit | uniqueness | 0.25 | 63.69 | C | 1.0 | 0.5 | 1.0 | 1.0 | 0.9446 | 0.6 | 1.0 |
| SouthGermanCredit | feature_accuracy | 0.25 | 69.0 | C | 1.0 | 1.0 | 1.0 | 1.0 | 0.93 | 0.6 | 1.0 |
| SouthGermanCredit | consistent_repr | 0.25 | 69.45 | C | 1.0 | 1.0 | 1.0 | 0.75 | 0.9408 | 0.6 | 1.0 |
| SouthGermanCredit | class_balance | 0.25 | 73.62 | C | 1.0 | 1.0 | 1.0 | 1.0 | 0.9368 | 0.75 | 1.0 |
| SouthGermanCredit | completeness | 0.5 | 58.9 | D | 0.5 | 1.0 | 1.0 | 1.0 | 0.8071 | 0.6 | 1.0 |
| SouthGermanCredit | uniqueness | 0.5 | 60.77 | C | 1.0 | 0.3333 | 1.0 | 1.0 | 0.9426 | 0.6 | 1.0 |
| SouthGermanCredit | feature_accuracy | 0.5 | 67.44 | C | 1.0 | 1.0 | 1.0 | 1.0 | 0.8633 | 0.6 | 1.0 |
| SouthGermanCredit | consistent_repr | 0.5 | 67.21 | C | 1.0 | 1.0 | 1.0 | 0.5 | 0.9408 | 0.6 | 1.0 |
| SouthGermanCredit | class_balance | 0.5 | 70.01 | C | 1.0 | 1.0 | 1.0 | 1.0 | 0.9535 | 0.5 | 1.0 |
| SouthGermanCredit | completeness | 0.75 | 51.65 | D | 0.25 | 0.9938 | 1.0 | 1.0 | 0.7367 | 0.6 | 1.0 |
| SouthGermanCredit | uniqueness | 0.75 | 60.3 | C | 1.0 | 0.25 | 1.0 | 1.0 | 0.9396 | 0.6 | 1.0 |
| SouthGermanCredit | feature_accuracy | 0.75 | 65.98 | C | 1.0 | 1.0 | 1.0 | 1.0 | 0.7671 | 0.6 | 1.0 |
| SouthGermanCredit | consistent_repr | 0.75 | 64.7 | C | 1.0 | 1.0 | 1.0 | 0.25 | 0.9408 | 0.6 | 1.0 |
| SouthGermanCredit | class_balance | 0.75 | 68.39 | C | 1.0 | 1.0 | 1.0 | 1.0 | 0.9576 | 0.25 | 1.0 |
| SouthGermanCredit | completeness | 0.9 | 44.27 | D | 0.1 | 0.7188 | 1.0 | 1.0 | 0.6938 | 0.6 | 1.0 |
| SouthGermanCredit | uniqueness | 0.9 | 58.95 | D | 1.0 | 0.2 | 1.0 | 1.0 | 0.9392 | 0.6 | 1.0 |
| SouthGermanCredit | feature_accuracy | 0.9 | 66.12 | C | 1.0 | 1.0 | 1.0 | 1.0 | 0.7188 | 0.6 | 1.0 |
| SouthGermanCredit | consistent_repr | 0.9 | 62.98 | C | 1.0 | 1.0 | 1.0 | 0.1 | 0.9408 | 0.6 | 1.0 |
| SouthGermanCredit | class_balance | 0.9 | 66.26 | C | 1.0 | 1.0 | 1.0 | 1.0 | 0.9576 | 0.1 | 1.0 |
| SouthGermanCredit | completeness | 0.95 | 38.46 | D | 0.05 | 0.4125 | 1.0 | 1.0 | 0.68 | 0.6 | 1.0 |
| SouthGermanCredit | uniqueness | 0.95 | 58.69 | D | 1.0 | 0.1667 | 1.0 | 1.0 | 0.9398 | 0.6 | 1.0 |
| SouthGermanCredit | feature_accuracy | 0.95 | 65.37 | C | 1.0 | 1.0 | 1.0 | 1.0 | 0.6917 | 0.6 | 1.0 |
| SouthGermanCredit | consistent_repr | 0.95 | 62.68 | C | 1.0 | 1.0 | 1.0 | 0.05 | 0.9408 | 0.6 | 1.0 |
| SouthGermanCredit | class_balance | 0.95 | 65.82 | C | 1.0 | 1.0 | 1.0 | 1.0 | 0.9576 | 0.05 | 1.0 |
| letter | none | 0.0 | 91.59 | A | 1.0 | 0.9401 | 1.0 | 1.0 | 0.9543 | 0.9539 | 1.0 |
| letter | completeness | 0.1 | 82.24 | B | 0.9 | 0.996 | 1.0 | 1.0 | 0.89 | 0.9539 | 1.0 |
| letter | uniqueness | 0.1 | 85.9 | B | 1.0 | 0.6665 | 1.0 | 1.0 | 0.9544 | 0.7351 | 1.0 |
| letter | feature_accuracy | 0.1 | 90.89 | A | 1.0 | 1.0 | 1.0 | 1.0 | 0.9396 | 0.9539 | 1.0 |
| letter | class_balance | 0.1 | 92.2 | A | 1.0 | 0.9622 | 1.0 | 1.0 | 0.9549 | 0.9183 | 1.0 |
| letter | completeness | 0.25 | 75.94 | B | 0.75 | 0.9999 | 1.0 | 1.0 | 0.7939 | 0.9539 | 1.0 |
| letter | uniqueness | 0.25 | 83.34 | B | 1.0 | 0.5 | 1.0 | 1.0 | 0.9543 | 0.7346 | 1.0 |
| letter | feature_accuracy | 0.25 | 83.53 | B | 1.0 | 1.0 | 1.0 | 1.0 | 0.895 | 0.9539 | 1.0 |
| letter | class_balance | 0.25 | 90.53 | A | 1.0 | 0.9613 | 1.0 | 1.0 | 0.9545 | 0.7549 | 1.0 |
| letter | completeness | 0.5 | 67.14 | C | 0.5 | 1.0 | 1.0 | 1.0 | 0.6336 | 0.9539 | 1.0 |
| letter | uniqueness | 0.5 | 80.89 | B | 1.0 | 0.3333 | 1.0 | 1.0 | 0.9543 | 0.7346 | 1.0 |
| letter | feature_accuracy | 0.5 | 74.48 | C | 1.0 | 1.0 | 1.0 | 1.0 | 0.7897 | 0.9539 | 1.0 |
| letter | class_balance | 0.5 | 88.2 | B | 1.0 | 0.9613 | 1.0 | 1.0 | 0.9537 | 0.5098 | 1.0 |
| letter | completeness | 0.75 | 56.86 | D | 0.25 | 0.9772 | 1.0 | 1.0 | 0.4728 | 0.9539 | 1.0 |
| letter | uniqueness | 0.75 | 79.2 | B | 1.0 | 0.25 | 1.0 | 1.0 | 0.9541 | 0.7346 | 1.0 |
| letter | feature_accuracy | 0.75 | 71.28 | C | 1.0 | 1.0 | 1.0 | 1.0 | 0.6812 | 0.9539 | 1.0 |
| letter | class_balance | 0.75 | 86.12 | B | 1.0 | 0.9589 | 1.0 | 1.0 | 0.9531 | 0.2647 | 1.0 |
| letter | completeness | 0.9 | 45.73 | D | 0.1 | 0.6126 | 1.0 | 1.0 | 0.3764 | 0.9539 | 1.0 |
| letter | uniqueness | 0.9 | 78.47 | B | 1.0 | 0.2 | 1.0 | 1.0 | 0.9545 | 0.7346 | 1.0 |
| letter | feature_accuracy | 0.9 | 70.16 | C | 1.0 | 1.0 | 1.0 | 1.0 | 0.6231 | 0.9539 | 1.0 |
| letter | class_balance | 0.9 | 84.74 | B | 1.0 | 0.9574 | 1.0 | 1.0 | 0.9529 | 0.1013 | 1.0 |
| letter | completeness | 0.95 | 40.01 | D | 0.05 | 0.3291 | 1.0 | 1.0 | 0.3445 | 0.9539 | 1.0 |
| letter | uniqueness | 0.95 | 78.07 | B | 1.0 | 0.1667 | 1.0 | 1.0 | 0.9546 | 0.7346 | 1.0 |
| letter | feature_accuracy | 0.95 | 69.91 | C | 1.0 | 1.0 | 1.0 | 1.0 | 0.6048 | 0.9539 | 1.0 |
| letter | class_balance | 0.95 | 84.71 | B | 1.0 | 0.9569 | 1.0 | 1.0 | 0.9529 | 0.0621 | 1.0 |

## 3. 오염 강도별 DSC 변화 요약

### TelcoCustomerChurn

| 오염 유형 | 0%(baseline) | 10% | 25% | 50% | 75% | 최대 하락폭 |
|---|---|---|---|---|---|---|
| class_balance | 80.75 | 85.57 | 84.23 | 80.5 | 76.39 | 4.36 |
| completeness | 80.75 | 78.46 | 74.55 | 66.91 | 57.22 | 23.53 |
| consistent_repr | 80.75 | 79.75 | 78.25 | 75.75 | 73.25 | 7.5 |
| feature_accuracy | 80.75 | 79.78 | 74.78 | 70.4 | 68.55 | 12.2 |
| uniqueness | 80.75 | 75.08 | 73.16 | 71.37 | 69.53 | 11.22 |

### SouthGermanCredit

| 오염 유형 | 0%(baseline) | 10% | 25% | 50% | 75% | 최대 하락폭 |
|---|---|---|---|---|---|---|
| class_balance | 71.44 | 75.09 | 73.62 | 70.01 | 68.39 | 3.05 |
| completeness | 71.44 | 68.76 | 64.74 | 58.9 | 51.65 | 19.79 |
| consistent_repr | 71.44 | 71.16 | 69.45 | 67.21 | 64.7 | 6.74 |
| feature_accuracy | 71.44 | 70.09 | 69.0 | 67.44 | 65.98 | 5.46 |
| uniqueness | 71.44 | 65.97 | 63.69 | 60.77 | 60.3 | 11.14 |

### letter

| 오염 유형 | 0%(baseline) | 10% | 25% | 50% | 75% | 최대 하락폭 |
|---|---|---|---|---|---|---|
| class_balance | 91.59 | 92.2 | 90.53 | 88.2 | 86.12 | 5.47 |
| completeness | 91.59 | 82.24 | 75.94 | 67.14 | 56.86 | 34.73 |
| feature_accuracy | 91.59 | 90.89 | 83.53 | 74.48 | 71.28 | 20.31 |
| uniqueness | 91.59 | 85.9 | 83.34 | 80.89 | 79.2 | 12.39 |

## 4. 에러 목록

_에러 없음_

## 5. 자동 감지된 문제점

- ⚠️ **TelcoCustomerChurn/class_balance_75%**: DSC 하락 4.4점 (5점 미만 — 감지 부족)
- ⚠️ **SouthGermanCredit/class_balance_75%**: DSC 하락 3.0점 (5점 미만 — 감지 부족)

## 6. 산출물

- `dsc_scores.csv` — DSC 점수 87건
- `data/polluted/` — 오염 데이터 파일
- `02_execution_log.md` — 이 로그 파일

---
*이 로그는 노트북 02 실행 시 자동 생성됨*