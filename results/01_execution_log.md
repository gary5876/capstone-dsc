# 노트북 01 실행 로그: 환경 세팅 & 베이스라인

- **실행 시각**: 2026-04-27 07:28:51
- **BASE 경로**: /content/drive/MyDrive/capstone/dsc

## 1. 데이터 다운로드

| 데이터셋 | 행 | 열 | 타겟 | 클래스 수 | Null 합계 |
|---|---|---|---|---|---|
| TelcoCustomerChurn | 7,043 | 21 | Churn | 2 | 0 |
| SouthGermanCredit | 1,000 | 21 | credit_risk | 2 | 0 |
| letter | 20,000 | 17 | lettr | 26 | 0 |

## 2. 베이스라인 DSC 점수

| 데이터셋 | DSC Score | 등급 | completeness | uniqueness | validity | consistency | outlier_ratio | class_balance | feature_correlation |
|---|---|---|---|---|---|---|---|---|---|
| TelcoCustomerChurn | 79.73 | B | 1.0 | 1.0 | 0.9999 | 1.0 | 1.0 | 0.5307 | 1.0 |
| SouthGermanCredit | 71.16 | C | 1.0 | 1.0 | 1.0 | 1.0 | 0.945 | 0.6 | 1.0 |
| letter | 91.94 | A | 1.0 | 0.9334 | 1.0 | 1.0 | 0.9671 | 0.9542 | 1.0 |

## 3. 베이스라인 모델 성능

| 데이터셋 | 모델 | Accuracy | F1(macro) | AUC-ROC |
|---|---|---|---|---|
| TelcoCustomerChurn | LogisticRegression | 0.8084 | 0.7389 | 0.8349 |
| TelcoCustomerChurn | RandomForest | 0.7835 | 0.698 | 0.8055 |
| TelcoCustomerChurn | XGBoost | 0.7835 | 0.7041 | 0.8124 |
| TelcoCustomerChurn | SVC | 0.8041 | 0.7303 | 0.8241 |
| TelcoCustomerChurn | MLP | 0.7509 | 0.6792 | 0.7632 |
| SouthGermanCredit | LogisticRegression | 0.76 | 0.6991 | 0.7926 |
| SouthGermanCredit | RandomForest | 0.785 | 0.6988 | 0.8049 |
| SouthGermanCredit | XGBoost | 0.795 | 0.7415 | 0.8119 |
| SouthGermanCredit | SVC | 0.74 | 0.674 | 0.781 |
| SouthGermanCredit | MLP | 0.77 | 0.7235 | 0.759 |
| letter | LogisticRegression | 0.766 | 0.7644 | 0.9786 |
| letter | RandomForest | 0.9577 | 0.9577 | 0.9993 |
| letter | XGBoost | 0.9593 | 0.9592 | 0.9996 |
| letter | SVC | 0.8455 | 0.8447 | 0.9935 |
| letter | MLP | 0.957 | 0.957 | 0.9996 |

## 4. 산출물

-  — 베이스라인 DSC 점수 3건
-  — 베이스라인 모델 성능 15건

---
*이 로그는 노트북 01 실행 시 자동 생성됨*