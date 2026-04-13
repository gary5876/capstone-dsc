# 노트북 03 실행 로그: 모델 학습 & 평가

- **실행 시각**: 2026-04-12 15:24:43
- **총 실험**: 295건
- **신규 학습**: 280건
- **스킵 (기존 결과)**: 15건
- **에러**: 0건
- **소요 시간**: 4238초 (70.6분)

## 1. 모델 설정

| 모델 | 계열 | 주요 하이퍼파라미터 |
|---|---|---|
| LogisticRegression | 선형 | solver=lbfgs, max_iter=2000 |
| RandomForest | 배깅 | n_estimators=100 |
| XGBoost | 부스팅 | n_estimators=100 |
| SVC | 커널 | kernel=linear |
| MLP | 신경망 | hidden_layers=(100,100,100,100,100), max_iter=1000 |

- 전처리: StandardScaler(수치) + OneHotEncoder(범주)
- 분할: train 80% / test 20% (stratified)
- 평가: Accuracy, F1(macro), AUC-ROC

## 2. 데이터셋별 결과 요약

### TelcoCustomerChurn (105건)

| 오염 유형 | 강도 | 모델 | F1(macro) | Accuracy | AUC-ROC |
|---|---|---|---|---|---|
| none | 0.0 | LogisticRegression | 0.7389 | 0.8084 | 0.8349 |
| none | 0.0 | RandomForest | 0.698 | 0.7835 | 0.8055 |
| none | 0.0 | XGBoost | 0.7041 | 0.7835 | 0.8124 |
| none | 0.0 | SVC | 0.7303 | 0.8041 | 0.8241 |
| none | 0.0 | MLP | 0.6792 | 0.7509 | 0.7632 |
| class_balance | 0.1 | LogisticRegression | 0.7212 | 0.7573 | 0.8341 |
| class_balance | 0.1 | RandomForest | 0.6962 | 0.7374 | 0.8095 |
| class_balance | 0.1 | XGBoost | 0.6972 | 0.7346 | 0.8059 |
| class_balance | 0.1 | SVC | 0.7104 | 0.7424 | 0.8339 |
| class_balance | 0.1 | MLP | 0.6425 | 0.6799 | 0.7301 |
| class_balance | 0.25 | LogisticRegression | 0.7285 | 0.7764 | 0.8347 |
| class_balance | 0.25 | RandomForest | 0.7161 | 0.7679 | 0.8096 |
| class_balance | 0.25 | XGBoost | 0.697 | 0.7466 | 0.7988 |
| class_balance | 0.25 | SVC | 0.7225 | 0.7722 | 0.8311 |
| class_balance | 0.25 | MLP | 0.6837 | 0.7317 | 0.7715 |
| class_balance | 0.5 | LogisticRegression | 0.7373 | 0.8084 | 0.8348 |
| class_balance | 0.5 | RandomForest | 0.692 | 0.7821 | 0.8103 |
| class_balance | 0.5 | XGBoost | 0.6873 | 0.7708 | 0.8028 |
| class_balance | 0.5 | SVC | 0.7284 | 0.8027 | 0.8237 |
| class_balance | 0.5 | MLP | 0.6413 | 0.7331 | 0.749 |
| class_balance | 0.75 | LogisticRegression | 0.5825 | 0.7743 | 0.8376 |
| class_balance | 0.75 | RandomForest | 0.6266 | 0.78 | 0.8139 |
| class_balance | 0.75 | XGBoost | 0.6274 | 0.7708 | 0.8037 |
| class_balance | 0.75 | SVC | 0.4235 | 0.7346 | 0.4189 |
| class_balance | 0.75 | MLP | 0.6343 | 0.7551 | 0.7701 |
| completeness | 0.1 | LogisticRegression | 0.7373 | 0.8062 | 0.8317 |
| completeness | 0.1 | RandomForest | 0.6947 | 0.7828 | 0.8202 |
| completeness | 0.1 | XGBoost | 0.7091 | 0.7793 | 0.8147 |
| completeness | 0.1 | SVC | 0.7303 | 0.8013 | 0.8254 |
| completeness | 0.1 | MLP | 0.6489 | 0.7537 | 0.7632 |
| completeness | 0.25 | LogisticRegression | 0.735 | 0.8055 | 0.8318 |
| completeness | 0.25 | RandomForest | 0.7105 | 0.7928 | 0.8217 |
| completeness | 0.25 | XGBoost | 0.7139 | 0.78 | 0.8069 |
| completeness | 0.25 | SVC | 0.7344 | 0.8013 | 0.8266 |
| completeness | 0.25 | MLP | 0.651 | 0.7246 | 0.756 |
| completeness | 0.5 | LogisticRegression | 0.7213 | 0.7977 | 0.8273 |
| completeness | 0.5 | RandomForest | 0.7118 | 0.8006 | 0.8193 |
| completeness | 0.5 | XGBoost | 0.6988 | 0.7786 | 0.8079 |
| completeness | 0.5 | SVC | 0.7268 | 0.7935 | 0.8233 |
| completeness | 0.5 | MLP | 0.6737 | 0.7573 | 0.7874 |
| completeness | 0.75 | LogisticRegression | 0.7356 | 0.7984 | 0.82 |
| completeness | 0.75 | RandomForest | 0.6398 | 0.7835 | 0.8108 |
| completeness | 0.75 | XGBoost | 0.6693 | 0.7665 | 0.7677 |
| completeness | 0.75 | SVC | 0.4235 | 0.7346 | 0.813 |
| completeness | 0.75 | MLP | 0.4309 | 0.7339 | 0.6999 |
| consistent_repr | 0.1 | LogisticRegression | 0.7449 | 0.8084 | 0.8346 |
| consistent_repr | 0.1 | RandomForest | 0.7176 | 0.7963 | 0.8244 |
| consistent_repr | 0.1 | XGBoost | 0.7298 | 0.7991 | 0.8218 |
| consistent_repr | 0.1 | SVC | 0.7371 | 0.8055 | 0.8245 |
| consistent_repr | 0.1 | MLP | 0.6708 | 0.7303 | 0.7575 |
| consistent_repr | 0.25 | LogisticRegression | 0.7453 | 0.8098 | 0.8346 |
| consistent_repr | 0.25 | RandomForest | 0.7114 | 0.7899 | 0.8233 |
| consistent_repr | 0.25 | XGBoost | 0.701 | 0.7771 | 0.8073 |
| consistent_repr | 0.25 | SVC | 0.735 | 0.8034 | 0.8243 |
| consistent_repr | 0.25 | MLP | 0.6719 | 0.731 | 0.7566 |
| consistent_repr | 0.5 | LogisticRegression | 0.7435 | 0.8141 | 0.8355 |
| consistent_repr | 0.5 | RandomForest | 0.7015 | 0.7963 | 0.8257 |
| consistent_repr | 0.5 | XGBoost | 0.6869 | 0.7644 | 0.8063 |
| consistent_repr | 0.5 | SVC | 0.7374 | 0.8048 | 0.8265 |
| consistent_repr | 0.5 | MLP | 0.5922 | 0.7246 | 0.724 |
| consistent_repr | 0.75 | LogisticRegression | 0.7305 | 0.8048 | 0.8294 |
| consistent_repr | 0.75 | RandomForest | 0.6195 | 0.78 | 0.8114 |
| consistent_repr | 0.75 | XGBoost | 0.6153 | 0.763 | 0.7781 |
| consistent_repr | 0.75 | SVC | 0.7332 | 0.8048 | 0.8201 |
| consistent_repr | 0.75 | MLP | 0.4868 | 0.7346 | 0.6481 |
| feature_accuracy | 0.1 | LogisticRegression | 0.7392 | 0.8041 | 0.8315 |
| feature_accuracy | 0.1 | RandomForest | 0.7189 | 0.7942 | 0.8138 |
| feature_accuracy | 0.1 | XGBoost | 0.7042 | 0.7757 | 0.8077 |
| feature_accuracy | 0.1 | SVC | 0.7301 | 0.8006 | 0.8238 |
| feature_accuracy | 0.1 | MLP | 0.6306 | 0.7104 | 0.7297 |
| feature_accuracy | 0.25 | LogisticRegression | 0.7345 | 0.7963 | 0.8303 |
| feature_accuracy | 0.25 | RandomForest | 0.7066 | 0.7842 | 0.8187 |
| feature_accuracy | 0.25 | XGBoost | 0.6991 | 0.7679 | 0.8059 |
| feature_accuracy | 0.25 | SVC | 0.7313 | 0.7991 | 0.8261 |
| feature_accuracy | 0.25 | MLP | 0.6741 | 0.7509 | 0.7495 |
| feature_accuracy | 0.5 | LogisticRegression | 0.7254 | 0.7899 | 0.822 |
| feature_accuracy | 0.5 | RandomForest | 0.694 | 0.7871 | 0.7952 |
| feature_accuracy | 0.5 | XGBoost | 0.7009 | 0.7722 | 0.7828 |
| feature_accuracy | 0.5 | SVC | 0.4235 | 0.7346 | 0.8207 |
| feature_accuracy | 0.5 | MLP | 0.6125 | 0.7126 | 0.7229 |
| feature_accuracy | 0.75 | LogisticRegression | 0.4216 | 0.7083 | 0.4217 |
| feature_accuracy | 0.75 | RandomForest | 0.4314 | 0.7161 | 0.4705 |
| feature_accuracy | 0.75 | XGBoost | 0.4373 | 0.6402 | 0.4599 |
| feature_accuracy | 0.75 | SVC | 0.4235 | 0.7346 | 0.3924 |
| feature_accuracy | 0.75 | MLP | 0.5053 | 0.6331 | 0.5302 |
| uniqueness | 0.1 | LogisticRegression | 0.7425 | 0.8105 | 0.8341 |
| uniqueness | 0.1 | RandomForest | 0.7051 | 0.7871 | 0.8014 |
| uniqueness | 0.1 | XGBoost | 0.7183 | 0.7906 | 0.8121 |
| uniqueness | 0.1 | SVC | 0.7348 | 0.807 | 0.8243 |
| uniqueness | 0.1 | MLP | 0.6656 | 0.7289 | 0.7514 |
| uniqueness | 0.25 | LogisticRegression | 0.7406 | 0.8091 | 0.8352 |
| uniqueness | 0.25 | RandomForest | 0.6906 | 0.7757 | 0.7928 |
| uniqueness | 0.25 | XGBoost | 0.6954 | 0.7736 | 0.8043 |
| uniqueness | 0.25 | SVC | 0.727 | 0.8013 | 0.8234 |
| uniqueness | 0.25 | MLP | 0.6549 | 0.736 | 0.7496 |
| uniqueness | 0.5 | LogisticRegression | 0.7425 | 0.8105 | 0.8346 |
| uniqueness | 0.5 | RandomForest | 0.6932 | 0.7771 | 0.7891 |
| uniqueness | 0.5 | XGBoost | 0.7028 | 0.7779 | 0.8089 |
| uniqueness | 0.5 | SVC | 0.7298 | 0.8041 | 0.824 |
| uniqueness | 0.5 | MLP | 0.6457 | 0.7232 | 0.7413 |
| uniqueness | 0.75 | LogisticRegression | 0.7404 | 0.8084 | 0.8355 |
| uniqueness | 0.75 | RandomForest | 0.6955 | 0.775 | 0.7909 |
| uniqueness | 0.75 | XGBoost | 0.7068 | 0.7821 | 0.8056 |
| uniqueness | 0.75 | SVC | 0.7332 | 0.8048 | 0.8253 |
| uniqueness | 0.75 | MLP | 0.6555 | 0.7339 | 0.734 |

### SouthGermanCredit (105건)

| 오염 유형 | 강도 | 모델 | F1(macro) | Accuracy | AUC-ROC |
|---|---|---|---|---|---|
| none | 0.0 | LogisticRegression | 0.6991 | 0.76 | 0.7926 |
| none | 0.0 | RandomForest | 0.6988 | 0.785 | 0.8049 |
| none | 0.0 | XGBoost | 0.7415 | 0.795 | 0.8119 |
| none | 0.0 | SVC | 0.674 | 0.74 | 0.781 |
| none | 0.0 | MLP | 0.7235 | 0.77 | 0.759 |
| class_balance | 0.1 | LogisticRegression | 0.6961 | 0.74 | 0.7813 |
| class_balance | 0.1 | RandomForest | 0.6992 | 0.735 | 0.7781 |
| class_balance | 0.1 | XGBoost | 0.6339 | 0.675 | 0.7482 |
| class_balance | 0.1 | SVC | 0.6844 | 0.73 | 0.7618 |
| class_balance | 0.1 | MLP | 0.6988 | 0.74 | 0.7398 |
| class_balance | 0.25 | LogisticRegression | 0.6919 | 0.745 | 0.7707 |
| class_balance | 0.25 | RandomForest | 0.6702 | 0.74 | 0.7851 |
| class_balance | 0.25 | XGBoost | 0.7132 | 0.765 | 0.7882 |
| class_balance | 0.25 | SVC | 0.6905 | 0.74 | 0.7539 |
| class_balance | 0.25 | MLP | 0.6934 | 0.74 | 0.7355 |
| class_balance | 0.5 | LogisticRegression | 0.6568 | 0.755 | 0.7574 |
| class_balance | 0.5 | RandomForest | 0.5534 | 0.745 | 0.774 |
| class_balance | 0.5 | XGBoost | 0.6893 | 0.785 | 0.8023 |
| class_balance | 0.5 | SVC | 0.6147 | 0.725 | 0.7343 |
| class_balance | 0.5 | MLP | 0.633 | 0.74 | 0.7307 |
| class_balance | 0.75 | LogisticRegression | 0.5614 | 0.73 | 0.7874 |
| class_balance | 0.75 | RandomForest | 0.4465 | 0.71 | 0.7678 |
| class_balance | 0.75 | XGBoost | 0.5462 | 0.735 | 0.7413 |
| class_balance | 0.75 | SVC | 0.5184 | 0.725 | 0.7764 |
| class_balance | 0.75 | MLP | 0.5979 | 0.755 | 0.7565 |
| completeness | 0.1 | LogisticRegression | 0.7116 | 0.77 | 0.8012 |
| completeness | 0.1 | RandomForest | 0.6894 | 0.775 | 0.8084 |
| completeness | 0.1 | XGBoost | 0.6829 | 0.75 | 0.7319 |
| completeness | 0.1 | SVC | 0.6843 | 0.74 | 0.7918 |
| completeness | 0.1 | MLP | 0.6534 | 0.73 | 0.7656 |
| completeness | 0.25 | LogisticRegression | 0.683 | 0.735 | 0.788 |
| completeness | 0.25 | RandomForest | 0.6891 | 0.77 | 0.7842 |
| completeness | 0.25 | XGBoost | 0.745 | 0.79 | 0.7894 |
| completeness | 0.25 | SVC | 0.671 | 0.725 | 0.7815 |
| completeness | 0.25 | MLP | 0.6366 | 0.685 | 0.7233 |
| completeness | 0.5 | LogisticRegression | 0.692 | 0.76 | 0.7795 |
| completeness | 0.5 | RandomForest | 0.6942 | 0.785 | 0.7867 |
| completeness | 0.5 | XGBoost | 0.6533 | 0.74 | 0.7502 |
| completeness | 0.5 | SVC | 0.6342 | 0.735 | 0.7623 |
| completeness | 0.5 | MLP | 0.5828 | 0.7 | 0.7327 |
| completeness | 0.75 | LogisticRegression | 0.6658 | 0.7 | 0.7702 |
| completeness | 0.75 | RandomForest | 0.5057 | 0.705 | 0.7381 |
| completeness | 0.75 | XGBoost | 0.6135 | 0.665 | 0.6846 |
| completeness | 0.75 | SVC | 0.6739 | 0.715 | 0.7404 |
| completeness | 0.75 | MLP | 0.4274 | 0.7 | 0.6604 |
| consistent_repr | 0.1 | LogisticRegression | 0.674 | 0.74 | 0.7799 |
| consistent_repr | 0.1 | RandomForest | 0.6459 | 0.755 | 0.7919 |
| consistent_repr | 0.1 | XGBoost | 0.7289 | 0.785 | 0.7825 |
| consistent_repr | 0.1 | SVC | 0.682 | 0.745 | 0.7736 |
| consistent_repr | 0.1 | MLP | 0.7272 | 0.78 | 0.7649 |
| consistent_repr | 0.25 | LogisticRegression | 0.686 | 0.735 | 0.7955 |
| consistent_repr | 0.25 | RandomForest | 0.721 | 0.795 | 0.8053 |
| consistent_repr | 0.25 | XGBoost | 0.721 | 0.78 | 0.811 |
| consistent_repr | 0.25 | SVC | 0.6827 | 0.725 | 0.7876 |
| consistent_repr | 0.25 | MLP | 0.7169 | 0.76 | 0.7798 |
| consistent_repr | 0.5 | LogisticRegression | 0.6667 | 0.72 | 0.7529 |
| consistent_repr | 0.5 | RandomForest | 0.617 | 0.735 | 0.7687 |
| consistent_repr | 0.5 | XGBoost | 0.6572 | 0.685 | 0.7414 |
| consistent_repr | 0.5 | SVC | 0.6696 | 0.735 | 0.7589 |
| consistent_repr | 0.5 | MLP | 0.657 | 0.69 | 0.7267 |
| consistent_repr | 0.75 | LogisticRegression | 0.6667 | 0.72 | 0.7501 |
| consistent_repr | 0.75 | RandomForest | 0.5291 | 0.725 | 0.7051 |
| consistent_repr | 0.75 | XGBoost | 0.5803 | 0.675 | 0.6735 |
| consistent_repr | 0.75 | SVC | 0.6269 | 0.67 | 0.723 |
| consistent_repr | 0.75 | MLP | 0.6278 | 0.71 | 0.729 |
| feature_accuracy | 0.1 | LogisticRegression | 0.6802 | 0.765 | 0.7985 |
| feature_accuracy | 0.1 | RandomForest | 0.6135 | 0.755 | 0.7969 |
| feature_accuracy | 0.1 | XGBoost | 0.6887 | 0.765 | 0.7565 |
| feature_accuracy | 0.1 | SVC | 0.6887 | 0.765 | 0.7938 |
| feature_accuracy | 0.1 | MLP | 0.7086 | 0.76 | 0.7554 |
| feature_accuracy | 0.25 | LogisticRegression | 0.6245 | 0.73 | 0.7852 |
| feature_accuracy | 0.25 | RandomForest | 0.5357 | 0.72 | 0.7607 |
| feature_accuracy | 0.25 | XGBoost | 0.6287 | 0.735 | 0.7542 |
| feature_accuracy | 0.25 | SVC | 0.5946 | 0.715 | 0.7717 |
| feature_accuracy | 0.25 | MLP | 0.6236 | 0.705 | 0.7174 |
| feature_accuracy | 0.5 | LogisticRegression | 0.5939 | 0.75 | 0.7698 |
| feature_accuracy | 0.5 | RandomForest | 0.41 | 0.695 | 0.7158 |
| feature_accuracy | 0.5 | XGBoost | 0.552 | 0.695 | 0.7263 |
| feature_accuracy | 0.5 | SVC | 0.4118 | 0.7 | 0.7685 |
| feature_accuracy | 0.5 | MLP | 0.5444 | 0.695 | 0.6952 |
| feature_accuracy | 0.75 | LogisticRegression | 0.4353 | 0.685 | 0.5762 |
| feature_accuracy | 0.75 | RandomForest | 0.4065 | 0.685 | 0.568 |
| feature_accuracy | 0.75 | XGBoost | 0.4072 | 0.65 | 0.5092 |
| feature_accuracy | 0.75 | SVC | 0.4118 | 0.7 | 0.429 |
| feature_accuracy | 0.75 | MLP | 0.4478 | 0.64 | 0.4586 |
| uniqueness | 0.1 | LogisticRegression | 0.691 | 0.755 | 0.7868 |
| uniqueness | 0.1 | RandomForest | 0.6799 | 0.775 | 0.7995 |
| uniqueness | 0.1 | XGBoost | 0.759 | 0.81 | 0.813 |
| uniqueness | 0.1 | SVC | 0.7083 | 0.77 | 0.7738 |
| uniqueness | 0.1 | MLP | 0.7254 | 0.775 | 0.7585 |
| uniqueness | 0.25 | LogisticRegression | 0.6945 | 0.755 | 0.7821 |
| uniqueness | 0.25 | RandomForest | 0.7285 | 0.795 | 0.7998 |
| uniqueness | 0.25 | XGBoost | 0.7433 | 0.8 | 0.7829 |
| uniqueness | 0.25 | SVC | 0.6978 | 0.755 | 0.7707 |
| uniqueness | 0.25 | MLP | 0.6732 | 0.735 | 0.7454 |
| uniqueness | 0.5 | LogisticRegression | 0.69 | 0.75 | 0.7827 |
| uniqueness | 0.5 | RandomForest | 0.6891 | 0.77 | 0.7986 |
| uniqueness | 0.5 | XGBoost | 0.7148 | 0.77 | 0.7894 |
| uniqueness | 0.5 | SVC | 0.6766 | 0.735 | 0.771 |
| uniqueness | 0.5 | MLP | 0.6776 | 0.74 | 0.7414 |
| uniqueness | 0.75 | LogisticRegression | 0.6945 | 0.755 | 0.7869 |
| uniqueness | 0.75 | RandomForest | 0.6848 | 0.775 | 0.8015 |
| uniqueness | 0.75 | XGBoost | 0.7105 | 0.78 | 0.799 |
| uniqueness | 0.75 | SVC | 0.6608 | 0.725 | 0.7682 |
| uniqueness | 0.75 | MLP | 0.6766 | 0.735 | 0.739 |

### letter (85건)

| 오염 유형 | 강도 | 모델 | F1(macro) | Accuracy | AUC-ROC |
|---|---|---|---|---|---|
| none | 0.0 | LogisticRegression | 0.7644 | 0.766 | 0.9786 |
| none | 0.0 | RandomForest | 0.9577 | 0.9577 | 0.9993 |
| none | 0.0 | XGBoost | 0.9592 | 0.9593 | 0.9996 |
| none | 0.0 | SVC | 0.8447 | 0.8455 | 0.9935 |
| none | 0.0 | MLP | 0.957 | 0.957 | 0.9996 |
| class_balance | 0.1 | LogisticRegression | 0.7666 | 0.7678 | 0.9773 |
| class_balance | 0.1 | RandomForest | 0.9324 | 0.9327 | 0.9984 |
| class_balance | 0.1 | XGBoost | 0.9319 | 0.932 | 0.9987 |
| class_balance | 0.1 | SVC | 0.8384 | 0.8387 | 0.992 |
| class_balance | 0.1 | MLP | 0.9378 | 0.938 | 0.9989 |
| class_balance | 0.25 | LogisticRegression | 0.7653 | 0.7675 | 0.9772 |
| class_balance | 0.25 | RandomForest | 0.93 | 0.9303 | 0.9984 |
| class_balance | 0.25 | XGBoost | 0.9264 | 0.9267 | 0.9985 |
| class_balance | 0.25 | SVC | 0.8368 | 0.8373 | 0.992 |
| class_balance | 0.25 | MLP | 0.9405 | 0.941 | 0.9989 |
| class_balance | 0.5 | LogisticRegression | 0.7562 | 0.7602 | 0.9771 |
| class_balance | 0.5 | RandomForest | 0.9242 | 0.9247 | 0.9978 |
| class_balance | 0.5 | XGBoost | 0.9233 | 0.924 | 0.9983 |
| class_balance | 0.5 | SVC | 0.8313 | 0.8325 | 0.9918 |
| class_balance | 0.5 | MLP | 0.9191 | 0.9195 | 0.9984 |
| class_balance | 0.75 | LogisticRegression | 0.738 | 0.7462 | 0.9762 |
| class_balance | 0.75 | RandomForest | 0.915 | 0.9163 | 0.9962 |
| class_balance | 0.75 | XGBoost | 0.915 | 0.9165 | 0.9976 |
| class_balance | 0.75 | SVC | 0.825 | 0.8273 | 0.9909 |
| class_balance | 0.75 | MLP | 0.9227 | 0.9243 | 0.9981 |
| completeness | 0.1 | LogisticRegression | 0.6415 | 0.646 | 0.9521 |
| completeness | 0.1 | RandomForest | 0.935 | 0.935 | 0.9987 |
| completeness | 0.1 | XGBoost | 0.9453 | 0.9455 | 0.9992 |
| completeness | 0.1 | SVC | 0.6899 | 0.6895 | 0.9609 |
| completeness | 0.1 | MLP | 0.9274 | 0.9275 | 0.9982 |
| completeness | 0.25 | LogisticRegression | 0.5208 | 0.5343 | 0.9299 |
| completeness | 0.25 | RandomForest | 0.8981 | 0.8975 | 0.9961 |
| completeness | 0.25 | XGBoost | 0.9241 | 0.9243 | 0.9987 |
| completeness | 0.25 | SVC | 0.5567 | 0.5625 | 0.9324 |
| completeness | 0.25 | MLP | 0.8696 | 0.8702 | 0.9958 |
| completeness | 0.5 | LogisticRegression | 0.2927 | 0.327 | 0.9065 |
| completeness | 0.5 | RandomForest | 0.8153 | 0.8137 | 0.9854 |
| completeness | 0.5 | XGBoost | 0.8635 | 0.8625 | 0.9957 |
| completeness | 0.5 | SVC | 0.2873 | 0.3235 | 0.9049 |
| completeness | 0.5 | MLP | 0.6885 | 0.6827 | 0.9726 |
| completeness | 0.75 | LogisticRegression | 0.1278 | 0.202 | 0.8846 |
| completeness | 0.75 | RandomForest | 0.6741 | 0.6693 | 0.952 |
| completeness | 0.75 | XGBoost | 0.7057 | 0.7025 | 0.977 |
| completeness | 0.75 | SVC | 0.0685 | 0.1445 | 0.8731 |
| completeness | 0.75 | MLP | 0.2164 | 0.2575 | 0.8625 |
| feature_accuracy | 0.1 | LogisticRegression | 0.7582 | 0.758 | 0.9763 |
| feature_accuracy | 0.1 | RandomForest | 0.9226 | 0.9223 | 0.9982 |
| feature_accuracy | 0.1 | XGBoost | 0.9325 | 0.9325 | 0.9989 |
| feature_accuracy | 0.1 | SVC | 0.8229 | 0.8225 | 0.9907 |
| feature_accuracy | 0.1 | MLP | 0.9379 | 0.938 | 0.9993 |
| feature_accuracy | 0.25 | LogisticRegression | 0.7157 | 0.715 | 0.964 |
| feature_accuracy | 0.25 | RandomForest | 0.8127 | 0.8093 | 0.9889 |
| feature_accuracy | 0.25 | XGBoost | 0.8203 | 0.8177 | 0.9932 |
| feature_accuracy | 0.25 | SVC | 0.7581 | 0.755 | 0.9767 |
| feature_accuracy | 0.25 | MLP | 0.8275 | 0.8265 | 0.9932 |
| feature_accuracy | 0.5 | LogisticRegression | 0.6444 | 0.649 | 0.9411 |
| feature_accuracy | 0.5 | RandomForest | 0.6296 | 0.624 | 0.9472 |
| feature_accuracy | 0.5 | XGBoost | 0.604 | 0.599 | 0.9614 |
| feature_accuracy | 0.5 | SVC | 0.6763 | 0.6753 | 0.9481 |
| feature_accuracy | 0.5 | MLP | 0.5132 | 0.5082 | 0.9343 |
| feature_accuracy | 0.75 | LogisticRegression | 0.6039 | 0.6118 | 0.9221 |
| feature_accuracy | 0.75 | RandomForest | 0.4619 | 0.4607 | 0.8742 |
| feature_accuracy | 0.75 | XGBoost | 0.4223 | 0.4138 | 0.9037 |
| feature_accuracy | 0.75 | SVC | 0.633 | 0.633 | 0.9258 |
| feature_accuracy | 0.75 | MLP | 0.2771 | 0.2702 | 0.8239 |
| uniqueness | 0.1 | LogisticRegression | 0.7668 | 0.768 | 0.9787 |
| uniqueness | 0.1 | RandomForest | 0.9609 | 0.961 | 0.9995 |
| uniqueness | 0.1 | XGBoost | 0.9592 | 0.9593 | 0.9996 |
| uniqueness | 0.1 | SVC | 0.8441 | 0.8448 | 0.9934 |
| uniqueness | 0.1 | MLP | 0.9554 | 0.9555 | 0.9995 |
| uniqueness | 0.25 | LogisticRegression | 0.7654 | 0.7665 | 0.9789 |
| uniqueness | 0.25 | RandomForest | 0.9633 | 0.9635 | 0.9994 |
| uniqueness | 0.25 | XGBoost | 0.9585 | 0.9585 | 0.9996 |
| uniqueness | 0.25 | SVC | 0.8449 | 0.8458 | 0.9932 |
| uniqueness | 0.25 | MLP | 0.9591 | 0.9593 | 0.9996 |
| uniqueness | 0.5 | LogisticRegression | 0.7652 | 0.7665 | 0.9788 |
| uniqueness | 0.5 | RandomForest | 0.9634 | 0.9635 | 0.9996 |
| uniqueness | 0.5 | XGBoost | 0.9612 | 0.9613 | 0.9996 |
| uniqueness | 0.5 | SVC | 0.8404 | 0.841 | 0.9932 |
| uniqueness | 0.5 | MLP | 0.9556 | 0.9557 | 0.9993 |
| uniqueness | 0.75 | LogisticRegression | 0.7651 | 0.7665 | 0.9787 |
| uniqueness | 0.75 | RandomForest | 0.9644 | 0.9645 | 0.9995 |
| uniqueness | 0.75 | XGBoost | 0.9629 | 0.963 | 0.9996 |
| uniqueness | 0.75 | SVC | 0.8461 | 0.8468 | 0.9932 |
| uniqueness | 0.75 | MLP | 0.9625 | 0.9627 | 0.9994 |

## 3. 모델별 평균 F1 (전체)

| 모델 | 평균 F1 | 최소 F1 | 최대 F1 | 건수 |
|---|---|---|---|---|
| LogisticRegression | 0.6780 | 0.1278 | 0.7668 | 59 |
| MLP | 0.6843 | 0.2164 | 0.9625 | 59 |
| RandomForest | 0.7091 | 0.4065 | 0.9644 | 59 |
| SVC | 0.6696 | 0.0685 | 0.8461 | 59 |
| XGBoost | 0.7276 | 0.4072 | 0.9629 | 59 |

## 5. 산출물

- `model_performance.csv` — 모델 성능 295건
- `03_execution_log.md` — 이 로그 파일

---
*이 로그는 노트북 03 실행 시 자동 생성됨*