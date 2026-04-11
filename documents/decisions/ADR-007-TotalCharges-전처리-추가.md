# ADR-007: TelcoCustomerChurn TotalCharges 전처리 추가

- **일자**: 2026-04-12
- **상태**: 확정

## 배경

노트북 02에서 FeatureAccuracyPolluter를 TelcoCustomerChurn에 적용할 때,
4개 강도(10/25/50/75%) 모두에서 동일한 에러가 발생했다:

```
ValueError: Could not convert string '29.851889.5108.15...' to numeric
```

## 원인 분석

1. TelcoCustomerChurn의 `TotalCharges` 컬럼에 **공백 문자열 `" "`이 11건** 포함
2. pandas가 해당 컬럼을 `object` (문자열) 타입으로 인식
3. FeatureAccuracyPolluter가 문자열 타입 컬럼으로 판단하여 **가우시안 노이즈 대신 문자열 연결(concatenation)** 수행
4. 결과: 수만 글자의 비정상 문자열이 생성되어 수치 변환 실패

원본 데이터의 문제:
```
# TotalCharges에 공백이 있는 행 (신규 가입자, tenure=0)
customerID  tenure  TotalCharges
4472-LVYGI  0       " "
3115-CZMZD  0       " "
...
```

## 선택지

| 선택지 | 장점 | 단점 |
|---|---|---|
| A. TotalCharges를 numerical_cols에서 제외 | 간단 | 주요 피처 손실, 원본 데이터 왜곡 |
| **B. polluter 적용 전 수치형 변환** | 데이터 보존, polluter 정상 동작 | 전처리 코드 추가 |
| C. 에러를 무시하고 feature_accuracy 결과 제외 | 코드 변경 없음 | Telco × feature_accuracy 4건 데이터 손실 |

## 결정

**선택지 B.** 오염 루프에서 데이터 로드 직후, polluter 적용 전에 TotalCharges를 수치형으로 변환한다.

```python
if 'TotalCharges' in df_clean.columns:
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce').fillna(0)
```

## 이유

- 원본 데이터의 `" "` → `0`은 의미적으로 타당 (tenure=0인 신규 가입자의 총 요금은 0)
- 피처를 제외하면 3개 수치형 중 1개를 잃어 실험 의미 축소
- 에러를 무시하면 60건 중 4건(Telco × feature_accuracy 전체)이 빠져 결과 불완전

## 영향받은 파일

- `02_pollution_and_dsc.ipynb` (cell-9: 오염 실행 루프)
