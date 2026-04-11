# ADR-008: DSC 스코어링 엔진의 오염 감지 능력 부족 — 개선 필요

- **일자**: 2026-04-12
- **상태**: 확정 (v2 구현 완료)

## 배경

노트북 02 실행 결과, DQ4AI polluter로 최대 75%까지 오염을 가했음에도
DSC 점수가 거의 변하지 않는 심각한 문제를 발견했다.

## 현상 (TelcoCustomerChurn 기준)

| 오염 유형 | 0% (baseline) | 10% | 25% | 50% | 75% | 변화폭 |
|---|---|---|---|---|---|---|
| completeness | 99.17 | 99.16 | 99.03 | 98.65 | 97.20 | **-1.97** |
| uniqueness | 99.17 | 92.51 | 89.17 | 85.84 | 84.17 | **-15.00** |
| feature_accuracy | 99.17 | ERROR | ERROR | ERROR | ERROR | - |
| consistent_repr | 99.17 | 92.14 | 92.14 | 92.14 | 92.14 | **고정** |
| class_balance | 99.17 | 99.94 | 99.77 | 99.06 | 97.72 | **-1.45** |

- uniqueness만 유의미한 변화 (15점 하락)
- completeness, class_balance는 2점 미만 변화
- consistent_repr는 강도에 무관하게 동일 점수
- 결과적으로 **C, D 등급이 하나도 없음** → 상관관계 입증 불가

## 원인 분석

### 1. Completeness: placeholder vs null 불일치

| 구성요소 | 동작 |
|---|---|
| CompletenessPolluter | 결측값을 **placeholder**로 삽입 (수치: `-1`, 범주: `"empty"`) |
| calc_completeness() | `df.isnull().sum()`으로 계산 |
| **결과** | placeholder는 null이 아니므로 감지 불가 → 점수 변화 없음 |

### 2. Consistency: 오염 강도와 점수 무관

| 구성요소 | 동작 |
|---|---|
| ConsistentRepresentationPolluter | 범주값에 `-1`, `-2` 접미사 추가 |
| calc_consistency() | `re.sub(r'-\d+$', '', x)`로 접미사 제거 후 unique 비교 |
| **결과** | 점수가 오염 강도와 무관하게 항상 동일값 |

추정 원인:
- polluter가 `percentage_polluted_rows` 파라미터를 내부적으로 다르게 처리할 가능성
- 또는 접미사 제거 후 비교하는 로직이 오염의 "양"이 아닌 "존재 여부"만 반영

### 3. Class Balance: 엔트로피의 낮은 민감도

이진 분류(Churn: Yes 26%, No 74%)에서 엔트로피는:
- 원본: `H = -0.26×log₂(0.26) - 0.74×log₂(0.74) = 0.827`
- 최대: `log₂(2) = 1.0`
- 점수: `0.827 / 1.0 = 0.827`

ClassBalancePolluter가 비율을 조정해도 엔트로피 변화가 작음.

## 수정 방향

### Completeness 수정
```python
# 기존: null만 카운트
null_count = feature_df.isnull().sum().sum()

# 수정: placeholder도 카운트
placeholder_numerical = meta.get('placeholder_numerical', -1)
placeholder_categorical = meta.get('placeholder_categorical', 'empty')
# 수치형 컬럼에서 placeholder_numerical 값 카운트
# 범주형 컬럼에서 placeholder_categorical 값 카운트
# null + placeholder = total missing
```

### Consistency 수정
접미사 제거 비교 방식 대신, **오염 전 원본의 unique values 대비 얼마나 새로운 값이 생겼는지**로 변경하거나,
접미사가 있는 값의 비율 자체를 직접 측정

### Class Balance 수정
엔트로피 대신 더 민감한 지표 검토 (Gini impurity, 최소 클래스 비율 등)
또는 가중치 자체가 0.05로 작으므로 영향 제한적 — 다른 지표 수정이 더 우선

## 영향

이 수정 없이는 프로젝트의 핵심 주장("DSC 점수 ↔ 모델 성능 상관관계")을 입증할 수 없다.
수정 후에는 오염 강도에 따라 DSC 점수가 유의미하게 하락하여 A~D 전 등급에 걸친 분포가 생길 것으로 기대.

## 영향받은 파일

- `01_setup_and_baseline.ipynb` (DSC 엔진 정의 셀)
- `02_pollution_and_dsc.ipynb` (DSC 엔진 정의 셀 — 01과 동일 코드 중복)
