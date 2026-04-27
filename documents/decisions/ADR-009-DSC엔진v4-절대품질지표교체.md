# ADR-009: DSC 엔진 v4 — value_accuracy 제거, 절대품질 지표 신설

- **일자**: 2026-04-27
- **상태**: 확정 (Phase 1 시작 전 사전 등록)

## 배경

v3.2의 `value_accuracy` 지표는 reference_df 분포 대비 KS / TVD 거리로 정의되어 있었다.
- 가중치 0.30으로 8개 지표 중 압도적 1위
- reference 의존이라 baseline 자체에서는 1.0 (자명) → 절대 품질 점수가 아닌 **drift 측정 도구**
- 정의(원본 = 100점)가 임의 anchor에 갇혀, "데이터 품질 점수"라는 제품 정체성과 모순

DSC는 반드시 **데이터 품질 점수**여야 한다는 사용자 요구사항(2026-04-27 결정)을 충족하려면, 이 지표를 reference-free 절대 지표로 교체해야 한다.

## 결정

`value_accuracy(0.30)`를 제거하고 다음 두 절대 품질 지표를 신설.

### `label_consistency` (가중치 0.20)

```python
def calc_label_consistency(df, target_col, numerical_cols, k=5, sample_cap=2000):
    """k-NN 라벨 일관성 — 각 샘플의 k 최근접 이웃 라벨 중 자기 라벨과 같은 비율의 평균.
    수치형 컬럼만 표준화 후 KNN. ML 학습 가능성 / 라벨 노이즈 / 결정경계 모호도 직접 측정.
    sample_cap: 큰 데이터에서 무작위 샘플링으로 계산 비용 제한 (재현성 위해 random_state=1)."""
```

**ML 인과**:
- 라벨 노이즈가 많으면 이웃 라벨이 자기와 다름 → 점수 하락
- 결정경계 근처 샘플이 많으면(클래스 분리 약함) 점수 하락
- ML 모델의 학습 가능성과 직접 연관

### `feature_informativeness` (가중치 0.10)

```python
def calc_feature_informativeness(df, target_col, numerical_cols, categorical_cols, sample_cap=2000):
    """sklearn.feature_selection.mutual_info_classif 평균.
    피처가 라벨에 주는 정보량의 평균. 0~1로 정규화 (관측 최댓값 기준)."""
```

**ML 인과**:
- 피처가 라벨과 무관하면 MI ≈ 0 → 학습 불가능
- 결측·노이즈 추가 시 MI 감소 → 정보 손실 정확히 반영

## 신 가중치 (Default profile)

| 지표 | v3.2 | **v4.0** | 변화 |
|---|---:|---:|---|
| completeness | 0.20 | 0.20 | — |
| uniqueness | 0.15 | 0.15 | — |
| validity | 0.05 | 0.05 | — |
| consistency | 0.10 | 0.10 | — |
| outlier_ratio | 0.05 | 0.05 | — |
| class_balance | 0.10 | 0.10 | — |
| feature_correlation | 0.05 | 0.05 | — |
| **value_accuracy** | **0.30** | **제거** | −0.30 |
| **label_consistency** | — | **0.20** | +0.20 (신설) |
| **feature_informativeness** | — | **0.10** | +0.10 (신설) |
| **합** | **1.00** | **1.00** | — |

value_accuracy 0.30이 가지던 자리를 label_consistency(0.20) + feature_informativeness(0.10)으로 분배.

## reference_df 인자의 운명

- `outlier_ratio`에서는 v3 그대로 유지 (자기참조 함정 회피용 IQR 고정)
- `value_accuracy` 제거로 더 이상 핵심 의존성 아님
- `consistency`도 reference 사용 (v3.2: 새 표현 행 비율) — 이것도 유지

reference_df는 "outlier·consistency 측정의 보조 anchor"로 역할이 좁혀짐. DSC 점수의 30%를 차지하던 의존성이 사라지므로, **reference 없는 데이터에서도 신 v4의 70% (label_consistency + feature_informativeness + completeness + uniqueness + validity + class_balance + feature_correlation)는 그대로 작동**.

## 위험과 대응

### 위험 1 — label_consistency가 ML 성능을 정의적으로 측정
k-NN 라벨 일관성은 ML 분류 가능성을 직접 측정하므로, F1과의 상관이 자명하게 높을 수 있음. 그러면 "DSC가 ML을 예측"이 동어반복으로 들림.

**대응**: Phase 2 hold-out 검증으로 fitting 위험 제어. label_consistency가 모든 polluter 슬라이스에서 ML 성능과 일관되게 움직이는지 확인.

### 위험 2 — 계산 비용
- 큰 데이터(letter 20K)에서 KNN 거리 계산이 O(n²). 메모리·시간 부담.
- mutual_info_classif도 큰 데이터에서 느림.

**대응**: `sample_cap=2000` 도입. 재현성 위해 `random_state=1` 고정 무작위 샘플링.

### 위험 3 — categorical 피처 처리
k-NN은 거리 기반이라 수치형만 사용. categorical은 제외 (Telco, SouthGerman은 numerical만으로도 충분히 신호 잡힘).

**대응**: 수치형이 너무 적은 데이터(예: 수치형 1개 이하)에서는 onehot 인코딩 후 사용 또는 1.0 fallback.

## 영향받는 파일

- `notebooks/01_setup_and_baseline.ipynb` (cell 11 — DSC 엔진 정의)
- `notebooks/02_pollution_and_dsc.ipynb` (cell 8 — DSC 엔진 정의)
- `documents/plans/20260427-01-DSC엔진v4-개선마스터플랜.md` (사전 등록)

## 사후 검증 — 노트북 재실행 후

- baseline에서 두 신지표 값 ≥ 0.7 (정상 데이터)
- feature_accuracy 0.75 폴루션 → label_consistency Δ ≤ −0.10
- completeness 0.75 폴루션 → 두 지표 모두 적절히 하락
- uniqueness 0.75 폴루션 → 두 지표 거의 변화 없음 (false positive 검증)

이 4개 검증 모두 통과해야 v4 엔진 정식 채택.
