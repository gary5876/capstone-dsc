# DSC v5 — Task-conditional Framework 확장 마스터플랜

- **작성일**: 2026-04-27
- **목적**: DSC를 (data_type, task) → {metric_set, weights} 매핑을 가진 framework로 확장. tabular × regression cell 추가로 framework 주장 성립 (n=2 instance).
- **선행 문서**:
  - `documents/decisions/ADR-011-Task-conditional-Framework-강한버전채택.md` (결정 기록)
  - `documents/reports/20260427-04-v4-정식결과확정.md` (분류 cell 결과 r=0.598)
  - `documents/reports/20260427-10-레퍼런스조사보고서.md` (학술 정당화)
- **선행 마스터플랜**: `documents/plans/20260427-01-DSC엔진v4-개선마스터플랜.md` (v4 — 분류 cell 완료)

---

## 0. 배경 — 한 화면 요약

v4에서 tabular × classification cell의 r=0.598 (p=1.6e-43)을 확정했다. 다음 한계 두 가지가 남아 있다.

1. **단일 instance** — "framework"라 부르려면 cell이 둘 이상 필요
2. **task-agnostic 비교 도구와의 차별점 미입증** — Metis/ydata-profiling 대비 DSC만의 가치(task-conditional)를 정량 입증할 데이터 없음

본 마스터플랜은 **tabular × regression cell**을 추가하여 두 한계를 동시 해소한다. ADR-011에서 채택한 강한 버전(메트릭 셋 자체가 cell별로 재정의)을 캡스톤 범위 내에서 구현 가능한 형태로 분해한다.

---

## 1. 작업 흐름 (Phase 의존성)

```
Phase 0  사전 등록 (본 마스터플랜)             ← 여기서 시작
   │     - regression cell 메트릭 정의식 + 가중치 고정
   │     - 데이터셋·모델·polluter 후보 확정
   ▼
Phase 1  Regression cell 인프라 구축             (재사용 75%)
   │     - 회귀 데이터셋 3개 확보 + 라이선스 검증
   │     - target_distribution_skew polluter 어댑터
   │     - 회귀 모델 5개 학습 루프
   │     - 회귀용 메트릭 9개 코드 (target_smoothness, target_distribution_quality, mutual_info_regression 포함)
   ▼
Phase 2  Regression cell 검증                    (분류 cell과 동급 검증)
   │     - 4단계 노트북 회귀 버전 실행
   │     - r(DSC ↔ R²) 측정, polluter hold-out, 모델별 r
   │     - 검증 기준: r ≥ 0.4, polluter hold-out 4/5 PASS
   ▼
Phase 3  Framework 통합                          (cell 횡단 일관성)
   │     - select_profile(data_type, task) 함수 구현
   │     - 분류 cell + 회귀 cell 결과 통합 표
   │     - cross-cell 비교 한계 명시 (cell-relative 점수)
   ▼
Phase 4  (선택) Clustering cell 또는 비교 실험   (시간 허락 시)
   │     - 옵션 A: tabular × clustering cell 추가
   │     - 옵션 B: Cleanlab + ydata-profiling 비교 실험 (20260427-10 권장사항)
   ▼
Phase 5  보고서·발표 갱신                         (외부 산출물)
         - README framework 섹션
         - 정식 결과 보고서 v5 (reports/20260427-NN)
         - 발표자료 framework 슬라이드
```

**필수 경로**: Phase 0 → 1 → 2 → 3 → 5. Phase 4는 옵션.

---

## 2. Phase별 상세

### Phase 0 — 사전 등록 (본 마스터플랜으로 완료)

본 문서 작성으로 다음 항목을 사전 고정한다 (결과 보고 후 변경 금지):

- 회귀 cell 8개 차원의 정의식 (3절)
- 회귀 cell 가중치 (3절)
- 5개 polluter 명세 (3-3절)
- 데이터셋 3개 후보 (3-4절)
- 모델 5개 (3-5절)
- 평가 메트릭 (3-6절)

### Phase 1 — Regression cell 인프라 구축

**1-1. 회귀 데이터셋 확보**

| 데이터셋 | 출처 | 행수 | 피처 (수치/범주) | 타겟 | 라이선스 |
|---|---|---:|---|---|---|
| California Housing | sklearn.datasets (StatLib 원전) | 20,640 | 8 / 0 | 주택가 중위 ($100K 단위, 회귀) | ⚠️ **형식 라이선스 미부착**. StatLib 학술 무료 공개 관행, sklearn 번들 유통. **Pace & Barry 1997 인용 필수** (Statistics and Probability Letters 33:291–297) |
| Bike Sharing (hour) | UCI ML Repository (dataset 275) | 17,389 (hour.csv) | 7 / 4 | 시간당 대여 수 (`cnt`) | ✅ CC BY 4.0 (DOI 10.24432/C5W894) |
| Wine Quality (red+white) | UCI ML Repository (dataset 186) | 6,497 (red 1,599 + white 4,898) | 11 / 0 | 품질 점수 0~10 (회귀로 처리) | ✅ CC BY 4.0 (DOI 10.24432/C56S3T, Cortez et al. 2009) |

**라이선스 검증 시점**: 2026-04-27 (3개 데이터셋 모두 본 마스터플랜 작성 시 직접 fetch 검증).

**California Housing 라이선스 caveat**: sklearn 공식 문서 및 StatLib 원전 어디에도 CC 류 명시 라이선스가 부착되지 않음. 캡스톤(학술 비상업) 용도로는 관행적 사용이 일반적이며, sklearn이 번들로 유통하는 사실 자체가 사용 허용을 시사함. 본 프로젝트는 README와 발표자료에 **Pace & Barry 1997 인용을 명시**하여 학술 출처 책임을 다한다. 형식 라이선스 부재의 잔여 위험은 인지하되, 본 캡스톤 범위에서 수용 가능 수준으로 판단.

**3개 데이터셋의 분류 cell과의 매칭** (size × type 다양성 보존):

| 분류 cell | 회귀 cell |
|---|---|
| letter (20K, numerical) | California Housing (20K, numerical) |
| Telco (7K, mixed) | Bike Sharing hour (17K, mixed) |
| SouthGerman (1K, mixed) | Wine Quality (6K, numerical) |

**검증 작업**: Phase 1 진입 시 UCI 페이지에서 데이터셋 라이선스·다운로드 URL 직접 확인(CLAUDE.md 외부 자원 검증).

**1-2. polluter 어댑터**

| polluter | 분류 cell | 회귀 cell | 변경 사항 |
|---|---|---|---|
| completeness | 결측 주입 | 동일 | 없음 |
| uniqueness | 중복 행 추가 | 동일 | 없음 |
| consistent_repr | 범주형 표현 불일치 | 동일 (범주형 있는 데이터만) | 없음 |
| feature_accuracy | 피처 값 왜곡 | 동일 | 없음 |
| **target_distribution_skew** ← class_balance 대체 | (분류) 클래스 비율 변경 | **(회귀) 타겟 분위 구간 샘플 비율 변경** — 예: 상위 25% 분위 샘플의 50% 제거 | **신규 코드** |

**신규 코드**: `dq4ai/polluters/target_distribution_skew.py`. 정의:

```python
def pollute_target_skew(df, target_col, skew_level):
    """타겟 분포의 특정 분위(상위 25%)에서 샘플의 skew_level 비율 제거.
    skew_level ∈ {0.10, 0.25, 0.50, 0.75}로 분류 cell의 4단계와 동등."""
    q75 = df[target_col].quantile(0.75)
    upper_mask = df[target_col] >= q75
    upper_df = df[upper_mask]
    n_drop = int(len(upper_df) * skew_level)
    drop_idx = upper_df.sample(n=n_drop, random_state=1).index
    return df.drop(drop_idx).reset_index(drop=True)
```

**1-3. 회귀 모델 5개**

| 분류 cell | 회귀 cell |
|---|---|
| LogisticRegression | LinearRegression |
| RandomForest (n=100) | RandomForestRegressor (n=100) |
| XGBoost (n=100) | XGBRegressor (objective='reg:squarederror', n=100) |
| SVC (linear) | SVR (kernel='linear') |
| MLP (5×100) | MLPRegressor (5×100) |

**1-4. 회귀용 메트릭 9개 코드**

3절 사전 등록 정의식대로 구현. 코드 위치: `notebooks/05_setup_and_baseline_regression.ipynb` cell N (분류 cell의 v4 엔진과 병렬 구조).

**1-5. 노트북 4종 신설**

분류 cell의 01~04와 1:1 매칭:

```
notebooks/
  05_setup_and_baseline_regression.ipynb     ← 01 회귀 버전
  06_pollution_and_dsc_regression.ipynb       ← 02
  07_training_regression.ipynb                ← 03
  08_scoreboard_regression.ipynb              ← 04
```

**Phase 1 검증 기준**:
- 3개 데이터셋 모두 baseline DSC_regression 점수 ≥ 0.7
- target_distribution_skew polluter 75% 강도에서 target_distribution_quality 점수가 baseline 대비 −0.10 이상 하락
- completeness 75% 폴루션에서 회귀 메트릭 모두 적절히 하락
- target_smoothness가 feature_accuracy 75%에서 −0.10 이상 하락 (false positive 검증: completeness에서는 target_smoothness 변화 작아야)

### Phase 2 — Regression cell 검증

분류 cell의 v4 검증과 동일 절차 수행:

**2-1. 핵심 통계**:
- Pearson r (DSC_regression ↔ R²)
- r², Spearman ρ
- 표본 크기 ≥ 80 settings × 5 model = 400 학습
- ANOVA: 등급 (A/B/C/D) × R²

**2-2. Polluter hold-out**:
- 5개 polluter 각각 hold-out → r 측정
- PASS 기준: 4/5 polluter에서 p < 0.05

**2-3. 모델별 r**: 5개 모델별로 측정. 모델 클래스 프로파일 (noise_sensitive vs tree_based) 확장 가능성 검토.

**2-4. 데이터셋별 r**: 3개 데이터셋 모두 p < 0.05 목표.

**Phase 2 검증 기준** (모두 충족 시 framework 주장 정식 성립):
| 항목 | 목표 |
|---|---|
| 통합 r (DSC_regression ↔ R²) | ≥ 0.40 (분류 cell의 0.598보다 약해도 됨, 학술적 비교 가능 수준) |
| 통합 p | < 0.001 |
| Polluter hold-out PASS | ≥ 4/5 |
| 데이터셋별 유의 | ≥ 2/3 |
| 표본 수 | ≥ 400 학습 |

### Phase 3 — Framework 통합

**3-1. `select_profile()` 함수 구현**

위치: `dq4ai/dsc_framework.py` (신규).

```python
def select_profile(data_type: str, task: str) -> dict:
    """Returns {'metrics': [...], 'weights': {...}, 'evaluator': callable}.

    Supported cells:
      - ('tabular', 'classification')  → v4 엔진
      - ('tabular', 'regression')      → v5 회귀 엔진
    """
    key = (data_type, task)
    if key == ('tabular', 'classification'):
        return TABULAR_CLASSIFICATION_PROFILE  # ADR-009 사전 등록
    elif key == ('tabular', 'regression'):
        return TABULAR_REGRESSION_PROFILE       # 본 문서 3절 사전 등록
    else:
        raise NotImplementedError(f"Cell {key} not yet supported")
```

**3-2. 통합 결과 보고**

| Cell | 데이터셋 수 | Polluter 수 | 표본 | r | r² | Spearman ρ |
|---|---:|---:|---:|---:|---:|---:|
| tabular × classification (v4) | 3 | 5 | 435 | 0.598 | 0.358 | 0.628 |
| tabular × regression (v5) | 3 | 5 | (Phase 2 결과로 채움) | (?) | (?) | (?) |

**3-3. Cross-cell 비교 한계 명시**

- 두 cell의 점수는 **독립 스케일** (Heinrich 2018 R1 만족하는 cell-relative)
- DSC_classification(D) = 0.7과 DSC_regression(D) = 0.7은 의미 다름 — 절대값 비교 부정당
- cell 내 데이터셋 ranking은 비교 가능, cell 간은 한계로 명시

### Phase 4 — (선택) Clustering cell 또는 비교 실험

**시간 잔량 따라 둘 중 하나 선택**:

**옵션 A — Clustering cell 추가**:
- 추가 작업: tabular × clustering 메트릭 정의 (silhouette/Davies-Bouldin 기반 task용 차원), 5개 clustering 모델 (K-Means, DBSCAN, Agglomerative, GMM, Spectral), 평가 메트릭 (silhouette score)
- 비용: 2주
- 효과: framework instance n=3, 더 강한 주장

**옵션 B — Cleanlab + ydata-profiling 비교 실험** (`20260427-10` 권장사항):
- 분류 cell 435 표본에 두 도구 적용 → 점수 산출 → DSC와 ranking 일치도 측정
- 비용: 1주
- 효과: 학술적 차별화 정량 입증 (수렴 타당도 / 차별 타당도)

**권고**: 시간 부족하면 옵션 B(1주). 시간 충분하면 옵션 A(2주).

### Phase 5 — 보고서·발표 갱신

**5-1. README**:
- "Framework" 섹션 추가 (v4의 분류 cell + v5의 회귀 cell 결과 통합 표)
- 학술 포지셔닝 한 줄: *"a task-conditional, fitness-for-use composite DQ score validated empirically against downstream ML performance across heterogeneous classifiers and regressors"*

**5-2. 정식 결과 보고서**:
- `documents/reports/20260428-NN-v5-Framework-정식결과.md` (Phase 2 완료 후)
- 분류·회귀 cell 비교 표, polluter hold-out 통합, 학술 인용 위치

**5-3. 발표자료**:
- Framework 슬라이드 (cell 매트릭스, select_profile 다이어그램)
- 두 cell 결과 비교 슬라이드
- Limitations 슬라이드 — cross-cell 비교 한계, 멀티모달은 future work

---

## 3. 사전 등록 (Pre-registration) — Regression Cell 정의

본 절의 정의식·가중치는 본 문서 작성 시점(2026-04-27)에 확정된다. **Phase 2 결과를 보고 변경하지 않음** (ADR-009·ADR-011 사전 등록 원칙 계승, F1 순환 논증 회피).

### 3-1. 9개 차원 정의식 (회귀 cell)

| # | 차원 | 분류 cell 정의 | **회귀 cell 정의 (사전 등록)** | 변경? |
|---|---|---|---|:---:|
| 1 | `completeness` | 결측 비율 1 − (n_missing / n_total) | **동일** | — |
| 2 | `uniqueness` | 1 − (n_dup / n_total) | **동일** | — |
| 3 | `validity` | 1 − (n_format_error / n_total) | **동일** | — |
| 4 | `consistency` | 1 − (n_inconsistent_repr / n_total) (범주형 한정) | **동일** | — |
| 5 | `outlier_ratio` | 1 − IQR-기반 outlier 비율 | **동일** | — |
| 6 | `class_balance` (분류) → **`target_distribution_quality`** (회귀) | 클래스 분포 엔트로피 / log(n_class) | **타겟을 10 bin으로 분할 후 entropy / log(10)** | ✅ |
| 7 | `feature_correlation` | 1 − 평균 피처 간 \|상관\| | **동일** | — |
| 8 | `label_consistency` (분류) → **`target_smoothness`** (회귀) | k=5 KNN 라벨 동일 비율 | **k=5 KNN 이웃 target과 자기 target의 표준화 거리. score = 1 / (1 + mean_distance)** | ✅ |
| 9 | `feature_informativeness` (분류) → **`feature_informativeness_reg`** (회귀) | mutual_info_classif 평균 | **mutual_info_regression 평균. 정규화: 관측 최댓값 기준** | ✅ |

**총 9 항목** (validity, consistency 포함). 회귀 cell도 분류 cell과 동일 슬롯 구조 — 호환성 유지.

### 3-2. 가중치 (회귀 cell, default profile)

```
회귀 default profile (합 = 1.00):
  completeness:                 0.20
  uniqueness:                   0.15
  validity:                     0.05
  consistency:                  0.10
  outlier_ratio:                0.05
  target_distribution_quality:  0.10  ← class_balance 슬롯
  feature_correlation:          0.05
  target_smoothness:            0.20  ← label_consistency 슬롯
  feature_informativeness_reg:  0.10  ← feature_informativeness 슬롯
```

**근거** (분류 cell v4와 동일 가중치 유지):
- `target_smoothness 0.20` = 분류 cell `label_consistency 0.20`의 회귀 등가물. ML 학습 가능성 직접 측정 — 회귀에선 이웃 간 target 거리가 직접 R²에 반영
- `target_distribution_quality 0.10` = 분류 cell `class_balance 0.10` 동급. 타겟 분포 균형성
- 나머지 6개 차원: 분류와 동일 비중 (완전 호환성 검증 가능)

### 3-3. 5개 polluter (회귀 cell)

| polluter | 강도 | 분류 cell 코드 재사용 |
|---|---|:---:|
| completeness | 0.10 / 0.25 / 0.50 / 0.75 | ✅ 그대로 |
| uniqueness | factor 1.5 / 2.0 / 3.0 / 4.0 | ✅ 그대로 |
| consistent_repr | 0.10 / 0.25 / 0.50 / 0.75 (범주형 있는 Bike Sharing 한정) | ✅ 그대로 |
| feature_accuracy | 0.10 / 0.25 / 0.50 / 0.75 | ✅ 그대로 |
| **target_distribution_skew** | 0.10 / 0.25 / 0.50 / 0.75 | ❌ **신규 코드** (1-2절) |

### 3-4. 데이터셋 3개 (1-1절 참조)

- California Housing (sklearn) — 20,640 × 8 numerical
- Bike Sharing hour (UCI) — 17,379 × 11 mixed
- Wine Quality (UCI) — 6,497 × 11 numerical

**예상 표본 수**:
- numerical-only 2개: 4 polluter × 4 level = 16 + 1 baseline = 17 settings × 5 model = 85
- mixed 1개 (Bike Sharing): 5 polluter × 4 level = 20 + 1 baseline = 21 settings × 5 model = 105
- 총합: 85 + 85 + 105 = **275 학습** (분류 cell 435보다 적으나 framework 주장에 충분)

**참고**: target_distribution_skew는 모든 데이터셋에 적용 → 5 polluter × 4 level + 1 baseline = 21 × 5 = 105/dataset. 위 계산은 numerical-only에서 consistent_repr를 뺀 것. **재계산**: 3 dataset × 21 setting × 5 model = **315 학습** (소수 데이터셋 별 setting 차이 무시 시).

### 3-5. 모델 5개 (1-3절 참조)

LinearRegression / RandomForestRegressor / XGBRegressor / SVR / MLPRegressor.

### 3-6. 평가 메트릭

- **Primary**: R² (음수 R²는 0으로 clip — 모델이 평균 예측보다 나쁨)
- **Secondary**: MAE, RMSE
- **DSC ↔ R²의 Pearson r**가 framework 검증의 핵심 지표

### 3-7. 보조 지표 — Baseline-relative Degradation Index (ADR-012, 사전 등록)

`compute_dsc_regression()` 결과의 9개 메트릭과 절대 점수에 더해, **clean baseline 대비 손상 정도**를 측정하는 보조 지표를 추가.

**정의**:

```
m_deg = max(0, 1 − polluted[m] / clean[m])         # per-metric ∈ [0, 1]
overall_degradation = Σ wᵢ · mᵢ_deg                 # weighted ∈ [0, 1]
preservation_score  = (1 − overall_degradation) × 100   # ∈ [0, 100]
```

가중치는 sect 3-2의 default profile 그대로 사용.

**역할 분리**:

- `DSC_absolute` — 데이터셋의 본질적 품질 (cell-relative 절대 점수)
- `overall_degradation` / `preservation_score` — clean baseline 대비 손상 (cross-dataset 비교 가능)

**도입 근거**: 약신호 데이터셋(예: Wine Quality, baseline TDQ=0.55)에서 절대 Δ는 floor effect로 작아도 상대 degradation은 다른 데이터셋과 비교 가능한 단위로 산출됨. ADR-012 sect 3 검증 결과 표 참조.

**Phase 2에서의 사용**:
- Primary: `r(DSC_absolute, R²)` (분류 cell v4와 동일 분석 형식 유지)
- Secondary: `r(preservation_score, R²)` (약신호 데이터셋의 신호 회복)
- 두 r 모두 보고하여 학술적 차별점 명시

**Cross-cell 비교 한계**: degradation 지표도 cell-relative. 회귀 cell preservation=90과 분류 cell preservation=90의 직접 비교는 부정당 (ADR-011 위험 2와 동일).

### 3-8. Split·평가 원칙 (분류 cell과 동일)

- train 80% / test 20%, `random_state=1`
- 회귀의 stratified split: target 5분위로 quantile bin 만들고 stratify
- **train만 오염, test는 clean** (split-first 원칙, ADR-009·feedback_ml_pipeline_process.md 준수)
- 자동 leakage 검증 셀: train과 test 인덱스 교집합 0건 보장

---

## 4. 검증 원칙 (CLAUDE.md 준수)

1. **외부 자원 검증**:
   - UCI 데이터셋 다운로드 URL·라이선스 Phase 1 진입 시 직접 확인
   - sklearn.datasets.fetch_california_housing 동작 검증 (Colab에서)
   - mutual_info_regression scipy/sklearn 버전 호환성 확인

2. **사전 등록 준수** (ADR-011 위험1 대응):
   - 본 문서 3절의 정의식·가중치는 Phase 2 결과를 보고 변경 금지
   - 가중치 미세조정이 필요하면 사전 등록을 무효화하고 ADR-012로 신규 의사결정 기록

3. **Split-first + train-only 오염 + clean test** (feedback_ml_pipeline_process.md):
   - 회귀 cell도 분류 cell과 동일 원칙 적용
   - train_polluted/, test_clean/ 디렉토리 구조 동일하게 유지 (regression 디렉토리 분리)

4. **단위 검증** (feedback_verify_before_deliver.md):
   - 메트릭 코드는 실제 California Housing 데이터에서 baseline 점수 산출 후 합리성 점검
   - polluter는 실제 데이터에 적용해서 분포 변화 시각화 후 다음 단계 진입

5. **재현성**: 모든 polluter, KNN, MLP는 `random_state=1` 고정.

---

## 5. 산출물 인덱스

| Phase | 파일/문서 |
|---|---|
| 0 | 본 마스터플랜, ADR-011 |
| 1 | `dq4ai/polluters/target_distribution_skew.py`, `notebooks/05~08_*regression*.ipynb`, `data/raw/california_housing.csv` 등 3개 데이터셋, `data/train_polluted_regression/`, `data/test_clean_regression/` |
| 2 | `results/dsc_scores_regression.csv`, `results/model_performance_regression.csv`, `results/merged_results_regression.csv`, `results/08_execution_log.md` |
| 3 | `dq4ai/dsc_framework.py` (`select_profile`), `results/framework_summary.csv` (cell × dataset 통합 표) |
| 4 | (옵션) `notebooks/09_clustering*.ipynb` 또는 `notebooks/_dev/cleanlab_comparison.py` |
| 5 | `README.md` framework 섹션, `documents/reports/20260428-NN-v5-Framework-정식결과.md`, 발표자료 |

---

## 6. 일정 추정

| Phase | 작업량 | 추정 기간 |
|---|---|---|
| 0 | 사전 등록 (본 문서) | 완료 |
| 1 | 인프라 구축 | 5~7일 |
| 2 | 검증 (노트북 4종 실행 + 분석) | 3~5일 |
| 3 | Framework 통합 | 2~3일 |
| 4 (옵션) | clustering 또는 비교 실험 | 7~14일 |
| 5 | 보고서·발표 갱신 | 3~5일 |

**최소 경로(Phase 0~3, 5)**: 13~20일.
**전체(Phase 4 포함)**: 20~34일.

캡스톤 발표 일정에 맞춰 Phase 4 옵션 결정.

---

## 7. 캡스톤 범위 (in/out of scope)

### In scope (이 마스터플랜)

- tabular × classification cell (v4, 완료)
- tabular × regression cell (Phase 1~3)
- (옵션) tabular × clustering cell **또는** 분류 cell의 Cleanlab/ydata-profiling 비교 실험 (Phase 4)
- Framework 통합·문서화 (Phase 3·5)

### Out of scope (Limitations에 명시)

- text × * cells (BERT 학습 비용 + 텍스트 polluter 신규 설계 부담)
- image × * cells (CIFAR-10-C 활용 가능하나 모델·메트릭 인프라 신규)
- audio × * cells
- time series, graph 등 비정형 task
- DSC 점수의 multi-modal 통합 (cross-modal 비교 가능성)

후속 연구로 명시. `20260427-10` 레퍼런스 보고서 6절(멀티모달 벤치마크)이 후속 연구 출발점.

---

## 8. 즉시 다음 작업

본 마스터플랜 확정 후:

1. UCI 회귀 데이터셋 3개 페이지 직접 fetch — 라이선스·다운로드 URL 검증
2. sklearn.datasets.fetch_california_housing Colab 동작 테스트
3. `dq4ai/polluters/target_distribution_skew.py` 초안 작성
4. `notebooks/05_setup_and_baseline_regression.ipynb` 셀 구조 설계 (분류 cell 01의 구조 복제)

각 작업의 첫 시도 결과를 `documents/progress/` 또는 `_dev/` 노트에 기록하고 Phase 1 정식 진입 판단.

---

**문서 끝.**
