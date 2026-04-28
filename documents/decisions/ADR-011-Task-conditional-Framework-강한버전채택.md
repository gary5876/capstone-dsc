# ADR-011: DSC를 강한 버전 Task-conditional Framework로 재정의

- **일자**: 2026-04-27
- **상태**: 확정 (구현 시작 전 사전 등록)
- **선행**: ADR-009 (v4 절대품질 지표), ADR-010 (Telco 한계), `documents/reports/20260427-04-v4-정식결과확정.md` (v4 결과 r=0.598), `documents/reports/20260427-10-레퍼런스조사보고서.md` (학술 정당화)
- **후속**: `documents/plans/20260427-02-DSC-Framework-v5-마스터플랜.md` (구현 계획)

---

## 0. 한 줄 결정

> DSC를 "단일 (data_type, task) 점수"가 아니라 **(data_type, task) → {metric_set, weights}** 매핑을 가진 framework로 재정의한다. 같은 차원이라도 cell마다 정의식이 달라지는 **강한 버전**을 채택한다.

---

## 1. 배경

### 1-1. v4 정식 결과의 학술적 위치

| 항목 | 값 |
|---|---|
| Pearson r (DSC ↔ F1 macro) | 0.598 (p=1.6e-43, n=435) |
| 표본 단위 | 87 (3 dataset × 5 polluter × 4 level + baseline) × 5 model |
| 적용 범위 | **tabular × binary/multi-class classification 한정** |

선행 연구 비교 (`20260427-10` 레퍼런스 조사 결과):

- Mohammed et al. 2022/2025 (DQ4AI 모태 논문, *Information Systems* 132): 6 DQ 차원 × 19 ML 알고리즘 × {분류/회귀/클러스터링} 폭넓게 다루나 **단일 DQ 점수 ↔ ML 성능 통합 r은 미보고**. 차원별 영향만.
- Cleanlab 논문 (Northcutt et al. 2021, JAIR): label 단일 축 + ML 연계.
- Heinrich et al. 2018 (JDIQ Vol 9-2): DQ 메트릭 5요건 R1~R5. DSC가 R1·R4를 만족하는 형태로 설계됨.
- ydata-profiling/Great Expectations/Metis: **모두 task-agnostic**. DSC가 task-conditional이라는 점이 차별화 가능 지점.

### 1-2. 현재 DSC의 한계 (자체 진단)

1. **단일 (data_type, task) 인스턴스만 존재** — tabular × classification 1개 cell. "framework"라 부르기엔 instance 수 부족 (n=1).
2. **Class_balance 같은 차원은 분류 의존적** — 회귀로 가면 의미 없음. 같은 5차원으로 다른 task를 평가하면 의미 왜곡.
3. **v4의 "모델 프로파일" (noise_sensitive / tree_based)은 *모델* 단위 약한 task-conditioning** — 진짜 필요한 것은 *task* 단위.
4. **"왜 이 5차원인가"의 정당화** — Wang & Strong 1996 4범주 매핑, Pipino 2002 측정 원칙 등 학술 근거를 task별로 명시할 인프라 없음.

---

## 2. 결정 — 강한 버전 채택

### 2-1. 정의

DSC는 다음 형태를 갖는 **점수 framework**다:

```
DSC(D, T) = Σᵢ wᵢ(T) · sᵢ(D, T)
```

- `D`: 데이터셋 (data_type ∈ {tabular_numeric, tabular_mixed, ...})
- `T`: ML task (∈ {classification, regression, clustering, ...})
- `sᵢ(D, T)`: i번째 차원의 점수. **차원 이름이 같아도 (D,T)에 따라 정의식이 다름**.
- `wᵢ(T)`: T에 대한 i번째 차원 가중치. 차원이 T에 정의되지 않으면 0.

`select_profile(data_type, task) → (metric_set, weights)` 함수가 cell마다 다른 정의식과 가중치를 반환한다.

### 2-2. 강한 버전 vs 약한 버전 비교

| 측면 | 약한 버전 | **강한 버전 (채택)** |
|---|---|---|
| 가중치 | (D,T)별로 다름 | (D,T)별로 다름 |
| 메트릭 셋 | 동일 5차원 | **(D,T)별로 메트릭 자체가 다름** |
| 정의식 | 동일 | **차원 이름이 같아도 (D,T)별 정의식 다름** (예: `class_balance`(분류) vs `target_distribution`(회귀)) |
| 학술적 의미 | 가중치 튜닝의 확장 | "framework"로 인정 가능 |
| 구현 비용 | 낮음 (1주) | 중간 (cell당 2~3주) |
| 캡스톤 차별화 | 약함 | 강함 |
| reviewer 비판 | "v4 가중치 튜닝의 연장" | (메트릭 정의 자체가 task에 맞게 재정의됨) |

### 2-3. 학술 근거 (검증된 인용)

`20260427-10`에서 직접 검증한 인용 사용:

| 근거 | 출처 | DSC framework가 활용하는 부분 |
|---|---|---|
| Fitness for use | Juran 1951 → Wang & Strong 1996 (JMIS 12-4) | "품질은 사용 목적에 따라 정의된다" — 강한 버전의 1차 정당화 |
| Contextual DQ category | Wang & Strong 1996 | 4범주(intrinsic/contextual/representational/accessibility) 중 contextual 축 = task 의존 |
| Task-conditional 측정 원칙 | Pipino, Lee, Wang 2002 (CACM 45-4) | "객관 지표 + 과제 연계" 3원칙 — DSC는 객관 측정값을 task에 맞게 결합 |
| 메트릭 5요건 | Heinrich et al. 2018 (JDIQ 9-2) | R1(min/max 존재) + R4(건전한 집계) 만족하는 형태로 설계 |
| 차원별 영향이 task별로 다름 | Mohammed et al. 2022/2025 (Information Systems 132) | 동일 polluter가 분류/회귀/클러스터링에서 매우 다르게 작용 → task 통합 framework가 부적절함을 실증 |

### 2-4. 캡스톤 범위 내 적용

| Cell | 상태 | 캡스톤 우선순위 |
|---|---|---|
| tabular × binary/multi-class classification | ✅ 완료 (v4) | (유지) |
| **tabular × regression** | 🔨 **신규 — Phase 1** | **1순위** |
| tabular × clustering | 📋 설계만 | 2순위 (시간 허락 시) |
| text/image/audio × * | ❌ 캡스톤 범위 외 | 후속 연구 |

**캡스톤은 tabular 안에서 2 cell (분류 + 회귀)로 framework 주장이 성립**. 멀티모달은 후속 연구 (Limitations에 명시).

---

## 3. 대안 검토

| 대안 | 검토 결과 | 기각 이유 |
|---|---|---|
| **A. 현상 유지** (v4 단일 cell) | r=0.598 그대로 발표 | "허접" 우려 미해결, 차별화 약함 |
| **B. 약한 버전** (메트릭 동일, 가중치만 task별) | 1주 구현 | "v4 모델 프로파일의 확장"으로 보일 위험. framework 주장 약함 |
| **C. 강한 버전, tabular 2 cell** | **채택** | framework 주장 성립 + 캡스톤 기간 내 가능 |
| D. 강한 버전, 멀티모달까지 4 cell | (기각) | 캡스톤 기간 내 깊이 부족, "넓고 얕다" 위험 |
| E. 멀티모달 + 약한 버전 | (기각) | D의 단점 + B의 단점 모두 포함 |

**채택**: C (강한 버전, tabular 2 cell. clustering은 옵션).

---

## 4. 위험과 대응

### 위험 1 — `select_profile()`의 자의성

> reviewer: "왜 회귀에 `target_distribution_skew`이고 분류에 `class_balance`인가? 임의 선택 아닌가?"

**대응**:
- 각 차원 정의식을 Wang & Strong 4범주에 명시적으로 매핑한 표를 v5 마스터플랜에 사전 등록
- Pipino 2002 측정 원칙(객관 지표 + 과제 연계) 인용으로 정당화
- 데이터 기반 weight tuning 금지 — 사전 등록한 weights를 결과 보고 변경하지 않음 (ADR-009의 사전 등록 원칙 계승)

### 위험 2 — Cross-cell 비교 불가

> DSC=0.7 (분류 cell) vs DSC=0.7 (회귀 cell)이 같은 의미인가?

**대응**:
- 점수의 의미를 **cell-relative**로 명시. 동일 cell 내 데이터셋 ranking·등급은 비교 가능, cell 간 절대값 비교는 부정당.
- 이는 Heinrich et al. 2018 R1(스케일 정의) 만족 — 각 cell이 자체 스케일을 가짐.
- 발표/논문 표기에서 항상 `DSC_classification(D)`, `DSC_regression(D)` 형태로 cell 명시.

### 위험 3 — 표본 부족

> regression cell의 표본이 분류 cell만큼 안 나오면 r 비교가 무의미

**대응**:
- regression cell 표본 목표 ≥ 80 (3 dataset × 5 polluter × 4 level + baseline = 63 settings × 5 model = 315 학습)
- 분류 cell의 87 settings × 5 = 435건과 동급 보장
- letter/Telco/SouthGerman과 통계적 동등 데이터셋 선정 (large-numerical / mixed-medium / small-mixed 매칭)

### 위험 4 — `class_balance` polluter의 회귀 어댑터

> DQ4AI의 class_balance polluter는 분류 라벨 비율을 변경. 회귀에선 의미 없음.

**대응**:
- 회귀 cell에서는 class_balance polluter를 **`target_distribution_skew` polluter**로 치환 (분포 꼬리 샘플 제거 또는 특정 구간 oversampling)
- 다른 4개 polluter (completeness, uniqueness, consistent_repr, feature_accuracy)는 회귀에 그대로 적용 가능
- 동등성: 분류 5 polluter ↔ 회귀 5 polluter (1:1 대응)

### 위험 5 — feature_correlation, validity 등 약신호 차원의 일관성

> v4에서 weight 0.05인 약신호 차원들은 cell마다 정의 부담 대비 효과 미미

**대응**:
- 약신호 차원도 cell별 정의는 유지하되, weight는 0.03~0.05로 동일하게 작게 부여
- 핵심 변별 차원 (label_consistency / feature_informativeness / class_balance↔target_distribution) 에 weight 집중

### 위험 6 — 시간 압박 (캡스톤 기한)

> v4 정식 확정이 2026-04-27. regression cell 구현 + 검증 + 보고에 충분한 시간 있나?

**대응**:
- 인프라 75% 재사용 (4단계 노트북 구조, 모델 학습 루프, 평가 파이프라인 동일)
- 회귀용 새 코드는 polluter target_distribution_skew + 모델 회귀 버전 5개 + 메트릭 회귀용 정의식 8개
- v5 마스터플랜에서 Phase별 시간 추정 명시. 시간 부족 시 clustering cell 포기, 분류+회귀 2 cell로 framework 주장 유지.

---

## 5. 영향받는 파일/문서

### 즉시 (이 ADR로 발생)

- `documents/decisions/ADR-011-Task-conditional-Framework-강한버전채택.md` (이 문서)
- `documents/plans/20260427-02-DSC-Framework-v5-마스터플랜.md` (분기 — 다음 작성)

### Phase 1 진입 시

- `notebooks/05_setup_regression.ipynb` (신규 — regression baseline)
- `notebooks/06_pollution_regression.ipynb` (신규)
- `notebooks/07_training_regression.ipynb` (신규)
- `notebooks/08_scoreboard_regression.ipynb` (신규)
- `dq4ai/polluters/target_distribution_skew.py` (신규 — class_balance 회귀 어댑터)
- `data/raw/` 회귀 데이터셋 3개 추가
- `results/dsc_scores_regression.csv`, `model_performance_regression.csv` 등 분리

### 발표/문서 갱신 (Phase 5 산출물)

- `README.md` — framework 정의 + 2 cell 결과 통합 섹션
- `documents/reports/20260427-04-v4-정식결과확정.md` (회귀 cell 결과 추가 또는 별도 보고서)
- 발표자료 — framework 슬라이드 추가, classification + regression 두 cell 결과 비교

---

## 6. 검증 원칙 (CLAUDE.md 준수)

1. **각 cell의 메트릭 정의식은 사전 등록 후 결과 따라 변경 금지** — ADR-009의 사전 등록 원칙 계승
2. **regression cell도 split-first + train-only 오염 + clean test 원칙 동일 적용** — `feedback_ml_pipeline_process.md` 메모리 준수
3. **데이터 기반 weight 학습 금지** — Phase별 검증 결과를 보고 가중치 미세조정 안 함. 사전 등록 값 그대로 발표
4. **외부 데이터셋·도구·라이브러리 사용 전 실재·라이선스 확인** — `CLAUDE.md` 준수. UCI 데이터셋도 라이선스 명시 후 사용

---

## 7. 결정 후 즉시 다음 작업

본 ADR 확정 후 분기되는 작업:

1. `documents/plans/20260427-02-DSC-Framework-v5-마스터플랜.md` 작성 (이 ADR 직후)
2. 마스터플랜 내에서 regression cell의 8개 차원 정의식 사전 등록
3. UCI 회귀 데이터셋 3개 후보 검토 + 라이선스 확인
4. Phase 1 진입 (regression cell 인프라 구축)

---

**문서 끝.**
