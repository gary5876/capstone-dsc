# aidq-platform v5 통합 가이드 — 변경점 중심

## 한 줄 요약
- v5는 기존 v3.2 기반 DSC 엔진과 달리 **task-aware framework**로 전환됩니다.
- 핵심 변경: `dsc_framework` 도입, `task` 자동 감지, 분류/회귀 별 9개 지표, `value_accuracy` 삭제, 결과 JSON에 `task` 추가.
- 이 문서는 **통합 시 반드시 반영해야 하는 변경점만** 다룹니다.

---

## 1. 핵심 구조 변경

### 1-1. 엔진 호출 변경
- 기존: webplatform 내부 또는 패키지에 `dsc_engine` 형태의 단일 엔진 코드가 포함됨.
- 변경: `dsc_framework` 패키지를 가져와 `compute_dsc()`를 호출.
- webplatform은 이제 `from dsc_framework import compute_dsc, auto_detect_columns`를 사용할 수 있어야 함.

### 1-2. task 자동 감지
- 기존: 모든 데이터에 분류 지표 8개를 적용하는 단일 흐름.
- 변경: `auto_detect_columns(df)`가 `target_col, numerical_cols, categorical_cols, task` 4-tuple을 반환.
- `task`는 `classification` 또는 `regression`.
- `task`는 `compute_dsc(..., task=task)` 호출에 그대로 전달되어야 함.

### 1-3. 결과 메타 정보 강화
- `compute_dsc()`는 반환 결과에 `task`와 `data_type`을 포함.
- `task`가 없는 기존 결과는 `legacy_v32`로 처리해야 함.
- DB JSON 컬럼(`resultDetail`)에 `task`를 그대로 저장하면 되므로 스키마 변경은 불필요.

---

## 2. 지표 구성 변경

### 2-1. classification cell (분류)
- `value_accuracy` 제거.
- 신규 지표: `label_consistency`, `feature_informativeness`.
- 가중치 재분배: `class_balance` 0.05→0.10.
- 최종 9개 지표:
  - `completeness`
  - `uniqueness`
  - `validity`
  - `consistency`
  - `outlier_ratio`
  - `class_balance`
  - `feature_correlation`
  - `label_consistency`
  - `feature_informativeness`

### 2-2. regression cell (회귀)
- 회귀 전용 지표 3개 신설:
  - `target_distribution_quality`
  - `target_smoothness`
  - `feature_informativeness_reg`
- 공통 지표 6개 유지:
  - `completeness`, `uniqueness`, `validity`, `consistency`, `outlier_ratio`, `feature_correlation`
- 회귀는 별도 9개 지표로 진단되며, 분류와 다른 metric set을 사용.
- 최종 9개 지표:
  - `completeness`
  - `uniqueness`
  - `validity`
  - `consistency`
  - `outlier_ratio`
  - `target_distribution_quality`
  - `feature_correlation`
  - `target_smoothness`
  - `feature_informativeness_reg`

### 2-3. image cell 참고 (후속)
- v5에는 image cell도 포함되지만, webplatform 통합 시점에서는 **후속 검토 항목**으로 둬도 무방.
- image cell 주요 지표 10개:
  - `completeness_image`
  - `uniqueness`
  - `validity`
  - `consistency`
  - `outlier_ratio`
  - `class_balance`
  - `feature_correlation`
  - `label_consistency`
  - `feature_informativeness`
  - `sample_quality_image`

### 2-4. legacy v3.2 결과
- 기존 v3.2 진단 결과는 `task` 필드가 없음.
- 조회 시 `legacy_v32` 플래그를 달아 `value_accuracy` 기반 8개 지표로 처리해야 함.
- 재진단은 v5 엔진으로 진행.

---

## 3. 통합 시 꼭 수정해야 할 영역

### 3-1. Python Worker / 백엔드
- `auto_detect_columns(df)` 호출이 `task`까지 반환하도록 변경.
- `compute_dsc(..., task=task)`로 호출.
- 결과 JSON에 `task`와 `data_type` 포함.
- `task` 강제 지정이 필요하다면 메시지에서 `task` 값을 받아 override 가능하게 구현.

### 3-2. 프론트엔드
- 결과 페이지에 `task` 배지 표시: `분류`, `회귀`, `legacy_v32`.
- 진단 지표 리스트를 `task`별로 분기.
  - 분류: 9개 지표
  - 회귀: 9개 지표
- `value_accuracy` 항목은 제거.
- 회귀 업로드 시 회귀용 지표가 노출되는지 확인.
- 기존 v3.2 결과는 legacy 모드로 보여야 함.

### 3-3. 빌드·배포
- `dsc_framework`가 webplatform Python 환경에 설치 또는 경로 설정되어야 함.
- Docker/Dockerfile에서 `PYTHONPATH` 또는 패키지 복사 경로를 설정하여 `from dsc_framework import compute_dsc`가 실패하지 않도록 함.

---

## 4. 수정해야 할 코드 위치
- `dsc_framework/router.py`: `compute_dsc()` 통합 진입점, `task` 자동 감지 및 profile 선택.
- `dsc_framework/column_detection.py`: `auto_detect_columns()`가 `task`를 반환하도록 확장.
- `dsc_framework/classification_cell.py`: `value_accuracy` 삭제, `label_consistency`·`feature_informativeness` 반영.
- `dsc_framework/regression_cell.py`: 회귀 전용 metric 및 `DEFAULT_WEIGHTS_REGRESSION` 정의.
- webplatform worker result 생성 코드: `task` 포함 JSON 전송.
- 프론트 metric 렌더링 로직: task별 키셋 분기.

---

## 5. 검증 체크리스트
- [ ] 분류 CSV 업로드 시 `task=='classification'` 확인.
- [ ] 분류 결과에 `value_accuracy`가 없고 `label_consistency`, `feature_informativeness`가 존재.
- [ ] 회귀 CSV 업로드 시 `task=='regression'` 확인.
- [ ] 회귀 결과에 `target_distribution_quality`, `target_smoothness`, `feature_informativeness_reg`가 존재.
- [ ] 결과 JSON에 `task`/`data_type` 키가 포함됨.
- [ ] legacy v3.2 기록은 `legacy_v32`로 표시됨.
- [ ] `dsc_framework` import 오류 없음.
- [ ] 프론트가 task별 올바른 항목을 노출.

---

## 6. 주의 사항
- 이미지 cell은 본 시점에서는 **선택적 후속**으로 유지.
- 회귀/분류 자동 감지가 잘못될 경우 `column_detection.detect_task()`의 `num_unique_threshold` 조정 필요.
- `task`와 metric set이 섞이면 `KeyError`가 발생하므로, 프론트와 백엔드 모두 같은 task 기준으로 처리해야 함.

---

## 7. 요약
- v5 통합의 핵심은 **task-aware 진단**과 **분류/회귀 별 지표 분리**입니다.
- webplatform에는 `dsc_framework` import, `task` 전달, 결과 metadata 저장, task별 UI 분기만 반영하면 됩니다.
- 기존 v3.2 결과는 legacy 모드로 유지하고, 새 진단은 v5 엔진으로 수행하세요.
