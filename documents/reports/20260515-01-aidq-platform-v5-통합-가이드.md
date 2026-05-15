# aidq-platform v5 통합 가이드 — 2026-05-08 이후 변경점

## 한 줄 요약
- 2026-05-08 가이드 이후, 회귀 cell과 이미지 cell이 v5 framework에 추가됨.
- 핵심 변경: `task`에 `regression`/`image` 추가, `data_type` 분기, 회귀/이미지 metric set 지원.
- 이 문서는 **2026-05-08 가이드 이후의 추가 변경점만** 다룹니다.

---

## 1. 2026-05-08 이후 추가된 핵심 구조 변경

### 1-1. task 확장
- 기존: `task`는 `classification`만 지원.
- 추가: `regression`과 `image` task 지원.
- `auto_detect_columns(df)`는 이제 `task`를 `classification`, `regression`, `image` 중 하나로 반환.

### 1-2. data_type 분기
- 기존: 모든 데이터가 `tabular`.
- 추가: `image` 데이터셋 감지 시 `data_type='image'`.
- `router.py`에서 `(data_type, task)`에 따라 cell 선택: `('tabular', 'classification')` → classification_cell, `('tabular', 'regression')` → regression_cell, `('image', 'classification')` → image_cell.

### 1-3. 결과 메타 정보 강화 (추가)
- `compute_dsc()` 반환에 `data_type` 키 추가 (기존 `task` 외).
- 결과 JSON에 `data_type` 포함.

---

## 2. 2026-05-08 이후 추가된 지표 구성 변경

### 2-1. regression cell (회귀) — 신규 추가
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

### 2-2. image cell — 신규 추가
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

---

## 3. 2026-05-08 이후 추가 통합 시 꼭 수정해야 할 영역

### 3-1. Python Worker / 백엔드 — 추가
- `task`가 `regression` 또는 `image`일 경우 지원.
- `data_type`이 `image`일 경우 image 데이터셋 처리 로직 추가 (예: PyTorch Dataset 변환).

### 3-2. 프론트엔드 — 추가
- 결과 페이지에 `data_type` 배지 표시 (tabular/image).
- 진단 지표 리스트를 `task`별로 분기 (기존 classification 외 regression/image 추가).
- image 업로드 시 image cell 지표가 노출되는지 확인.

### 3-3. 빌드·배포 — 변경 없음
- 2026-05-08 가이드 그대로 유지.

---

## 4. 2026-05-08 이후 추가 수정해야 할 코드 위치
- `dsc_framework/regression_cell.py`: 회귀 전용 metric 및 `DEFAULT_WEIGHTS_REGRESSION` 정의 — 신규.
- `dsc_framework/image_cell.py`: 이미지 cell metric 및 weights 정의 — 신규.
- `dsc_framework/data_type_detection.py`: DataFrame vs ImageDataset 감지 — 신규.
- `dsc_framework/router.py`: data_type까지 분기 추가 — 업데이트.
- webplatform worker result 생성 코드: `data_type` 포함 JSON 전송 — 추가.
- 프론트 metric 렌더링 로직: task별 키셋 분기 (regression/image 추가) — 업데이트.

---

## 5. 2026-05-08 이후 추가 검증 체크리스트
- [ ] 회귀 CSV 업로드 시 `task=='regression'` 확인.
- [ ] 회귀 결과에 `target_distribution_quality`, `target_smoothness`, `feature_informativeness_reg`가 존재.
- [ ] 이미지 데이터셋 업로드 시 `data_type=='image'`, `task=='classification'` 확인.
- [ ] 이미지 결과에 `completeness_image`, `sample_quality_image` 등 10개 지표 존재.
- [ ] 결과 JSON에 `data_type` 키가 포함됨.

---

## 6. 2026-05-08 이후 추가 주의 사항
- 회귀 cell과 이미지 cell은 본 통합 변경점에 포함됩니다. `documents/decisions/ADR-014-이미지-cell-사전등록.md`와 관련 progress 문서를 참고하여 설계해야 합니다.
- `data_type`이 `image`일 경우, worker에서 PyTorch Dataset으로 변환하는 로직 필요.
- 회귀/분류 자동 감지가 잘못될 경우 `column_detection.detect_task()`의 `num_unique_threshold` 조정 필요.
- `task`와 metric set이 섞이면 `KeyError`가 발생하므로, 프론트와 백엔드 모두 같은 task 기준으로 처리해야 함.

---

## 7. 요약
- 2026-05-08 가이드 이후 추가된 핵심은 **회귀 cell과 이미지 cell 통합**입니다.
- webplatform에는 `task`/`data_type` 확장, 회귀/이미지 metric set 지원만 추가 반영하면 됩니다.
- 기존 v3.2 결과는 legacy 모드로 유지하고, 새 진단은 v5 엔진으로 수행하세요.
- [ ] 회귀 CSV 업로드 시 `task=='regression'` 확인.
- [ ] 회귀 결과에 `target_distribution_quality`, `target_smoothness`, `feature_informativeness_reg`가 존재.
- [ ] 결과 JSON에 `task`/`data_type` 키가 포함됨.
- [ ] legacy v3.2 기록은 `legacy_v32`로 표시됨.
- [ ] 2026-05-08 이후 추가 검증 체크리스트
- [ ] 회귀 CSV 업로드 시 `task=='regression'` 확인.
- [ ] 회귀 결과에 `target_distribution_quality`, `target_smoothness`, `feature_informativeness_reg`가 존재.
- [ ] 이미지 데이터셋 업로드 시 `data_type=='image'`, `task=='classification'` 확인.
- [ ] 이미지 결과에 `completeness_image`, `sample_quality_image` 등 10개 지표 존재.
- [ ] 결과 JSON에 `data_type` 키가 포함됨: data_type까지 분기 추가 — 업데이트.
- webplatform worker result 생성 코드: `data_type` 포함 JSON 전송 — 추가.
- 프론트 metric 렌더링 로직: task별 키셋 분기 (regression/image 추가) — 업데이트

## 7. 요약
- v5 통합의 핵심은 **task-aware 진단**과 **분류/회귀 별 지표 분리**입니다.
- webplatform에는 `dsc_framework` import, `task` 전달, 결과 metadata 저장, task별 UI 분기만 반영하면 됩니다.
- 기존 v3.2 결과는 legacy 모드로 유지하고, 새 진단은 v5 엔진으로 수행하세요.
