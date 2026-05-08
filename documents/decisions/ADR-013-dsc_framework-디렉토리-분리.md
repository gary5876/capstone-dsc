# ADR-013: dsc_framework/ 디렉토리 분리

- **상태**: Accepted
- **결정일**: 2026-05-08
- **선행 결정**: ADR-011 (Task-conditional Framework 강한 버전), ADR-012 (Degradation Index)
- **관련 plan**: 20260427-02 (DSC Framework v5 마스터플랜)

---

## 1. 컨텍스트

v5 framework의 회귀 cell 엔진을 작업하던 중, 회귀 엔진 코드(`dsc_engine_regression_v5.py`)가
외부 HPI 패키지 디렉토리(`dq4ai/`) 안에 위치하고 있었다.

`dq4ai/`는 Budach 2022 논문 코드(Copyright 2021 HPI)로, 외부 라이선스가 부착된 패키지다.
우리 산출물을 그 안에 섞어 두면 다음 문제가 발생한다:

1. **저작권·라이선스 경계 모호** — 어디까지가 HPI 코드이고 어디부터가 우리 산출물인지 불명확
2. **외부 import 경로 부적절** — 통합 측 webplatform 등 외부 사용자가 `from dq4ai.dsc_engine_regression_v5 import ...`로 호출하면 외부 패키지 사용처럼 보임
3. **분류 cell 엔진은 모듈 파일조차 없음** — 노트북 셀(`apply_dsc_engine_v4.py`의 NEW_DSC_ENGINE 문자열) 안에만 존재해서 외부에서 import 불가

## 2. 결정

우리 산출물을 `dsc_framework/` 디렉토리로 분리한다. 외부 패키지 `dq4ai/`는 그대로 유지하되 우리 코드를 더 추가하지 않는다.

```
dsc/
├── dsc_framework/              ← 우리 산출물 (NEW)
│   ├── __init__.py             — public API export
│   ├── shared_metrics.py       — 6개 공통 메트릭 단일 출처
│   ├── classification_cell.py  — 분류 cell (v4 정의식 그대로, r=0.598 보존)
│   ├── regression_cell.py      — 회귀 cell (v5 신설)
│   ├── column_detection.py     — auto_detect_columns + detect_task
│   └── router.py               — compute_dsc(df, task=None) 통합 진입점
│
├── dq4ai/                      — 외부 HPI 패키지 (변경 없음)
│   └── dsc_engine_regression_v5.py — DEPRECATED, verify 스크립트 호환용 사본만 유지
└── ...
```

## 3. 외부 사용 인터페이스

웹 백엔드 측 webplatform 등 외부에서는 다음 한 줄로 충분:

```python
from dsc_framework import compute_dsc, auto_detect_columns

result = compute_dsc(df)  # task 자동 감지
# → {'score': ..., 'grade': ..., 'task': 'classification'|'regression', '...': ...}
```

## 4. 구현 시 보존된 사전등록 결과

분류 cell의 r=0.598 결과(ADR-009 사전등록)를 보존하기 위해, 노트북 01·02 cell 11/8의 NEW_DSC_ENGINE 코드를 비트단위 그대로 `classification_cell.py`로 이동했다.

### 비트단위 동일성 검증 (2026-05-08 실행)

| 비교 | 입력 | 결과 |
|---|---|---|
| 분류: 노트북 NEW_DSC_ENGINE vs `classification_cell.py` | Iris (150행) | **score=96.94 동일, 모든 9개 지표 동일** |
| 회귀: `dq4ai/dsc_engine_regression_v5.py` vs `regression_cell.py` | California Housing (20,640행) | **score=89.74 동일, 모든 9개 지표 동일** |

## 5. 결과 변경 금지 보장

- 분류 cell의 가중치·정의식 freeze (ADR-009, v4 마스터플랜 사전등록)
- 회귀 cell의 가중치·정의식 freeze (ADR-011, v5 마스터플랜 sect 3-2)
- 6개 공통 메트릭은 `shared_metrics.py` 단일 출처에서 두 cell이 import — 일관성 자동 보장

## 6. dq4ai/dsc_engine_regression_v5.py 처리

**유지 + deprecate 주석 추가**. 이유:
- 기존 verify 스크립트 3개(`verify_dsc_engine_regression_v5.py`, `verify_dsc_regression_cross_dataset.py`, `verify_dsc_degradation.py`)가 import 경로를 `dq4ai.dsc_engine_regression_v5`로 사용 중
- verify 스크립트들의 사전등록·재현성 보장을 위해 import 경로 변경 시 새 검증 라운드 필요
- 캡스톤 일정 효율 위해 일단 deprecate 주석만 추가, verify 스크립트 정리는 회귀 cell Phase 2 검증 통과 후

## 7. 외부 진입점·라우팅 구현

`router.py`의 `compute_dsc(df, task=None)`:
1. `target_col`/`numerical_cols`/`categorical_cols`/`task` 중 하나라도 None이면 `auto_detect_columns` 호출
2. `task`에 따라 `compute_dsc_classification` 또는 `compute_dsc_regression` 라우팅
3. 결과 dict에 `task` 키 추가하여 라우팅 결과 명시

`column_detection.detect_task` 휴리스틱:
- dtype object/category → classification
- 정수형 + nunique ≤ 20 → classification
- nunique ≤ 2 → classification
- 그 외 수치형 → regression

캡스톤 단계 한계: ordinal target(예: 1~5 별점)이 정수면 분류로 분류됨. 사용자가 `task='regression'` override 가능.

## 8. 후속 작업

1. 회귀 cell 노트북 4개에서 `from dsc_framework import compute_dsc_regression` 사용 (ADR-014 이후)
2. 분류 cell 노트북 01·02 cell 11/8을 `from dsc_framework import compute_dsc_classification` 호출로 교체 (회귀 cell 검증 통과 후 동시 적용)
3. verify 스크립트 3개 import 경로 정리 (회귀 Phase 2 후)
4. `dq4ai/dsc_engine_regression_v5.py` 삭제 (verify 스크립트 정리 후)
