# 진행 기록 — 2026-05-08 작업 종합

- **작업 범위**: dsc_framework 분리 + 회귀 cell 노트북 4개 + 이미지 cell 사전등록 + webplatform 통합 명세
- **선행 작업**: 20260508-01-dsc_framework-구축및노트북01회귀버전.md (오전)

---

## 1. 산출물 목록

### 1-1. 코드

| 경로 | 내용 |
|---|---|
| `dsc_framework/__init__.py` | public API (compute_dsc, auto_detect_columns 등) |
| `dsc_framework/shared_metrics.py` | 6개 공통 메트릭 단일 출처 |
| `dsc_framework/classification_cell.py` | v5 분류 cell (v4 정의식 그대로) |
| `dsc_framework/regression_cell.py` | v5 회귀 cell + degradation index |
| `dsc_framework/column_detection.py` | auto_detect_columns + detect_task |
| `dsc_framework/router.py` | compute_dsc 통합 진입점 |
| `notebooks/01_setup_and_baseline_regression.ipynb` | 19 셀 |
| `notebooks/02_pollution_and_dsc_regression.ipynb` | 12 셀 |
| `notebooks/03_training_regression.ipynb` | 13 셀 |
| `notebooks/04_scoreboard_regression.ipynb` | 26 셀 |
| `notebooks/_dev/build_nb01_regression.py` | 노트북 01 빌드 스크립트 |
| `notebooks/_dev/build_nb02_regression.py` | 노트북 02 빌드 스크립트 |
| `notebooks/_dev/build_nb03_regression.py` | 노트북 03 빌드 스크립트 |
| `notebooks/_dev/build_nb04_regression.py` | 노트북 04 빌드 스크립트 |
| `dq4ai/dsc_engine_regression_v5.py` | DEPRECATED 헤더 추가 (verify 호환) |

### 1-2. 문서

| 경로 | 종류 | 내용 |
|---|---|---|
| `documents/decisions/ADR-013-dsc_framework-디렉토리-분리.md` | ADR | 우리 산출물을 외부 dq4ai/에서 분리 |
| `documents/decisions/ADR-014-이미지-cell-사전등록.md` | ADR | image × classification cell 사전등록 (데이터·모델·polluter·메트릭) |
| `documents/plans/20260508-01-이미지-cell-마스터플랜.md` | plan | Phase 1~4 단계 + 일정 + 리스크 |
| `documents/reports/20260508-01-aidq-platform-v5-통합-가이드.md` | report | aidq-platform v3.2 → v5 통합 작업 |
| `documents/progress/20260508-01-...` | progress | 오전 작업 (dsc_framework + 노트북 01) |
| `documents/progress/20260508-02-...` | progress | 본 문서 (오후 작업 + 종합) |

---

## 2. 회귀 cell Phase 1 — 노트북 4개 변경점 요약

### 2-1. 노트북 01 (Setup & Baseline)

분류 노트북 01 대비:
- DSC 엔진: NEW_DSC_ENGINE 임베딩 → `from dsc_framework import compute_dsc_regression`
- 데이터셋: SGC/Telco/letter → CalH/Bike/Wine
- 모델: 분류 5개 → 회귀 5개 (LinearReg/RFReg/XGBReg/SVR(rbf)/MLPReg)
- 평가: accuracy/F1/AUC → R² + R²_clipped
- BikeSharing leakage(`casual`+`registered`=`cnt`) + 인덱스 컬럼(`instant`,`dteday`) 명시 제거

**로컬 검증 결과** (실제 데이터):
- DSC baseline: CalH 89.74(B) / Bike 88.05(B) / Wine 79.39(B) — 세 데이터셋 모두 baseline ≥ 0.7 통과
- 모델 baseline (CalH 검증용 4개): LinearReg +0.597 / RFReg +0.807 / XGB +0.833 / SVR +0.733

### 2-2. 노트북 02 (Pollution & DSC)

분류 노트북 02 대비:
- DSC 엔진: NEW_DSC_ENGINE 임베딩 → `from dsc_framework import compute_dsc_regression`
- Polluter 라인업: classbalance 제거, target_distribution_skew 추가
- split: stratify=y 제거 (회귀)
- TRAIN_DIR/TEST_DIR/SPLIT_META_DIR 분리 (분류와 별도 경로)
- 결과: dsc_scores_regression.csv

**로컬 검증** (WineQuality 단일):
- baseline DSC=79.08, level=0.5 폴루션 4종:
  - completeness: -12.23
  - uniqueness: -7.99
  - feature_accuracy: -3.15
  - target_distribution_skew: +0.83 (drop으로 분포 변화 — 단조 감소 미보장, 노트북 04에서 추가 검증)

### 2-3. 노트북 03 (Training)

분류 노트북 03 대비:
- 모델: classifier → regressor
- preprocess: target LabelEncoder 제거
- evaluate_model: accuracy/F1/AUC → R² + R²_clipped
- TRAIN_DIR/TEST_DIR/SPLIT_META_DIR 분리
- 체크포인트 메커니즘 그대로 (model_performance_regression.csv)
- Leakage 검증 (split 인덱스 disjoint + row hash) 그대로

### 2-4. 노트북 04 (Scoreboard)

분류 노트북 04(38셀)을 회귀 cell 핵심으로 간소(26셀):
- 차트 5개 (산점도, 라인, 히트맵, 박스플롯, 레이더)
- Pearson r, Spearman ρ, 비선형 RF 5-fold R²
- 모델별 r
- Polluter hold-out (F1)
- Degradation index (ADR-012) — absolute vs preservation r 비교
- 자동 검증 판정: 5개 가설 (H1~H5)

**검증 기준 (Phase 2 통과 조건)**:
- H1: Pearson r(DSC, R²_clipped) ≥ 0.4
- H2: Spearman ρ ≥ 0.4
- H3: 비선형 RF 5-fold R² > 선형 r²
- H4: 모든 모델에서 양의 r
- H5: Polluter hold-out 4/5 PASS

(분류 cell v4 결과 비교: r=0.598, ρ=0.628, 비선형 R²=0.632)

---

## 3. 이미지 cell 사전등록 (ADR-014)

캡스톤 안에서 framework 3번째 instance 추가 결정 (2026-05-07 팀 회의 → ADR-014).

**Freeze 항목**:
- 데이터셋 3개: CIFAR-10, Fashion-MNIST, Flowers102
- 모델 5개: ResNet-18, EfficientNet-B0, MobileNetV3-small, ViT-Tiny, CNN-Simple
- Polluter 5개: completeness_image, noise_injection, blur, class_balance, label_swap
- 메트릭 10개 (가중치 합 1.00) — 9개는 cell 패턴 유지 + sample_quality_image 신설
- 평가: accuracy + r ≥ 0.4

**일정 추정** (이미지 cell):
- Phase 1 (인프라): 7일
- Phase 2 (검증, GPU 변수): 4일
- Phase 3 (framework 통합): 2일
- 총 13일 (이상적), 실제 GPU 가용에 따라 +α

**리스크 헷지**:
- ViT 어려우면 4 모델로 축소
- Flowers102 어려우면 2 데이터셋
- 캡스톤 안에 못 들어가면 Phase 1만 + Limitations 명시

---

## 4. webplatform 통합 가이드 (aidq-platform v5 통합)

`documents/reports/20260508-01-aidq-platform-v5-통합-가이드.md`.

**핵심 변경 4가지**:
1. 엔진: dsc_framework 패키지 import (git submodule 권장)
2. 인터페이스: `task` 파라미터 추가
3. DB JSON: `task` 필드 추가 (스키마 변경 없음)
4. 프론트엔드: 동적 Slider (분류/회귀 9개씩)

**의존성**: 회귀 cell Phase 2 검증 통과 후 시작 권장.

---

## 5. 다음 단계

### 즉시 실행 가능
- 노트북 02·03·04를 Colab에서 실제 실행 (회귀 cell Phase 1 정식 데이터 생산)
- 결과 csv 4개 생성: dsc_scores_regression.csv, model_performance_regression.csv + 실행 로그 4개

### 회귀 cell Phase 2 검증 결과에 따라
- **PASS** (H1~H5 다 통과): 정식 결과 보고서 작성 → 이미지 cell Phase 1 시작
- **FAIL** (일부 가설 실패):
  - 사전등록 가중치·정의식 변경 금지 (F1 순환 논증 회피)
  - polluter 추가 또는 데이터셋 추가 검토 (사전등록 보완 ADR로)
  - reviewer 방어 위해 정직하게 보고

### 의사결정 미정 항목 (사용자 답변 대기)
1. 학교 카드 결제 — 어느 형태? (메모리: 카드 연동 결제 같음)
2. 이미지 cell이 회의에서 "필수"였는지 vs "이왕이면" — 우선순위 결정
3. dsc_framework sync 방식 (웹 백엔드 측에 옵션 제시: submodule vs 스크립트)
4. 팀원 한 명의 역할 — 미확인 (보고서 수신자 중 1인)

---

## 6. 메모리 업데이트 사항

본 작업으로 다음 메모리 업데이트 필요:
- `project_dsc_v5_framework.md` — 회귀 cell 노트북 4개 작성 완료, ADR-014 추가
- `project_capstone_scope_expansion.md` — 이미지 cell 정식 등록(Limitations에서 제거)
- `project_team_and_webplatform.md` — webplatform 통합 가이드 작성 완료
- 신규 메모리: ADR-013 (dsc_framework 분리) 결정 기록

(메모리 업데이트는 별도 작업으로 진행 예정)
