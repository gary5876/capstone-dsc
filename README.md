# DSC 검증 실험

> **연구 질문**: 데이터 품질 점수(DSC Score)가 머신러닝 모델 성능을 예측할 수 있는가?

3개의 공개 데이터셋에 5가지 차원의 오염(결측·중복·값 오류·표현 불일치·클래스 불균형)을 4단계 강도로 주입하여, DSC 점수와 5개 모델(LR/RF/XGB/SVC/MLP)의 F1-score 간 상관관계를 실증 분석합니다.

---

## 빠른 시작 — 어디서부터 볼까

**결과만 빠르게 보고 싶다면:**

| 보고 싶은 것 | 파일 |
|---|---|
| 최종 상관관계·통계 결과 | [`results/04_execution_log.md`](results/04_execution_log.md) |
| 차트 5종 | [`results/charts/`](results/charts/) |
| 전체 실험 통합 테이블 | [`results/merged_results.csv`](results/merged_results.csv) |
| 모델별 성능 피벗 | [`results/scoreboard.csv`](results/scoreboard.csv) |

**코드를 따라가며 이해하고 싶다면:** [실험 파이프라인](#실험-파이프라인) 순서로 노트북을 열어보세요.

**프로젝트 배경·설계 의도를 알고 싶다면:** [`documents/`](documents/) — 마스터플랜, 의사결정 기록(ADR), 진행 기록.

---

## 실험 파이프라인

노트북은 순서대로 실행하며, 각각의 산출물이 다음 단계의 입력이 됩니다.

```
01_setup_and_baseline.ipynb
  ↓ data/raw/*.csv, results/dsc_scores.csv (baseline)
02_pollution_and_dsc.ipynb
  ↓ data/polluted/, data/train_polluted/, data/test_clean/, results/dsc_scores.csv
03_training.ipynb
  ↓ results/model_performance.csv
04_scoreboard.ipynb
  ↓ results/merged_results.csv, scoreboard.csv, charts/, 04_execution_log.md
```

| 단계 | 노트북 | 하는 일 |
|---|---|---|
| **01** | `01_setup_and_baseline.ipynb` | 데이터 다운로드, 베이스라인 DSC 점수 & 모델 성능 측정 |
| **02** | `02_pollution_and_dsc.ipynb` | DQ4AI polluter로 데이터 오염 + DSC 점수 재계산. **ML용 train-only 오염 데이터 별도 생성** (leakage 방지) |
| **03** | `03_training.ipynb` | 오염된 train으로 5개 모델 학습, clean test로 평가. **학습 전 leakage 자동 검증** |
| **04** | `04_scoreboard.ipynb` | DSC ↔ F1 상관분석, 차트 5종 생성, 통계 검증(Pearson/Spearman/ANOVA) |

각 노트북 실행 후 `results/0N_execution_log.md`에 결과가 자동 기록됩니다.

---

## 디렉토리 구조

```
dsc/
├── README.md                     ← 이 문서
├── COMMIT_CONVENTION.md          ← 커밋 메시지 규칙
│
├── notebooks/                    ← 실험 파이프라인 (4단계)
│   ├── 01_setup_and_baseline.ipynb
│   ├── 02_pollution_and_dsc.ipynb
│   ├── 03_training.ipynb
│   └── 04_scoreboard.ipynb
│
├── data/
│   ├── raw/                      ← 원본 CSV 3개 (Telco, SouthGermanCredit, letter)
│   ├── polluted/                 ← 전체 데이터 오염본 (DSC 점수 계산용)
│   ├── train_polluted/           ← train에만 오염 적용한 데이터 (ML 학습용)
│   └── test_clean/               ← 오염 없는 clean test 데이터셋
│
├── results/
│   ├── dsc_scores.csv            ← 데이터셋×오염유형×강도별 DSC 점수
│   ├── model_performance.csv     ← 295건 학습 결과 (accuracy/F1/AUC)
│   ├── merged_results.csv        ← DSC + 모델성능 통합 테이블
│   ├── scoreboard.csv            ← 모델별 F1 피벗
│   ├── charts/                   ← 시각화 5종 (산점도/라인/히트맵/박스플롯/레이더)
│   └── 0N_execution_log.md       ← 각 노트북 실행 로그
│
├── documents/                    ← 프로젝트 문서
│   ├── plans/                    ← 마스터플랜, 로드맵
│   ├── decisions/                ← 의사결정 기록 (ADR)
│   └── progress/                 ← 진행 기록
│
└── dq4ai/                        ← DQ4AI polluter (외부 의존성)
```

자세한 디렉토리·용어 설명은 [`documents/README.md`](documents/README.md) 참조.

---

## 실험 설계

### 데이터셋 (3개)

| 이름 | 샘플 수 | 특징 | 타겟 |
|---|---|---|---|
| TelcoCustomerChurn | 7,043 | 혼합형 (수치+범주) | 이진 분류 |
| SouthGermanCredit | 1,000 | 혼합형 | 이진 분류 |
| letter | 20,000 | 수치형만 | 26-클래스 다중분류 |

### 오염 종류 (5개 × 4단계)

| Polluter | 내용 | 강도 (level) |
|---|---|---|
| completeness | 결측치 주입 | 10% / 25% / 50% / 75% |
| uniqueness | 중복 행 추가 | factor 1.5 / 2.0 / 3.0 / 4.0 |
| feature_accuracy | 피처 값 왜곡 | 10% / 25% / 50% / 75% |
| consistent_repr | 범주형 표현 불일치 (범주형 있는 데이터만) | 10% / 25% / 50% / 75% |
| class_balance | 클래스 비율 불균형 | 10% / 25% / 50% / 75% |

총 실험: 3 데이터셋 × (5 polluter × 4 level + 1 baseline) ≈ 59건 → × 5 모델 = **295회 학습**

### 모델 (5개)

| 모델 | 계열 |
|---|---|
| LogisticRegression | 선형 |
| RandomForest (n=100) | 배깅 |
| XGBoost (n=100) | 부스팅 |
| SVC (linear) | 커널 |
| MLP (5층 × 100) | 신경망 |

### 분할·평가

- **Split**: train 80% / test 20%, stratified, `random_state=1`
- **원칙**: test 데이터는 **절대 오염되지 않음** (구조적 leakage 차단)
- **평가 지표**: Accuracy, F1 (macro), AUC-ROC

---

## 핵심 설계 결정 — Leakage 방지

이 프로젝트는 초기에 **train/test leakage 문제**로 여러 번 실험이 무효화되었습니다. 현재 파이프라인은 다음 원칙으로 leakage를 구조적으로 차단합니다:

1. **Split-first**: clean 데이터를 먼저 train/test로 분할
2. **Train-only 오염**: polluter는 train 파트에만 적용 (test는 깨끗하게 보존)
3. **Clean test 재사용**: 모든 실험에서 동일한 clean test로 평가
4. **자동 검증 안전장치**: 03 학습 루프 직전에 train 데이터가 test와 겹치지 않는지 자동 검증. 문제 발견 시 `RuntimeError`로 학습 즉시 중단

관련 의사결정: [`documents/decisions/ADR-008-DSC엔진-오염감지-개선-필요.md`](documents/decisions/ADR-008-DSC엔진-오염감지-개선-필요.md)

---

## 실행 환경

- **플랫폼**: Google Colab (BASE 경로 `/content/drive/MyDrive/capstone/dsc`)
- **주요 패키지**: pandas, numpy, scikit-learn, xgboost, seaborn, matplotlib, scipy
- **외부 의존성**: [DQ4AI](https://github.com/HPI-Information-Systems/DQ4AI) (polluter)

로컬 실행 시 `BASE` 경로만 환경에 맞게 조정하면 됩니다.

---

## 현재 실험 결과 요약 (2026-04-13 기준)

| 지표 | 값 | 판정 |
|---|---|---|
| Pearson r (DSC ↔ F1) | 0.085 (p=0.147) | 비유의 |
| Spearman ρ | -0.022 (p=0.708) | 비유의 |
| DSC 등급 간 ANOVA | F=0.79, p=0.375 | 비유의 |

**현재 DSC 지표 세트는 ML 성능을 유의하게 예측하지 못합니다.** 원인은 DSC 엔진이 feature_accuracy·class_balance 오염을 충분히 감지하지 못하는 데 있으며, 개선 방향은 [`documents/decisions/ADR-008`](documents/decisions/ADR-008-DSC엔진-오염감지-개선-필요.md)에 정리되어 있습니다.

자세한 결과와 차트는 [`results/04_execution_log.md`](results/04_execution_log.md) 참조.
