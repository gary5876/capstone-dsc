# ADR-018: 이미지 cell 회귀 트랙 사전등록 (v5 framework 확장 — 6번째 instance)

- **상태**: Proposed (사전등록). 셀 코드 구현 완료(`dsc_framework/image_cell_regression.py`, 2026-05-30), target 계열 polluter 2종 + 실험 미수행.
- **결정일**: 2026-05-30
- **선행 결정**:
  - ADR-011 (강한 버전 framework — cell마다 정의식 재정의)
  - ADR-012 (Degradation Index 보조 지표, 회귀 cell에 그대로 승계)
  - ADR-014 (이미지 cell 분류 트랙 사전등록 — 본 ADR과 짝)
  - ADR-017 (텍스트 cell 회귀 트랙 — 동일 미러 패턴의 직전 사례)
- **결정 근거**: (tabular, image, text) × (classification, regression) 매트릭스에서 image 행만 분류 1칸으로 비어 있어 framework 일반성 주장이 불완전. image × regression을 추가하면 6칸 매트릭스가 **5/6 instance**(image×reg 포함, 원래 제외했던 칸)로 채워져 "(data_type, task) → metric_set" 매핑이 두 task 축 모두에서 세 데이터유형 전부 검증된다.
- **선행 cell 패턴**: text × regression(ADR-017)이 text × classification(ADR-016)을 미러링한 것과 **동일 구조**로, image × classification(ADR-014)을 회귀 트랙으로 미러링. `class_balance → target_distribution_quality`, `label_consistency → target_smoothness`, `feature_informativeness → feature_informativeness_reg(mutual_info_regression)`.

---

## 1. 컨텍스트

ADR-014에서 image × classification은 사전등록·Phase 1 통과(default r=0.45, tuned held-out r=0.57). 본 ADR은 같은 데이터 유형(image)의 회귀 트랙을 추가한다. 회귀 트랙은 분류 트랙과 다음 두 점에서 다르다.

1. **타겟이 연속형/ordinal** — class_balance / label_consistency / feature_informativeness 정의식 적용 불가
2. **평가 메트릭이 R²** — accuracy 대신 R²(음수 clip to 0)로 합격선 측정

tabular·text regression cell이 이미 메트릭 재정의 패턴을 검증했으므로 동일 패턴을 ResNet18 임베딩 공간에서 그대로 재사용한다.

## 2. 결정

이미지 cell의 회귀 트랙(image × regression)을 v5 framework instance로 정식 사전등록한다. 본 ADR에서 freeze한 항목은 결과 보고 후 변경 금지(가중치 제외).

## 3. 사전등록 항목 (구현 전 freeze)

### 3-1. 데이터셋 (튜닝 + held-out)

| 데이터셋 | HuggingFace ID | 타겟(회귀) | 규모 | 역할 | 검증 |
|---|---|---|---:|---|---|
| **UTKFace** | `Subh775/UTKFace_demographics_V1` | age 0–116 (int→float) | 10,135 · 200×200 | **튜닝** | ✅ 2026-05-30 load_dataset 스트리밍 실제 성공, 컬럼 `image,age,gender,race` 확인 |
| **SCUT-FBP5500** | `MnLgt/scut-fbp5500` | beauty_score 1–5 (연속) | 5,500 | **held-out** | ✅ 2026-05-30 load_dataset 스트리밍 실제 성공, 컬럼 `image,beauty_score,...` 확인 |

**선정 근거**:
- image × classification(ADR-014)이 CIFAR-10(튜닝) / FashionMNIST(held-out) 2개로 Phase 1을 통과한 선례를 따라, image × regression도 **튜닝 1 + held-out 1** 구조로 시작. DSC 셀은 target 의미와 무관하게 동일 지표를 계산하므로, age(UTKFace)에서 가중치를 정하고 beauty(SCUT)에서 held-out 평가하는 cross-dataset·cross-target 일반화는 분류 트랙의 CIFAR→FMNIST(객체→의류)와 동일한 강도의 검증이다.
- **UTKFace 튜닝 근거**: age는 회귀의 사실상 표준 타겟(0–116 광범위), 분포가 청년 과다로 편향돼 있어 `target_distribution_quality`가 신호를 낼 여지가 큼. 10K 규모로 충분.
- **SCUT held-out 근거**: 다른 타깃(미모 점수 1–5, 평균 ~3 정규분포 근사) + 다른 도메인 → domain·target transfer 신호 측정.

**target ordinal/continuous 처리**: age는 정수 별로, beauty는 연속 평균점수를 그대로 float 캐스팅 후 regression 학습. R² 계산 시 rescale 없음.

**held-out 형식화**: 합격선·held-out 분리 plan 문서로 분기 (작성 예정 — `documents/plans/20260530-02-이미지-회귀-합격선-heldout-사전등록.md`).

**3번째 데이터셋(stretch)**: LODO 통계력 강화를 위해 검증된 3번째(추가 age 또는 aesthetic score) 1개 확보 시 튜닝셋 보강. 단 깔끔히 parquet 로드되는 후보가 제한적이라 2개로 시작.

### 3-2. 모델 (5개)

image × classification(ADR-014)의 5개 backbone에 regression head(linear, no softmax) + MSELoss 적용. 미러.

| 모델 | backbone | head | GPU |
|---|---|---|:---:|
| **ResNet18-Reg** | ResNet18 (ImageNet pretrained) | GAP → linear(1) | T4 |
| **CNNSimple-Reg** | 직접 정의 소형 CNN (from scratch) | flatten → linear(1) | CPU 가능 |
| **EfficientNet-B0-Reg** | EfficientNet-B0 pretrained | linear(1) | T4 |
| **MobileNetV3-small-Reg** | MobileNetV3-small pretrained | linear(1) | T4 |
| **ViT-Tiny-Reg** | ViT-Tiny pretrained | [CLS] → linear(1) | T4 |

**선정 근거**: image 분류 cell의 ResNet18 / CNNSimple / EfficientNetB0 / MobileNetV3small / ViTTiny 5종(`results/model_performance_image.csv`에 등장)에 1:1 대응. 아키텍처 다양성(전통 CNN / 경량 / Transformer) 유지.

**하이퍼파라미터 사전등록**:

| 항목 | CNNSimple-Reg | pretrained 4종 |
|---|---|---|
| image_size | native (32–200) | 224 resize |
| batch_size | 64 | 64 |
| epochs | 10 | 10 |
| optimizer | Adam | AdamW |
| lr | 1e-3 | 2e-5 (head 1e-3) |
| loss | MSE | MSE |
| pretrained | False | True |
| random_state | 1 | 1 |

(image cell이 `random_state=1` 관례 — text cell의 42와 다름. 일관성 위해 1 유지.)

### 3-3. Polluter 라인업 (5개)

분류 트랙(ADR-014)의 5개 중 `class_balance_image`·`label_swap` 2개를 회귀용으로 재정의. 픽셀 변형 3종은 동일 공유.

| Polluter | 정의 | level 의미 | 구현 출처 |
|---|---|---|---|
| **completeness_image** | 이미지 일부를 배경색 마스킹 | level = 마스킹 비율 | 기존 `image_polluters/completeness_image.py` (공유) |
| **noise_injection** | 픽셀 Gaussian noise | level = noise σ / image_std | 기존 `image_polluters/noise_injection.py` (공유) |
| **blur** | Gaussian blur | level = sigma 강도 | 기존 `image_polluters/blur.py` (공유) |
| **target_distribution_skew** | target 분포 편향 (Q3 이상 제거) | level = 제거 비율 | ✅ `image_polluters/target_distribution_skew.py` — `dq4ai.TargetDistributionSkewPolluter` wrapper |
| **target_noise** | target에 Gaussian noise | level = σ / target_std | ✅ `image_polluters/target_noise.py` — `dq4ai.TargetAccuracyPolluter` 회귀 분기 wrapper |

**구현 방침**: target 계열 2종은 text cell의 `text_polluters/target_distribution_skew.py`·`target_noise.py`와 동일 패턴으로 `dsc_framework/image_polluters/target_distribution_skew.py`·`target_noise.py`에 신규 작성. `(images, targets) → DataFrame(target) → dq4ai polluter → (images_out, targets_out)`. 이미지 본문은 불변, target만 조작.

### 3-4. 메트릭 정의식 (10개, 사전등록 — 가중치는 fallback)

분류 트랙(ADR-014) 10개 중 3개를 회귀용으로 재정의. 7개는 정의식 동일. **셀 코드 구현 완료**(`dsc_framework/image_cell_regression.py`).

| 메트릭 | 정의식 | 가중치(fallback) | vs 분류 트랙 |
|---|---|---:|---|
| `completeness_image` | 동일 (마스킹 픽셀 비율 보수) | 0.15 | 동일 |
| `uniqueness` | 동일 (perceptual hash 중복) | 0.10 | 동일 |
| `validity` | 동일 (load 성공 비율) | 0.05 | 동일 |
| `consistency` | 동일 (mode·size entropy 보수) | 0.05 | 동일 |
| `outlier_ratio` | 동일 (mean intensity IQR) | 0.05 | 동일 |
| **`target_distribution_quality`** (재정의) | target 10-bin equal-width → normalized Shannon entropy | 0.10 | `class_balance` 대응 |
| `feature_correlation` | 동일 (ResNet18 임베딩 cosine 상관 보수) | 0.05 | 동일 |
| **`target_smoothness`** (재정의) | k-NN(k=5, ResNet18 임베딩) 이웃 target std / target_std 보수 | 0.20 | `label_consistency` 대응 |
| **`feature_informativeness_reg`** (재정의) | `mutual_info_regression(embedding, target)` 합 / log(10) clip[0,1] | 0.10 | `feature_informativeness` 대응 |
| `sample_quality_image` | 동일 (blur Laplacian + contrast 결합) | 0.15 | 동일 |

**가중치 합 = 1.00**. 정의식만 freeze. **운영 가중치는 fallback이 아니라 셀별 튜닝/held-out 분리 선정으로 대체**(개선계획 `documents/plans/20260530-01-파라미터-가중치-개선계획.md` §2 절차 — 약신호 지표 상한·핵심 지표 하한 제약 하 최적화).

**ADR-011 강한 버전 원칙 (회귀 트랙)**:
- `target_smoothness`: tabular = 수치형 컬럼 k-NN, text = DistilBERT 임베딩 k-NN, **image = ResNet18 임베딩 k-NN**
- `feature_informativeness_reg`: **image = ResNet18 embedding MI(연속 target)**

### 3-5. 평가 메트릭

- R² (음수 clip to 0)
- **합격선**: r(DSC, R²) ≥ 0.4 (Pearson + Spearman 모두)
- **Polluter hold-out**: 5개 중 4개 PASS
- **모델별 r**: 5/5 양의 r
- **보조 지표**: ADR-012 Degradation Index — `m_deg = max(0, 1 − R²_polluted / R²_clean)`. SCUT(beauty 1–5, 좁은 범위)·UTKFace 약신호 구간 floor effect 회피.

## 4. 구현 범위 (캡스톤 한계)

**최소 범위**:
- 2 dataset × 5 model × 5 polluter × 5 level + baseline. image cell(ADR-014) level grid {0.1,0.3,0.5,0.7,0.9} 승계.
- = 2 × 5 × (1 + 5×5) = 260건 학습.
- T4 GPU 기준 pretrained finetune 위주, 총 GPU 시간 분류 트랙과 동급(수십 시간).
- sample_cap: DSC·train 5,000 / test 전체 (이미지 분류 트랙과 동일), split seed 1, test 0.2 (freeze).
- Phase 1 축소(이미지 분류 선례): 모델 2종(ResNet18 pretrained + CNNSimple), EPOCHS 10, level 5단계 {0.1,0.3,0.5,0.7,0.9}. FULL setting은 노트북에 *_FULL로 보존.

**Stretch**: 3번째 데이터셋, DeBERTa류 대형 backbone.

**범위 외**: bounding-box 회귀, dense 회귀(depth map), 시퀀스/비디오.

## 5. dsc_framework 통합 (현재 상태)

```
dsc_framework/
├── image_cell.py                     # image × classification (ADR-014)
├── image_cell_regression.py          # image × regression (NEW, 본 ADR) ✅ 구현·검증 완료
├── image_polluters/
│   ├── completeness_image.py         # 공유 (ADR-014)
│   ├── noise_injection.py            # 공유 (ADR-014)
│   ├── blur.py                       # 공유 (ADR-014)
│   ├── class_balance_image.py        # 분류 전용 (ADR-014)
│   ├── label_swap.py                 # 분류 전용 (ADR-014)
│   ├── target_distribution_skew.py   # 회귀 전용 ✅ 구현·검증 완료
│   └── target_noise.py               # 회귀 전용 ✅ 구현·검증 완료
└── router.py                         # ('image','regression') 분기 추가 ✅ 완료
```

`router.py`: `('image','classification') → image_cell`, `('image','regression') → image_cell_regression`. ✅ 등록 완료. `compute_dsc(images=..., targets=..., data_type='image', task='regression')` 동작 확인(2026-05-30, 비임베딩 경로). 임베딩 3지표는 torchvision(ResNet18) 필요 → Colab에서 검증.

## 6. 검증 통과 기준 (Phase 2)

- Pearson r(DSC, R²) ≥ 0.4 (튜닝셋 UTKFace)
- Spearman ρ ≥ 0.4
- Polluter hold-out 4/5 PASS
- 모델 5/5 양의 r
- held-out(SCUT) r 보고 (default vs 셀별 선정 가중치 비교)
- **Degradation 지표 보고**: ADR-012에 따라 absolute r + preservation r 두 값.

## 7. 외부 의존성 사전 검증 (CLAUDE.md 준수)

본 ADR 작성 시점(2026-05-30) **실제 load_dataset 스트리밍 실행으로** 확인:

| 자원 | 상태 | 비고 |
|---|---|---|
| `Subh775/UTKFace_demographics_V1` | ✅ 실측 로드 성공 | parquet, viewer ON, remote code 불필요. age 0–116, 10,135행 |
| `MnLgt/scut-fbp5500` | ✅ 실측 로드 성공 | beauty_score 1–5, 5,500행. 라이선스 연구용(other) |
| ResNet18/EfficientNet-B0/MobileNetV3/ViT-Tiny pretrained | torchvision 내장 | image 분류 cell(ADR-014)과 공유. Colab에서 동작 확인됨 |
| `torchvision` | Colab 필요 | 본 로컬 환경 미설치 → 임베딩 경로는 Colab 실행 |

**Python 패키지**: torch/torchvision/datasets는 ADR-014와 공유. 추가 없음.

## 8. 사전등록 freeze 항목

- 데이터셋: UTKFace(튜닝) + SCUT-FBP5500(held-out) + 샘플링 cap(8K/2K)
- 모델 5개 (하이퍼파라미터 포함)
- Polluter 5개 (정의 + level grid {0.1,0.3,0.5,0.7,0.9})
- 메트릭 10개 (정의식만 freeze, 가중치는 §2 데이터 기반 선정)
- 평가 메트릭 (R², 음수 clip to 0), 합격선 r ≥ 0.4 (Pearson + Spearman)
- Polluter hold-out 4/5 PASS, 모델별 양의 r 5/5
- ADR-012 Degradation Index 보조 보고 의무

추가 변경 시 ADR-018a.

## 9. 후속 작업

1. ~~**target polluter 2종 구현**~~ ✅ 완료 (2026-05-30, `image_polluters/target_distribution_skew.py`·`target_noise.py`, 합성 데이터 검증)
2. **held-out 사전등록 plan**: `documents/plans/20260530-02-이미지-회귀-합격선-heldout-사전등록.md`
3. ~~**Phase 2 노트북 4종**~~ ✅ 완료 (2026-05-30): `notebooks/01~04_*_image_regression.ipynb` 생성. 이미지 분류 노트북 미러 + 회귀 적응(HF UTKFace/SCUT 로더, target polluter, MSE/R², 회귀 헤드). polluter 5종 (images,연속 targets) 인터페이스 로컬 검증 완료. **실행은 Colab(GPU)**.
4. **Colab(GPU) 실행**: 01→04 순서로 Colab에서 실행 → ResNet18 임베딩 DSC + 모델 5종 학습 + DSC↔R² 상관 + §2 가중치 선정(`tuned_weights_image_regression.json`)

---

**관련 문서**:
- `documents/decisions/ADR-014-이미지-cell-사전등록.md` (분류 트랙, 본 ADR의 짝)
- `documents/decisions/ADR-017-텍스트-cell-회귀트랙-사전등록.md` (동일 미러 패턴 직전 사례)
- `documents/decisions/ADR-011-...` (강한 버전), `ADR-012-...` (회귀 보조 지표)
- `documents/plans/20260530-01-파라미터-가중치-개선계획.md` (가중치 선정 §2 절차 — 운영 가중치 출처)
- `dsc_framework/image_cell_regression.py` (구현 — 본 ADR)
