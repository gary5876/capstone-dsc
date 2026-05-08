# ADR-014: 이미지 cell 사전등록 (v5 framework 확장)

- **상태**: Proposed (사전등록 시작, 구현 대기)
- **결정일**: 2026-05-08
- **선행 결정**: ADR-011 (강한 버전 framework), ADR-013 (dsc_framework 분리)
- **결정 근거**: 2026-05-07 팀 회의 — "다양한 데이터, 최소 이미지는 해야 함"
- **마스터플랜**: `documents/plans/20260508-01-이미지-cell-마스터플랜.md`

---

## 1. 컨텍스트

v5 마스터플랜(2026-04-27)에서 멀티모달(text/image/audio)은 캡스톤 한계 안에서 어렵다고 판단해 Limitations에 후속 연구로 명시했다. 그러나 2026-05-07 팀 회의에서 다음이 결정됨:

1. **최소 이미지 cell은 캡스톤에 포함**
2. 멀티모달까지는 stretch goal
3. GPU는 Colab Pay-as-you-go (학교 카드 결제) 또는 Colab Pro+로 확보

framework 주장 강화 측면에서도 (tabular×classification, tabular×regression, **image×classification**) 3개 cell이면 학술 instance가 더 견고해진다.

## 2. 결정

이미지 cell을 v5 framework의 3번째 instance로 정식 사전등록한다. 구현은 회귀 cell Phase 2 검증 통과 후 시작.

## 3. 사전등록 항목 (구현 전 freeze)

### 3-1. 데이터셋 후보 (3개)

| 데이터셋 | 출처 | 클래스 | 행수 (train+test) | 이미지 크기 | 라이선스 |
|---|---|---:|---:|---|---|
| **CIFAR-10** | torchvision (Krizhevsky 2009) | 10 | 50K + 10K | 32×32 | MIT (PyTorch) / academic |
| **Fashion-MNIST** | torchvision (Xiao 2017) | 10 | 60K + 10K | 28×28 grayscale | MIT |
| **Flowers102** | Oxford Visual Geometry Group | 102 | ~2K + 6K | 가변, resize 224 | Non-commercial academic |

**선정 근거**:
- CIFAR-10: image classification 표준 벤치마크, GPU/CPU 모두 부담 적음
- Fashion-MNIST: 그레이스케일이라 polluter 일부(예: color shift)가 의미 없음. 대비군으로 활용
- Flowers102: 클래스 102개로 clase_balance 변동을 폭넓게 측정 가능. 이미지 크기 큼 → GPU 권장

**대체안**: ImageNet 부분 (subset 50 classes), Caltech-101. 첫 라운드는 위 3개로 진행, 시간 여유 시 ImageNet subset 추가.

### 3-2. 모델 후보 (5개)

| 모델 | torchvision | 파라미터 | GPU 권장 |
|---|---|---:|:---:|
| **ResNet-18** | `torchvision.models.resnet18` | 11.7M | T4 |
| **EfficientNet-B0** | `torchvision.models.efficientnet_b0` | 5.3M | T4 |
| **MobileNetV3-small** | `torchvision.models.mobilenet_v3_small` | 2.5M | T4 |
| **ViT-Tiny** (timm) | `timm.create_model('vit_tiny_patch16_224')` | 5.7M | L4 권장 |
| **CNN-Simple** (베이스라인) | 3 × Conv + 2 × FC, 직접 정의 | <1M | CPU도 가능 |

**선정 근거**:
- 분류 cell·회귀 cell의 5개 모델과 짝수 맞춤 (framework 일관성)
- ResNet/EfficientNet/MobileNet은 standard architecture
- ViT-Tiny는 attention 기반 — 다양성 확보
- CNN-Simple은 sanity check (DSC가 매우 단순한 모델에서도 신호를 주는지)

**하이퍼파라미터 사전등록**:
- pretrained=False (DSC 점수의 영향 분리 위해 random init)
- optimizer: Adam (lr=1e-3)
- batch_size: 128
- epochs: 10 (sanity), 30 (정식)
- random_state: 42

### 3-3. Polluter 라인업 (5개)

| Polluter | 정의 | level 의미 |
|---|---|---|
| **completeness_image** | 픽셀 일부를 배경색(black)으로 마스킹 | level = 마스킹 비율 (0.1~0.95) |
| **noise_injection** | Gaussian noise 추가 | level = noise std / image_std |
| **blur** | Gaussian blur 적용 | level = sigma 정규화 |
| **class_balance** | 클래스별 샘플 수 불균형화 | tabular cell과 동일 |
| **label_swap** | label 비율 무작위 swap | level = 변경 비율 |

**구현**: `polluters/image/`에 별도 모듈로 작성. tabular polluter와 인터페이스 통일 (`pollute(df_or_dataset)` 또는 PyTorch Dataset wrapper).

### 3-4. 메트릭 정의식 (10개, 사전등록)

이미지 cell은 9 + 1 = 10개 메트릭. 9개는 cell 패턴 유지, 1개 신설:

| 메트릭 | 정의식 | 가중치 (사전등록) |
|---|---|---:|
| `completeness_image` | 1 - (마스킹 픽셀 비율 평균) | 0.15 |
| `uniqueness` | 1 - (perceptual hash 중복 비율) | 0.10 |
| `validity` | 손상되지 않은 이미지 비율 (load 성공) | 0.05 |
| `consistency` | 색공간/크기 일관성 (mode + size 분포 entropy) | 0.05 |
| `outlier_ratio` | mean intensity의 IQR-based outlier 비율 보수 | 0.05 |
| `class_balance` | tabular cell과 동일 | 0.10 |
| `feature_correlation` | feature embedding(ResNet18 pretrained) 간 cosine 상관 | 0.05 |
| `label_consistency` | k-NN feature embedding 라벨 일관성 (chance 보정) | 0.20 |
| `feature_informativeness` | embedding → label MI / H(Y) | 0.10 |
| **`sample_quality_image`** (NEW) | blur (Laplacian variance) + contrast (RMS) 결합 점수 | 0.15 |

**가중치 합 = 1.00**.

`sample_quality_image`는 이미지 cell만의 신설 지표 — tabular cell의 어떤 지표와도 1:1 매칭이 안 되는 image-intrinsic 품질 측정.

**ADR-011 강한 버전 원칙 준수**: 차원 이름이 같아도 정의식이 cell마다 다름. 예를 들어 `feature_correlation`은 tabular에서는 컬럼 간 Pearson, image에서는 ResNet embedding 간 cosine.

### 3-5. 평가 메트릭

- 분류 cell과 동일: accuracy, F1(macro), top-1
- 다중 클래스 표준: 학습 후 test set 평가
- 평가 기준: r(DSC, accuracy) ≥ 0.4

## 4. 구현 범위 (캡스톤 한계)

**최소 범위 (반드시)**:
- 3 dataset × 5 model × 5 polluter × 6 level = 450 학습 + 18 baseline = 468건
- T4 GPU 가정 시 1 model fit ≈ 5~15분 → 총 수십 시간. Colab Pro 권장.

**Stretch goal (시간 여유 시)**:
- ImageNet subset 추가
- Vision Transformer 모델 추가
- 멀티모달 cell (image + text)

## 5. dsc_framework 통합

```
dsc_framework/
├── shared_metrics.py            # tabular 6개 (변경 없음)
├── classification_cell.py        # tabular × classification
├── regression_cell.py            # tabular × regression
├── image_cell.py                 # image × classification (NEW)
├── column_detection.py           # tabular용 (변경 없음)
├── data_type_detection.py        # NEW — DataFrame vs ImageDataset 감지
└── router.py                     # data_type까지 분기 추가
```

`router.py`의 `select_profile`은 캡스톤에선 `(data_type, task) → profile`로 확장:
- `('tabular', 'classification')` → classification_cell
- `('tabular', 'regression')` → regression_cell
- `('image', 'classification')` → image_cell

## 6. 검증 통과 기준 (Phase 2)

회귀 cell과 동일한 메타 검증:
- Pearson r(DSC, accuracy) ≥ 0.4
- Spearman ρ ≥ 0.4
- Polluter hold-out 4/5 PASS
- 모델 5/5 양의 r

## 7. 후속 작업 (마스터플랜 참조)

`documents/plans/20260508-01-이미지-cell-마스터플랜.md`:
- Phase 1: 데이터·polluter·메트릭 인프라 (≈7일)
- Phase 2: 학습·검증 (≈4일, GPU 시간이 변수)
- Phase 3: framework 통합 (≈2일)

## 8. 사전등록 freeze 항목

본 ADR로 다음 항목이 freeze됨 (결과 확인 후 변경 금지 — F1 순환 논증 회피):
- 데이터셋 3개
- 모델 5개 (하이퍼파라미터 포함)
- Polluter 5개 (정의 + level 의미)
- 메트릭 10개 (정의식 + 가중치)
- 평가 메트릭 (accuracy)

추가 가중치 조정·정의식 수정 시 ADR-014의 보충 ADR(예: ADR-014a)로 명시적 기록.
