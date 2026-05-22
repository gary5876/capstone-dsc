# 이미지 cell 지표 신설 및 테스트 정리

## 이미지 cell 지표 신설

v5 framework의 이미지 cell은 10개 지표로 구성됨. ADR-014에 따라 사전등록된 지표:

| 메트릭 | 정의식 | 가중치 (fallback) |
|---|---|---:|
| `completeness_image` | 픽셀 마스킹 비율 평균 (1 - 마스킹 비율) | 0.15 |
| `uniqueness` | perceptual hash 중복 비율 (1 - 중복 비율) | 0.10 |
| `validity` | load 성공 비율 | 0.05 |
| `consistency` | 색공간/크기 일관성 (mode + size entropy) | 0.05 |
| `outlier_ratio` | mean intensity IQR-based outlier 비율 (보수) | 0.05 |
| `class_balance` | 클래스별 샘플 수 불균형 (tabular과 동일) | 0.10 |
| `feature_correlation` | ResNet18 embedding 간 cosine 상관 | 0.05 |
| `label_consistency` | k-NN embedding 라벨 일관성 (chance 보정) | 0.20 |
| `feature_informativeness` | embedding → label MI / H(Y) | 0.10 |
| `sample_quality_image` | blur (Laplacian variance) + contrast (RMS) 결합 점수 | 0.15 |

**합계 가중치: 1.00**

## 테스트 관련 정리

### 데이터셋
- CIFAR-10: 10 classes, 50K train + 10K test, 32×32
- Fashion-MNIST: 10 classes, 60K train + 10K test, 28×28 grayscale
- Flowers102: 102 classes, ~2K train + 6K test, 가변 size (resize 224)

### 모델
- ResNet-18, EfficientNet-B0, MobileNetV3-small, ViT-Tiny, CNN-Simple

### Polluter
- completeness_image, noise_injection, blur, class_balance, label_swap

### 검증 기준
- Pearson r(DSC, accuracy) ≥ 0.4
- Spearman ρ ≥ 0.4
- Polluter hold-out 4/5 PASS
- 모델 5/5 양의 r

### 노트북
- 01_setup_and_baseline_image.ipynb: baseline 설정
- 02_pollution_and_dsc_image.ipynb: pollution 적용 및 DSC 계산
- 03_training_image.ipynb: 모델 학습
- 04_scoreboard_image.ipynb: 결과 분석

### 통합 테스트
- 이미지 업로드 시 `data_type='image'`, `task='classification'`
- 결과에 10개 지표 포함 확인
- webplatform에서 image cell metric set 지원
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
