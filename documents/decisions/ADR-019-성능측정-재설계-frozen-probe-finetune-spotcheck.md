# ADR-019: 성능 측정 재설계 — frozen-feature probe(주) + finetune spot-check(보조)

- **상태**: Accepted (2026-06-02). ADR-014/016/017/018의 "성능 측정 = 모델 full finetune" 부분을 supersede. DSC 지표 계산·polluter·데이터셋·합격선은 그대로 유지.
- **동기**: full finetune이 (polluter × level × model)마다 GPU 학습을 요구 → 이미지 회귀 03이 ~2.5h GPU, Colab 무료 한도 초과로 중단 반복. 6셀 전체로는 비현실적 시간.
- **선행**: 정형 분류 cell(r=0.598)은 **이미 finetune이 아니라 고정 특징 + sklearn 모델**로 성능을 쟀다(DQ4AI Mohammed et al. 동일). 즉 본 재설계는 정형 cell의 검증된 프로토콜을 이미지·텍스트로 확장·통일하는 것.

---

## 1. 결정

성능 측정을 두 트랙으로 재설계하고 6셀에 통일 적용한다.

### (주) Probe — 고정 특징 위 경량 모델 (모든 config)
- **특징(고정)**: 정형 = 전처리된 컬럼 / 이미지 = frozen ResNet18 임베딩 / 텍스트 = frozen DistilBERT 임베딩
- **학습**: 각 (dataset, polluter, level)의 train 특징으로 경량 모델 fit → **clean test 특징**으로 평가
- **모델 셋(task별 고정)**:
  - 회귀: `Ridge`, `RandomForestRegressor`, `MLPRegressor`, `KNeighborsRegressor`
  - 분류: `LogisticRegression`, `RandomForestClassifier`, `MLPClassifier`, `KNeighborsClassifier`
- **지표**: 회귀 R²(음수 clip 0), 분류 F1(macro)
- **비용**: 특징은 1회 추출(이미지/텍스트는 GPU forward-only, DSC 계산과 공유), 모델 fit은 건당 <1초 CPU → 전체 수 분

### (보조) Finetune spot-check — proxy 신뢰성 검증 (소수 config)
- **목적**: probe 성능이 "실제 학습 성능"을 따라가는지 입증 (proxy 타당성 방어)
- **범위**: backbone 1종(이미지 ResNet18 / 텍스트 DistilBERT) finetune을, **DSC 범위를 가르는 축소 subset**에서만:
  - 각 데이터셋 × {baseline + 가장 강신호 polluter(이미지=blur, 텍스트=completeness) × level 5단계} ≈ 6 config/데이터셋 = 12 run/셀
  - epochs 축소(이미지 5, 텍스트 3)
- **비용**: 12 run × ~1~2분 ≈ ~20분/셀 (GPU)
- **검증 산출**: r(probe_perf, finetune_perf) 와 r(DSC, finetune_perf). probe와 finetune이 같은 방향(양의 상관)이면 proxy 정당.
- 정형은 pretrained backbone이 없어 **probe = 실제 모델** → finetune spot-check 불필요(해당 없음).

---

## 2. 셀별 적용

| 셀 | 특징 | probe(주) | finetune spot-check |
|---|---|---|---|
| 정형×분류 | 컬럼 | 기존 sklearn 5종 = 이미 probe (변경 없음, r=0.598 유지) | 해당 없음 |
| 정형×회귀 | 컬럼 | 회귀 4종 (SVR 제거→속도, uniqueness 복제 상한) | 해당 없음 |
| 이미지×분류 | ResNet18 emb | 분류 4종 | ResNet18, blur×5level |
| 이미지×회귀 | ResNet18 emb | 회귀 4종 | ResNet18, blur×5level |
| 텍스트×분류 | DistilBERT emb | 분류 4종 | DistilBERT, completeness×5level |
| 텍스트×회귀 | DistilBERT emb | 회귀 4종 | DistilBERT, completeness×5level |

- 이미지 분류 cell은 통일 위해 probe로 **재측정**(기존 finetune r=0.45는 finetune spot-check 결과로 보존·비교).

---

## 3. 데이터 흐름 (이미지/텍스트)

기존 "02=DSC, 03=finetune"을 **embedding 1회 추출로 통합**:

```
config(dataset,polluter,level)마다:
  pollute(train) → frozen backbone forward → train 임베딩
                 → DSC 계산(임베딩 재사용) + probe 학습(train emb → clean test emb 평가)
clean test 임베딩은 데이터셋당 1회 추출·캐시.
```
- **GPU = forward-only 임베딩 추출**(backprop 없음). 학습 단계의 GPU finetune 제거.
- **raw 이미지 npz 저장 폐지** → Drive 용량 절감, RAM 부담 감소.
- finetune spot-check만 별도로 backbone 학습(소수 config).

---

## 4. 학술적 타당성 + 한계 (발표 명시)

**타당성**:
- Linear probing on frozen features = 표현학습 표준 평가(SimCLR/MoCo/CLIP). "데이터/표현 사용성"의 인정된 척도.
- 정형 cell(r=0.598)·기반 논문(DQ4AI)이 이미 "고정 특징 + 표준 모델" → 본 재설계로 6셀 **단일 프로토콜** 통일(framework 강점).
- 상관 연구엔 finetune의 최적화 노이즈(init/lr/epoch)를 제거해 데이터 품질 효과를 더 깨끗이 분리.
- finetune spot-check로 "probe가 실제 학습 성능을 따라간다"를 정량 입증.

**한계(limitations에 기재)**:
1. probe 성능은 "고정 표현 위 성능 proxy" — full-finetune 성능보다 좁은 개념.
2. 이미지/텍스트 frozen backbone(ImageNet/대규모 코퍼스 사전학습)은 일부 오염에 강건 → 해당 오염의 성능 저하를 과소평가 가능. (정형은 backbone 없어 무관)
→ spot-check가 1·2를 부분 보정/방어.

---

## 5. 합격선 (변경 없음)

- dataset별 Pearson r(DSC, probe성능) ≥ 0.40 (+ Spearman), polluter hold-out, 모델별 양의 r — 기존 plan 20260511-01 그대로.
- **추가 보고**: r(DSC, finetune성능) 및 r(probe성능, finetune성능) (spot-check, proxy 타당성).

## 6. 예상 시간 (재설계 후)

| 셀 | 기존 | 재설계 |
|---|---|---|
| 이미지×회귀 | ~3h GPU | 임베딩+DSC+probe ~30~40분 + spot-check ~20분 ≈ **~1시간** |
| 이미지×분류 | ~3h | 동급 ~1시간 |
| 텍스트 각 | ~수시간 | ~1시간 |
| 정형×회귀 | ~2.5~4.5h(SVR) | SVR 제거 → ~20~40분 |

→ 6셀 전체가 무료 Colab 세션 범위로 들어옴.

## 7. 구현

- 공용 모듈 `dsc_framework/perf_probe.py`: `evaluate_probes(X_tr, y_tr, X_te, y_te, task)` → {model: score}. 모달리티 무관.
- 이미지/텍스트 노트북: embed+DSC+probe 통합 + finetune spot-check 셀.
- 정형 회귀 노트북: 모델셋 probe 4종으로 교체, uniqueness 복제 상한.
- freeze 갱신: 성능 측정 = 본 ADR 프로토콜. polluter/데이터셋/DSC 정의식은 ADR-014/016/017/018 그대로.

## 8. 후속 발견·해결 — DSC additive 노이즈 blind spot (2026-06-03)

probe 재설계로 이미지 회귀 검증 중, **probe가 DSC의 구조적 한계를 노출**: DSC가 노이즈(additive 열화)에 blind(`sample_quality_image`가 Laplacian↑를 "선명"으로 오판) → `noise_injection`에서 DSC 평평·probe 붕괴로 상관 깨짐(held r 0.42).
**해결**: `signal_integrity` 메트릭 신설(Immerkær 노이즈 σ 추정), image cell 분류·회귀 추가. 로컬 시뮬 §2 최적화 held r 0.42→0.94. 상세: `documents/reports/20260603-01-DSC-노이즈-blindspot-진단과해결.md`. (메트릭 추가로 ADR-014/018 metric set 개정 — image 재실행 필요.)

**관련**: ADR-014/016/017/018(성능측정 부분 supersede), plans/20260530-01(가중치 §2), 20260511-01(합격선).
