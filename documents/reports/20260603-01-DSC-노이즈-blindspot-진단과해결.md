# DSC 노이즈 blind spot — 진단과 해결 (signal_integrity)

> 이미지 성능측정을 frozen-feature probe로 재설계(ADR-019)하는 과정에서 발견한
> DSC 프레임워크의 구조적 한계와 그 해결. 발표용 핵심 스토리.

---

## 1. 발견 — DSC가 노이즈를 "품질 양호"로 오판

이미지 회귀 cell을 probe로 검증하니, **5개 오염 중 4개는 DSC↔성능 상관이 강했는데(r=+0.65~0.94) `noise_injection` 하나가 전체를 깨뜨렸다**:

| | 전체 5 polluter | noise_injection 제외(4개) |
|---|---|---|
| UTKFace | r=−0.16 (FAIL) | **r=+0.94** |
| SCUT | r=+0.16 (FAIL) | **r=+0.65** |

`noise_injection`에서 **DSC는 ~85~87로 평평한데 probe 성능(R²)은 0.43→0.00으로 붕괴**. DSC가 "품질 멀쩡"이라는데 모델은 학습 불가 — 정반대.

## 2. 근본 원인 — DSC는 "차감형 열화"만 잡게 설계됨

DSC 10개 메트릭의 탐지 논리가 전부 **신호를 빼거나·흐리거나·중복**시키는 열화용이다(완전성=누락, blur=선명도↓, uniqueness=중복…). tabular DQ 차원(DQ4AI)을 물려받은 결과.

노이즈는 정반대 — **가짜 고주파를 *더한다*.** 이미지 품질 담당 `sample_quality_image`는 **Laplacian 분산**을 쓰는데, 노이즈가 이걸 폭증시켜 "더 선명=더 좋음"으로 읽는다.

**실측 (합성 이미지):**

| 노이즈 σ | Laplacian 분산 | sample_quality |
|---|---|---|
| 0 (clean) | 369 | 1.000 |
| 30 | 5,301 | 1.000 |
| 60 | 18,324 | 1.000 |

→ 노이즈가 심해질수록 Laplacian은 폭증, sample_quality는 최대치(1.0)에 박혀 노이즈를 "완벽한 고품질"로 오판. 게다가 `feature_informativeness_reg`는 1.0에 clip돼 포화 → 떨어질 여력 없음. **노이즈가 모든 메트릭의 사각지대로 빠진다.**

## 3. 왜 여태 안 들켰나 — finetune이 가렸다

기존 평가(full finetune)는 노이즈에 **적응**해서 R²가 ~0.7~0.8 유지 → DSC도 평평, 성능도 평평 → **둘 다 안 움직여 우연히 일치**, gap이 숨었다. probe(frozen feature)는 적응 못 하니 노이즈에 붕괴 → **DSC가 틀렸다는 걸 노출**. 재설계가 한계를 드러낸 것.

## 4. 메타 통찰 (발표 포인트)

> DSC는 **차감형(subtractive) 열화엔 강하고, 추가형(additive/corruptive) 열화엔 약하다.** 이는 단일 버그가 아니라 메트릭 셋의 **커버리지 편향**이며, probe 재설계가 이를 자가진단했다.

## 5. 해결 — signal_integrity 메트릭 신설

**Immerkær(1996) 빠른 노이즈 σ 추정**을 보수로 변환. 3×3 마스크가 구조(엣지)를 상쇄해 노이즈만 추정 → blur/디테일엔 오발 없이 노이즈 강도만 잡는다.

`signal_integrity = mean( 1 − clip(σ̂ / noise_norm, 0, 1) )`, noise_norm=25 사전등록.

**검증 (합성, 노이즈 추정량 복원):**

| 조건 | Immerkær σ̂ | signal_integrity |
|---|---|---|
| clean | 0.25 | 0.99 |
| blur s2 | 0.00 | 0.99 (오발 없음) |
| noise s10 | 10.2 | 0.76 |
| noise s30 | 30.3 | 0.31 |
| noise s60 | 53.7 | 0.00 |

→ clean·blur는 안 떨어지고, 노이즈는 실제 σ를 거의 정확히 복원하며 강도따라 하락. (비교: MAD-Laplacian은 blur에 오발, 디노이징 잔차는 clean에 오발 → Immerkær 채택.)

`image_cell` + `image_cell_regression` 둘 다 추가. 가중치: `sample_quality_image` 0.15→0.10 + `signal_integrity` 0.05 (합 1.00 유지).

## 6. 효과 검증 (Colab 재실행 전, 로컬 시뮬)

실제 probe 결과 + 추정 signal_integrity를 §2 가중치 최적화에 주입:

| | held-out r (전체 5 polluter) |
|---|---|
| signal_integrity 없음 | +0.42 (턱걸이) |
| **signal_integrity 추가** | **+0.94** |

→ 새 메트릭을 §2 최적화가 활용해 **noise 포함 전체 상관이 0.42→0.94로 회복**. 단 signal_integrity 값은 노이즈 레벨로 *추정*한 것 → **실값은 02 재실행에서 확정**(시뮬은 0.94 예측).

## 7. 후속

- **이미지 회귀**: 02 재실행(DSC에 signal_integrity 포함) → 04 §2 최적화로 실 r 확인.
- **이미지 분류**: 같은 메트릭 추가됨 → 재실행 시 `04_scoreboard_image`의 hardcoded METRICS 목록에 signal_integrity 추가 필요(현재 미반영, follow-up).
- **텍스트**: char-noise에 동일 blind spot 가능성 → 텍스트 cell 재설계 시 점검.

**관련**: ADR-019(진단), `image_cell.py`(`calc_signal_integrity`), progress 20260602-01.
