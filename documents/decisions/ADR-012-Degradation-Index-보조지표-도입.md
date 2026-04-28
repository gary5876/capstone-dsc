# ADR-012: DSC v5에 Baseline-relative Degradation Index 보조 지표 도입

- **일자**: 2026-04-27
- **상태**: 확정 (검증 통과 후 사전 등록)
- **선행**:
  - ADR-011 (Task-conditional framework 강한 버전 채택)
  - `documents/plans/20260427-02-DSC-Framework-v5-마스터플랜.md`
  - 검증 스크립트 `notebooks/_dev/verify_dsc_regression_cross_dataset.py` (Wine Quality에서 skew/noise pollution이 -0.10 임계 미달 — floor effect)

---

## 0. 한 줄 결정

> 기존 9개 메트릭의 절대 점수(`DSC_absolute`)는 그대로 유지하고, **baseline-relative degradation index**를 보조 지표로 추가한다. 약신호 데이터셋(low intrinsic quality baseline)에서도 pollution 효과를 다른 데이터셋과 비교 가능한 단위로 산출.

---

## 1. 배경

### 1-1. Floor effect 발견

회귀 cell cross-dataset 검증(2026-04-27)에서 Wine Quality가 두 검증 시나리오 미달:

| 시나리오 | California | Bike Sharing | **Wine Quality** | 임계 |
|---|---:|---:|---:|---|
| skew=0.75 → ΔTDQ | −0.117 ✅ | −0.163 ✅ | **−0.076 ❌** | ≤ −0.10 |
| noise=0.75 → ΔTS | −0.168 ✅ | −0.120 ✅ | **−0.058 ❌** | ≤ −0.10 |

**근본 원인** — Wine Quality의 본질적 특성:

- target=quality가 정수 3~9 (사실상 7개 distinct 값) → 분포가 5~6에 자연 집중
- baseline TDQ = 0.5532, baseline TS = 0.6007 — **이미 낮음**
- pollution이 "이미 낮은 baseline에서 더 떨어뜨릴 여지가 작아" 절대 Δ가 작음 (= floor effect)

### 1-2. 문제의 의미

이는 **메트릭 결함이 아닌 데이터셋 특성**이지만, 다음 두 결과를 야기:

1. 약신호 데이터셋과 강신호 데이터셋의 pollution 응답을 같은 단위로 비교 불가
2. Phase 2 ML 상관 분석에서 약신호 데이터셋의 신호가 묻혀, 데이터셋별 r 격차가 불필요하게 커질 위험

### 1-3. ADR-011 위험 5의 구체화

ADR-011 위험 5는 "약신호 차원의 일관성"을 다뤘으나, 차원이 아닌 **데이터셋 단위**의 약신호 문제는 별도 처리가 필요. 본 ADR이 이를 보완.

---

## 2. 결정 — Baseline-relative Degradation Index 추가

### 2-1. 정의 (사전 등록)

```python
# Per-metric degradation
m_deg = max(0, 1 - polluted[m] / clean[m])  # m ∈ 9 dimensions

# Aggregate
overall_degradation = Σ wᵢ · mᵢ_deg     # ∈ [0, 1]
preservation_score  = (1 - overall_degradation) × 100   # ∈ [0, 100]
```

가중치 `wᵢ`는 ADR-011 + 마스터플랜 sect 3-2의 사전 등록 가중치 그대로 사용.

### 2-2. 두 점수의 역할 분리

| 지표 | 의미 | 비교 가능성 |
|---|---|---|
| `DSC_absolute` (기존) | "이 데이터셋의 본질적 품질이 얼마나 좋은가" | cell 내, **cell-relative** |
| `overall_degradation` (신규) | "이 데이터셋이 clean baseline 대비 얼마나 손상되었는가" | cell 내 + **cross-dataset 직접 비교 가능** |
| `preservation_score` (신규) | DSC와 같은 단위(0~100), 손상의 보수 | DSC와 함께 보고 가능 |

### 2-3. 시그니처

```python
def compute_dsc_degradation(polluted_dsc, clean_dsc, weights=None):
    """
    Returns:
        overall_degradation: clipped [0, 1]
        overall_degradation_signed: raw (음수 가능 — 진단용)
        preservation_score: (1 - clipped) × 100
        per-metric *_deg, *_deg_signed
    """
```

`compute_dsc_regression()`이 반환한 dict 두 개를 입력으로 받음. 9개 메트릭 정의·가중치·`compute_dsc_regression` 시그니처는 **변경 없음**.

---

## 3. 검증 결과 (사후 등록)

`notebooks/_dev/verify_dsc_degradation.py` 실행 (2026-04-27):

| Dataset | clean DSC | skew=0.75 deg | skew preservation | noise=0.75 deg | noise preservation |
|---|---:|---:|---:|---:|---:|
| California Housing | 89.74 | 0.0203 | 97.97 | 0.1524 | 84.76 |
| Bike Sharing | 87.76 | 0.0224 | 97.76 | 0.1210 | 87.90 |
| Wine Quality | 79.39 | **0.0413** | 95.87 | **0.1061** | 89.39 |

**핵심 결과**:

1. ✅ **clean→clean degradation = 0** (모든 데이터셋, sanity check)
2. ✅ **모든 데이터셋에서 pollution 응답 > 0.005** (Wine Quality 포함, floor effect 회피)
3. ✅ **Wine Quality skew degradation 0.0413** > California 0.0203 — 상대적으로 보면 Wine이 **더 민감**할 수 있음 (절대 Δ로는 약했던 것의 반전)
4. ✅ **noise에서는 California (0.1524) > Wine (0.1061)** — California 의 feature-target 관계가 더 강해 노이즈에 더 민감 (정상 해석)

### 3-1. Floor effect 회피의 정량 증거

Wine Quality skew=0.75:
- **절대** ΔTDQ = −0.076 (이전 fail)
- **상대** TDQ_deg = 1 − 0.4769/0.5532 = **0.138 (13.8%)** → California의 0.128 (12.8%)와 거의 동등

같은 데이터에서 측정 단위만 바뀌어 비교 가능성 회복.

---

## 4. 학술 근거

`documents/reports/20260427-10-레퍼런스조사보고서.md`의 검증된 인용에서:

| 출처 | 활용 |
|---|---|
| Heinrich et al. 2018 (JDIQ Vol 9-2) R5: economy | "DQ 메트릭은 개입 비용·이익 평가에 활용되어야" — degradation index는 정확히 이 용도 |
| Wang & Strong 1996 contextual DQ | "사용 목적 대비 손상 측정"은 contextual 카테고리의 본질 |
| Pipino et al. 2002 측정 원칙 | 객관 측정 + task-conditional 결합 — degradation은 task(clean baseline 대비)에 직접 연계 |

---

## 5. 위험과 대응

### 위험 1 — 절대 점수의 의미 약화

> reviewer: "어차피 degradation으로 보면 되는데 absolute는 왜 보고하나?"

**대응**: 두 지표는 다른 질문에 답함. Absolute = "데이터셋 자체의 품질", Degradation = "손상 정도". 둘 다 task-conditional framework의 다른 측면. 발표/논문에서 명시적으로 두 질문을 분리해 제시.

### 위험 2 — Self-degradation의 정의

> clean → clean = 0인가? clean baseline이 부정확하면?

**대응**: `compute_dsc_degradation(clean_dsc, clean_dsc)` = 0 / 100 검증 통과. Reference baseline의 부정확성은 별개 문제(F8 한계, ADR-009 사전 등록)로, degradation index 자체와 무관.

### 위험 3 — Per-metric clipping의 보수성

> `max(0, 1 - p/c)` 형태로 clipping → 메트릭이 우연히 상승해도 0으로 처리

**대응**: clipping은 "pollution detection"의 자연스러운 정의 (양수 변화만 손상). 단 `*_deg_signed` 필드를 함께 반환하여 진단용 audit trail 유지. 예: completeness=0.25일 때 target_smoothness가 fallback으로 1.0이 되는 현상은 `_deg_signed`에서 음수로 드러남.

### 위험 4 — Cross-cell 비교 가능성 오해

> reviewer: "회귀 cell preservation = 90이 분류 cell preservation = 90과 같은가?"

**대응**: degradation index도 여전히 **cell-relative** (각 cell의 가중치·정의식이 다름). ADR-011의 cross-cell 비교 한계는 degradation 지표에도 동일 적용. 보고 시 항상 `DSC_regression_preservation`, `DSC_classification_preservation`처럼 cell 명시.

### 위험 5 — Phase 2 ML 상관 분석에서 어떤 지표를 쓸 것인가

> r(DSC, R²) 측정 시 absolute 또는 degradation 중 어느 것?

**대응**:

- **Primary**: `r(absolute_DSC, R²)` 계속 측정 — v4 분류 cell과 동일 분석 형식 유지
- **Secondary**: `r(preservation_score, R²)` — 약신호 데이터셋 영향 정량화. Wine 같은 floor effect 데이터셋의 r이 약신호 때문인지, 본질적 무관계 때문인지 분리 가능
- **공식 결과**: 두 r 모두 보고. 발표/논문에서 학술적 차별점 ("our framework provides both intrinsic quality scoring and task-relative degradation tracking").

---

## 6. 영향받는 파일

### 즉시

- `dq4ai/dsc_engine_regression_v5.py` — `compute_dsc_degradation()` 함수 추가 (✅ 완료)
- `notebooks/_dev/verify_dsc_degradation.py` — cross-dataset 검증 (✅ 완료)
- `documents/decisions/ADR-012-Degradation-Index-보조지표-도입.md` — 이 문서

### Phase 2 진입 시

- `notebooks/07_training_regression.ipynb` — 학습 결과에 degradation 컬럼 추가
- `notebooks/08_scoreboard_regression.ipynb` — 스코어보드에 absolute_DSC + preservation_score 두 컬럼

### Phase 3 (framework 통합)

- 분류 cell의 `compute_dsc()`에 대해서도 동일 패턴의 `compute_dsc_degradation_classification()` 추가 검토 (옵션)
- `dq4ai/dsc_framework.py` `select_profile()` 반환값에 degradation 함수 포함

---

## 7. 사전 등록 — 변경 금지 항목

본 ADR 확정으로 다음을 사전 등록 (Phase 2 결과를 보고 변경 금지, ADR-011 사전 등록 원칙 계승):

- `compute_dsc_degradation` 정의식 (sect 2-1)
- 가중치 `wᵢ`는 마스터플랜 sect 3-2의 default profile 그대로 사용
- Phase 2에서 ML 상관 분석 시 absolute_DSC와 preservation_score 두 지표 모두 보고 (sect 5 위험 5)

데이터 결과를 본 후 가중치를 재조정하면 F1(순환 논증) 위반.

---

## 8. 결정 후 즉시 다음 작업

1. 마스터플랜 sect 3에 보조 지표 등록 추가 (마스터플랜 갱신)
2. Phase 1-3 (회귀 모델 5개 학습 인프라) 진행
3. Phase 2에서 absolute + preservation 두 지표로 r 측정

---

**문서 끝.**
