# ADR-010: TelcoCustomerChurn — DSC 신호 약화의 본질적 데이터셋 한계 인정

- **일자**: 2026-04-27
- **상태**: 확정
- **선행**: F7 진단 (`notebooks/_dev/diagnose_telco.py`)

## 배경

v3.2 정식 결과에서 데이터셋별 r:
- letter: 0.659 (강함)
- SouthGermanCredit: 0.330 (중간)
- **TelcoCustomerChurn: 0.146 (p=0.07, 비유의)**

v4.0 시뮬에서 회복:
- letter: 0.798 (개선)
- SouthGermanCredit: 0.236
- **TelcoCustomerChurn: 0.284 (p=0.0003, 유의로 진입했으나 여전히 약함)**

Telco가 다른 데이터셋 대비 r이 낮은 원인을 진단했다.

## 3가지 가설 진단 결과

### H1. TotalCharges `fillna(0)` 부작용 — 기각

- TotalCharges 결측 = 11건 (0.16%)
- 모든 결측 행: `tenure=0`인 신규 고객, Churn=No
- 데이터셋 의미상 합리적인 결측(아직 청구되지 않은 신규 가입자)
- fillna(0)을 fillna(median)이나 drop으로 바꿔도 r 개선 기대 ≤ 0.01
- **결정**: ADR-007 (`fillna(0)`) 유지. 변경 비용 대비 효과 미미.

### H2. Onehot 차원 폭증 — 채택 (핵심 원인)

| 데이터셋 | rows | num cols | cat cols | **onehot 차원** |
|---|---:|---:|---:|---:|
| TelcoCustomerChurn | 7,043 | 3 | 17 | **13,615** |
| SouthGermanCredit | 1,000 | 8 | 13 | 54 |
| letter | 20,000 | 16 | 0 | 0 |

- Telco는 단순 onehot으로 13,615차원 — 다른 데이터셋의 252배 / ∞배.
- 선형(LR), 트리(RF, XGB) 모델은 고차원 sparse에서 신호 약화 (r 0.09~0.27).
- MLP만 r=0.61로 강함 (학습 capacity 충분).
- SVC는 v3.2 `class_weight='balanced'` 도입 후 F1=0.4235 픽스 케이스 0건 (S4 해결).

### H3. F1 macro 둔감 — 기각

- Telco 슬라이스에서 metric별 r:
  - F1 macro: 0.284
  - accuracy: 0.086 (가장 약함, Churn 75:25 불균형 영향)
  - AUC: 0.133

- F1 macro가 다른 두 지표보다 강함. 가설 기각.

## 결정

Telco r 약함을 **본질적 데이터셋 한계**로 인정한다.

근거:
- Telco의 cat 컬럼 17개 × 다양한 levels = 13,615차원 onehot
- 본 실험의 단순 OHE 전처리 + 하이퍼파라미터 고정이 Telco 같은 고차원 sparse에서 작동하기 어려움
- 추가 코드 수정 (target encoding, 임베딩, 차원 축소)은 본 캡스톤 범위 초과

## 한계 명시 (발표/보고서)

다음 문장을 README, 04 execution log, 발표자료에 추가:

> Telco는 onehot 후 13,615차원의 sparse 데이터로, 본 실험의 단순 전처리 + 고정 하이퍼파라미터
> setup에서 선형·트리 모델의 학습이 상대적으로 불안정하다. MLP를 제외하면 모델별 r이 0.09~0.31에
> 그쳐 데이터셋 단위 r이 +0.28에 머문다. **이 한계는 DSC의 결함이 아닌 ML setup의 한계**로
> 해석되며, target encoding 또는 차원 축소 도입 시 회복 여지가 있다.

## 영향받은 파일

- `documents/decisions/ADR-010-Telco-비유의-데이터셋-한계.md` (이 문서)
- `documents/reports/20260427-02-v4-시뮬결과보고.md` (시뮬 결과)
- `README.md` (한계 섹션 — Phase 7에서 처리)
- `results/04_execution_log.md` (자동 생성, Phase 8에서 갱신)
