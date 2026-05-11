# ADR-015: LLM 가중치 생성기 + 튜닝/held-out dataset 분리 검증 체계

- **일자**: 2026-05-11
- **상태**: 확정 (구현 시작 전 사전 등록)
- **선행**: ADR-009 (v4 절대품질 지표), ADR-011 (강한 버전 framework), ADR-014 (이미지 cell 사전등록)
- **부분 대체**: ADR-011 §2-4 "사전 등록한 weights를 결과 보고 변경하지 않음", §4.위험1 "데이터 기반 weight tuning 금지", §6.1 "각 cell의 메트릭 정의식은 사전 등록 후 결과 따라 변경 금지" 중 *가중치 freeze 부분*, §6.3 "데이터 기반 weight 학습 금지" → 본 ADR로 대체. **정의식 freeze 원칙은 유지**.
- **후속**: `documents/plans/20260427-02-DSC-Framework-v5-마스터플랜.md` (가중치 freeze 절 + 데이터셋 분리 절 갱신 필요)

---

## 0. 한 줄 결정

> 가중치는 운영 시 LLM이 `(data_type, task, dataset metadata)`만 보고 생성. 학술 검증은 *튜닝 dataset* 에서 메커니즘 개선 후 *held-out dataset* 1회 측정으로 합격선 통과 여부를 보는 acceptance test로 재정의한다.

---

## 1. 배경

### 1-1. 사전등록 가중치(ADR-011)의 운영 한계

ADR-011 §2-4 / §6.1 / §6.3은 가중치를 사전 등록하고 결과 보고 변경 금지(F1 순환 논증 회피 목적). 그러나 운영 환경에서:

- 가중치 한 세트가 다양한 (dataset, task) 조합에 적합하지 않음. 팀 회의(2026-05-07) 이후 논의에서 동일 가중치를 universal CSV profiler에 적용 시 약신호 dataset의 floor effect, 강신호 dataset의 ceiling effect가 동시에 발생함이 지적됨.
- 사용자가 임의 dataset을 가져옴 → 사전등록 가중치가 그 dataset에 부적합한 경우 신호 손실.

### 1-2. 운영 SLA 제약

- 운영 응답 시간 ≤ 1분.
- r 측정은 dataset 변형 N개에 ML 학습 + 평가 필요 → 운영 시점 r 기반 가중치 조정은 ground truth 부재로 원리상 불가능.

### 1-3. 검증 framing 재정의

| 시점 | 검증 framing |
|---|---|
| v4까지 | "DSC 점수 ↔ ML 성능 상관관계 입증" — 수동, 사전등록 가중치 한 세트의 신호 *존재성* lower bound 증명 |
| 본 ADR 이후 | "framework + LLM 가중치 생성기가 held-out dataset에서 r ≥ 합격선 안정 달성" — 능동, 메커니즘 *성능* acceptance test |

캡스톤 목적은 r 최대화이며, 학술적 정직성은 *최종 보고용 dataset의 격리*로 충족된다. ML 엔지니어링의 train/val/test 분리 원리와 동일.

---

## 2. 결정

### 2-1. 운영 메커니즘

운영 시:

1. 사용자 dataset + `(data_type, task)` 입력
2. LLM에 `(data_type, task, dataset metadata)` 전달
   - dataset metadata = schema, dtypes, missing rate, basic statistics, target distribution summary
   - 실측 label/오염 정보는 차단 (overfitting 방지)
3. LLM이 가중치 세트 출력 (해당 cell의 metric set에 대응)
4. (정의식 + 출력 가중치)로 DSC 점수 계산
5. SLA: 1~4번까지 ≤ 1분. **운영 시 r 측정 안 함**.

### 2-2. 학술 검증 구조

데이터셋을 두 집단으로 분리:

| 집단 | 용도 | 사용 자유도 |
|---|---|---|
| **튜닝 set** | LLM prompt / 가중치 결정 로직 / 정의식 가중치 범위 등 개선 | r 반복 측정 가능, dataset 자유 사용 |
| **held-out set** | 최종 보고용 r 측정 | 메커니즘 freeze 후 **단 1회 측정**, 측정 후 어떤 조정도 금지 |

분리 원칙:

- held-out set에 한 번이라도 측정한 dataset은 튜닝 set으로 강등 (held-out 재사용 금지)
- held-out 신규 확보 시 ADR 또는 plan 문서에 사전 등록 후 사용
- 분류/회귀/이미지 cell마다 held-out dataset 목록을 독립 관리

### 2-3. 합격선 (사전 등록)

| Cell | 합격선 (Pearson r, held-out) | 근거 |
|---|---|---|
| tabular × classification | **r ≥ 0.50** | v4 사람 가중치 r=0.598 baseline. LLM 가중치는 이 lower bound 이상 유지 |
| tabular × regression | **Phase 1 진입 전 사전 등록** | 회귀 cell metric set 검증 후 ADR 또는 plan에 명시 |
| image × classification | **Phase 1 진입 전 사전 등록** | ADR-014 후속으로 명시 |

합격선 변경 시 후속 ADR 필요.

### 2-4. 사전 등록 대상 재정의

본 ADR 후 사전 등록 대상은 다음으로 변경:

| 항목 | 사전 등록 여부 | 비고 |
|---|---|---|
| 정의식 (cell별 metric set) | ✅ freeze 유지 | 변경 시 ADR 필요 (ADR-009/011 계승) |
| 가중치 | ❌ **freeze 해제** | LLM 출력에 위임 |
| 합격선 (Pearson r 임계) | ✅ freeze | 변경 시 ADR 필요 |
| held-out dataset 목록 | ✅ freeze | 변경 시 ADR 필요 |
| LLM prompt template | ✅ held-out 측정 직전 freeze | 튜닝 단계에서는 자유 |
| LLM 모델/버전/temperature | ✅ held-out 측정 직전 freeze | reproducibility 확보 |

---

## 3. 대안 검토

| 대안 | 검토 | 결론 |
|---|---|---|
| A. ADR-011 사전등록 가중치 유지 | 운영 dataset 다양성 대응 불가, 단일 가중치 세트로 universal profiler 불가능 | 기각 |
| B. 운영 시 r 측정 후 LLM 피드백 루프 | 단일 dataset에서 r은 정의되지 않음 (변형 N개 + 학습 + 평가 필요), 1분 SLA 불가 | 기각 |
| C. 학술 검증 시 r 보고 가중치 튜닝, 같은 dataset에서 보고 | 같은 dataset 튜닝/보고는 일반화 못 함 (overfitting) | 기각 |
| **D. LLM 가중치 + 튜닝/held-out 분리** | overfitting 회피 + 운영 일치 + 합격선으로 의미 검증 | **채택** |

---

## 4. 위험과 대응

### 위험 1 — held-out dataset 고갈

여러 측정 사이클 후 held-out set 소진 가능.

**대응**:
- 캡스톤 발표는 cell당 1회 측정으로 충분.
- held-out dataset 목록을 cell별 ≥ 2개 사전 등록.
- 추가 필요 시 신규 dataset 확보 + 후속 ADR 또는 plan에 등록.

### 위험 2 — LLM 출력 분산

같은 입력에 LLM 재호출 시 가중치 변동, 운영 시 사용자별로 r이 흔들림.

**대응**:
- LLM 호출 시 temperature = 0 또는 명시 고정.
- 튜닝 단계에서 동일 입력 N=10 호출 → 가중치 분산 및 r 분산 측정. 분산이 사전 등록 임계 초과 시 prompt 강화 또는 LLM 교체.
- held-out 보고 시 r 단일 값과 가중치 분산 모두 보고.

### 위험 3 — LLM이 비합리적 가중치 출력

한 차원에 가중치 1.0 몰아주기 등 무의미한 출력 가능.

**대응**:
- LLM 출력 형식 제약: `sum(w) = 1`, 각 `w ∈ [0.01, 0.6]`, JSON schema validation.
- 출력 검증 실패 시 cell별 사전등록된 등가중(`1/k`) 또는 v4 가중치로 fallback.
- fallback 발생 비율도 합격선 대상에 포함 (예: fallback ≤ 5%).

### 위험 4 — held-out r이 합격선 미달

**대응**:
- 미달 시 LLM/prompt/정의식 개선 후 *새로운* held-out dataset에서 재측정.
- 미달 측정에 사용된 held-out dataset은 튜닝 set으로 강등 (재사용 금지).
- 캡스톤 기간 내 합격선 미달 지속 시 Limitations에 명시 + 합격선 조정(후속 ADR) 또는 framework 한계 보고.

### 위험 5 — LLM 외부 의존성 (API 비용, 가용성, 라이선스)

**대응**:
- CLAUDE.md "외부 요청 사전 검증 필수" 원칙 준수: LLM API 실재·요금·라이선스 확인 후 코드 반영.
- 학교 카드 결제 가능 (메모리 `project_capstone_scope_expansion.md` 참조).
- 발표 재현성 확보: prompt + 모델 버전 + 응답 raw log를 결과 산출물과 함께 저장.

### 위험 6 — reviewer가 "튜닝/held-out 분리 진정성" 의심

> "튜닝 결과를 보고 held-out도 슬쩍 조정한 거 아닌가?"

**대응**:
- held-out dataset 목록 + 합격선 + prompt freeze 시점을 본 ADR/후속 ADR에 사전 등록 (날짜·git commit hash 명시 가능).
- held-out 측정 raw log를 git 커밋 (측정 시점·코드 상태 추적 가능).
- 측정 후 *튜닝 set으로 강등* 원칙으로 cherry-picking 자동 차단.

---

## 5. 영향받는 파일/문서

### 즉시 (본 ADR로 발생)

- `documents/decisions/ADR-011-Task-conditional-Framework-강한버전채택.md` (헤더에 partial supersede 표시 추가)
- `documents/decisions/ADR-015-LLM-가중치-생성기-튜닝-heldout-분리.md` (본 문서)
- `documents/plans/20260427-02-DSC-Framework-v5-마스터플랜.md` (가중치 freeze 절 + 데이터셋 분리 절 갱신 — 본 ADR 직후 작업)

### Phase 진입 시

- `dsc_framework/llm_weight_generator.py` (신규 — LLM 호출 + 출력 검증 + fallback)
- `dsc_framework/prompts/weight_generator_v1.txt` (신규 — prompt template, freeze 대상)
- 노트북에 LLM 가중치 호출 cell 통합 (분류·회귀·이미지)
- held-out dataset 추가: 분류 cell ≥ 1개, 회귀 cell ≥ 1개, 이미지 cell ≥ 1개 (튜닝 set과 별도 확보, 사전 등록)
- `results/heldout_*/` 디렉토리 신설 (튜닝 결과와 물리적으로 분리)

---

## 6. 검증 원칙 (CLAUDE.md 준수)

1. **정의식 freeze 유지** (ADR-009/011 계승)
2. **튜닝 dataset과 held-out dataset 분리** (본 ADR 신규)
3. **held-out 1회 측정 원칙** — 측정 후 어떤 메커니즘 조정도 금지, 사용된 dataset은 튜닝 set으로 강등
4. **합격선 사전 등록** — cell별로 본 ADR 또는 후속 ADR에 명시
5. **LLM prompt / 모델 / temperature freeze 시점** — held-out 측정 직전. 변경 시 후속 ADR 필요
6. **외부 LLM API 사용 전 실재·요금·라이선스 확인** (CLAUDE.md 준수)
7. **held-out 측정 raw log + prompt + 모델 버전 git 커밋** — reproducibility

---

## 7. 결정 후 즉시 다음 작업

1. ADR-011 헤더에 partial supersede 표시 추가 (가중치 freeze 조항만)
2. `plans/20260427-02` 가중치 freeze 절 갱신 + 데이터셋 분리 절 추가
3. 회귀 cell·이미지 cell 합격선 + held-out dataset 사전 등록 (별도 plan 문서 또는 후속 ADR)
4. LLM weight generator 구현 (튜닝 set 단계, prompt v1)
5. 분류 cell에서 LLM 가중치 vs 사람 가중치 r 비교 (튜닝 set 기준 prototype)

---

**문서 끝.**
