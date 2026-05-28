# 텍스트 cell 합격선 + held-out dataset 사전등록

- **작성일**: 2026-05-28
- **선행**:
  - ADR-015 (LLM 가중치 생성기 + 튜닝/held-out 분리)
  - ADR-016 (text × classification 사전등록)
  - ADR-017 (text × regression 사전등록)
  - 20260511-01 (합격선 + held-out 사전등록, 기존 cell들 — 본 문서가 텍스트 cell로 확장)
- **목적**: 텍스트 cell 두 트랙(분류/회귀)의 합격선 + held-out dataset 목록 freeze. ADR-015 §2-3 원칙에 따라 측정 직전 prompt + LLM model + temperature도 별도 freeze.
- **변경 정책**: 본 문서 §1~§3의 freeze 항목은 변경 시 후속 ADR. §2의 "라이선스 미확인"·"Phase 4 진입 시 confirm" 표시 항목은 Phase 진입 시 출처 페이지 직접 확인 후 갱신 가능.

---

## 1. 합격선 (텍스트 cell)

| Cell | 합격선 (r ≥) | 보조 보고 | 근거 |
|---|---:|---|---|
| text × classification | **0.40** | p < 0.001, Spearman ρ 동반 | ADR-016 §6 통과 기준 = image cell ADR-014와 동일. 텍스트는 dataset shift 폭이 image cell보다 큼 (도메인 다양성) → conservative 합격선 |
| text × regression | **0.40** | p < 0.001, Spearman ρ + ADR-012 Degradation Index `m_deg` | ADR-017 §6 + ADR-012 보조 보고 의무. 텍스트 회귀 baseline R² 변동 폭 큼 → Degradation Index 병행 보고 |

**LLM 메커니즘 추가 합격 조건** (20260511-01 §1과 동일 — 텍스트 cell도 적용):
- LLM fallback 발생률 ≤ 5%
- 가중치 분산 (N=10 호출, 동일 입력): max coefficient of variation ≤ 0.20

합격선 변경 시 후속 ADR 필요.

---

## 2. held-out dataset 사전등록

각 트랙마다 ≥ 2개 후보 사전 확보. 튜닝 set과 절대 중복 금지. 측정 후 즉시 튜닝 set으로 강등.

### 2-1. text × classification

**튜닝 set** (ADR-016): `fancyzhx/ag_news`, `stanfordnlp/imdb`, `SetFit/20_newsgroups`

**held-out 후보 (3종, 모두 HuggingFace 접근 확인 완료, 2026-05-28)**:

| Dataset | HuggingFace ID | 행수 (train+test) | 클래스 | 선정 근거 |
|---|---|---:|---:|---|
| **DBpedia-14** | `fancyzhx/dbpedia_14` | 560K + 70K | 14 | 14-class topic. 클래스 수 vs 튜닝 set(AG=4, 20news=20) 중간. 라이선스 CC-BY-SA-3.0 |
| **Yahoo Answers Topics** | `community-datasets/yahoo_answers_topics` | 1.4M + 60K | 10 | 10-class topic, 도메인 Q&A로 신문/리뷰와 다름. 라이선스 미확인 (Phase 4 진입 시 확인) |
| **TREC (coarse)** | `CogComp/trec` | 5,452 + 500 | 6 | 6-class question type, short text(~10토큰), 도메인 의문문. 라이선스 academic |

**선정 기준**: 클래스 수 폭(6/10/14) × 텍스트 길이 폭(short TREC / medium DBpedia / long Yahoo) × 도메인 다양성(질문/Q&A/위키 ontology). 튜닝 set의 신문/리뷰/뉴스그룹과 겹치지 않게 골랐음.

### 2-2. text × regression

**튜닝 set** (ADR-017): `Yelp/yelp_review_full` (sample_cap 50K), `SetFit/amazon_reviews_multi_en` (sample_cap 200K), `SetFit/sst5`

**held-out 후보 (2종 confirmed + 1 candidate slot)**:

| Dataset | HuggingFace ID | 행수 | 타겟 | 입력 처리 | 선정 근거 |
|---|---|---:|---|---|---|
| **STS-B** | `sentence-transformers/stsb` | 5,750 + 1,380 | 유사도 0-1 (continuous) | sentence pair → `f"{s1} [SEP] {s2}"`로 joined single text | GLUE 표준 semantic similarity. 회귀 target이 0-1 continuous로 ordinal 5-star 튜닝 set과 분포 형태가 다름 → robust한 held-out 신호 |
| **App Reviews** | `sealuzh/app_reviews` | 288K (single split) | star 1-5 (1-5) | 그대로 단일 텍스트 | Android F-Droid 리뷰. 도메인 commerce(Amazon)/restaurant(Yelp)/movie(SST)와 무관한 모바일 앱. 라이선스 미확인 (Phase 4 진입 시 확인) |

**Candidate slot (Phase 4 진입 직전 freeze)**: 텍스트 회귀 single-text English benchmark pool이 협소함을 인정. 다음 3개 후보 중 1개를 Phase 4 진입 직전 선택해 등록:

| 후보 | 출처 | 비고 |
|---|---|---|
| `Convo-AI/Goodreads-books-reviews` | HuggingFace | 2026-05-28 401 응답. Phase 4 직전 접근성 재확인 |
| TripAdvisor reviews | HuggingFace 미러 검색 | 2026-05-28 검색 시 미러 401. 다른 미러 탐색 필요 |
| SemEval-2018 Task 1 EI-reg | CodaLab 외부 다운로드 | HuggingFace에는 E-c(분류)만 있음. EI-reg는 외부 사이트 |

**대안 (3종 확보 실패 시)**: held-out 2종(STS-B + App Reviews)으로 합격선 측정 + Limitations에 "텍스트 회귀 held-out pool 협소" 명시. 5 cell 중 4 cell은 3종, 1 cell은 2종 측정. ADR-015 §2-3은 "≥ 2개" 요구이므로 2종도 합법.

**선정 기준**: target range 다양성(0-1 continuous vs 1-5 ordinal) × 도메인 신규성(commerce 외).

---

## 3. LLM 호출 freeze 시점

20260511-01 §3과 동일 원칙. 텍스트 cell 추가 사항만 명시:

### 3-1. freeze 대상 항목

- LLM provider + 모델 ID (예: `anthropic/claude-sonnet-4-6`)
- temperature (사전등록 default 0.0)
- prompt template version 파일 (`dsc_framework/prompts/weight_generator_v?.txt`)
- WEIGHT_BOUNDS (현재 `[0.01, 0.60]`) + SUM_TOLERANCE (현재 `0.01`)
- 텍스트 cell 메트릭 키 셋:
  - 분류: `completeness_text`, `uniqueness`, `validity`, `consistency`, `outlier_ratio`, `class_balance`, `feature_correlation`, `label_consistency`, `feature_informativeness`, `sample_quality_text`
  - 회귀: `completeness_text`, `uniqueness`, `validity`, `consistency`, `outlier_ratio`, `target_distribution_quality`, `feature_correlation`, `target_smoothness`, `feature_informativeness_reg`, `sample_quality_text`

### 3-2. freeze 시점

Phase 4 (held-out 측정) 진입 직전. 진입 후 prompt 한 줄이라도 변경 시 held-out 측정 무효 → 다른 held-out 후보로 재측정 (튜닝 set 강등).

### 3-3. freeze 직전 sanity 권장

- 튜닝 set 1개(예: AG News)로 weight generation 5회 호출 → 가중치 평균/표준편차 보고
- fallback 발생 0회 확인 (1회라도 발생 시 prompt 수정 후 재검증)

---

## 4. 측정 절차 (Phase 4)

20260511-01 §4와 동일 원칙. 텍스트 cell 측정 시 다음 추가 작업 필요:

1. STS-B는 sentence pair → `f"{s1} [SEP] {s2}"` joined 처리. `text_cell_regression.py` 호출 시 그대로 list[str] 전달
2. App Reviews는 split이 train 하나뿐이므로 random_state=42로 sklearn.train_test_split로 80/20 분할 후 학습. 분할 자체는 본 문서에 freeze 됨
3. 회귀 cell은 ADR-012 Degradation Index 보조 보고 의무
4. POOLED r은 보조 (Heinrich 2018 R1 — cross-dataset 비교 부정당)

---

## 5. 외부 의존성 사전 검증 (CLAUDE.md 준수)

| 자원 | 상태 | 비고 |
|---|---|---|
| `fancyzhx/dbpedia_14` | ✅ 접근 가능 (2026-05-28) | 560K + 70K, 14-class |
| `community-datasets/yahoo_answers_topics` | ✅ 접근 가능 | 1.4M + 60K, 10-class. 라이선스 unknown — Phase 4 진입 시 확인 |
| `CogComp/trec` | ✅ 접근 가능 | 5.5K + 500, 6/50-class |
| `sentence-transformers/stsb` | ✅ 접근 가능 | 5.7K + 1.4K + 1.4K, pair input |
| `sealuzh/app_reviews` | ✅ 접근 가능 | 288K single split, 1-5 star. 라이선스 unknown — Phase 4 진입 시 확인 |
| 회귀 candidate 3종 | Phase 4 직전 재확인 | 2026-05-28 시점 일부 401/미러 미정 |

---

## 6. 사전등록 freeze 항목 (본 문서 발효)

- 합격선 (분류 r ≥ 0.40, 회귀 r ≥ 0.40)
- 분류 held-out 3종 (DBpedia-14, Yahoo Answers, TREC)
- 회귀 held-out 2종 sealed (STS-B, App Reviews) + 1 candidate slot
- LLM 호출 freeze 항목 리스트 (3-1)
- 측정 절차 추가 사항 (4. STS-B SEP join, App Reviews train/test 분할)
- LLM 메커니즘 추가 합격 조건 (fallback ≤ 5%, CV ≤ 0.20)

추가 변경 시 본 문서의 a-suffix 후속 ADR (예: 20260528-02a) 또는 supersede 문서 작성.

---

## 7. 관련 문서

- `documents/decisions/ADR-015-LLM-가중치-생성기-튜닝-heldout-분리.md` (튜닝/held-out 분리 원칙)
- `documents/decisions/ADR-016-텍스트-cell-사전등록.md` (분류 트랙 메트릭·polluter·모델 freeze)
- `documents/decisions/ADR-017-텍스트-cell-회귀트랙-사전등록.md` (회귀 트랙 freeze + Degradation Index 의무)
- `documents/plans/20260511-01-합격선-heldout-사전등록.md` (기존 cell들 held-out, 본 문서가 동일 패턴 미러)
- `documents/plans/20260528-01-텍스트-cell-마스터플랜.md` (Phase 의존성 + 일정)
