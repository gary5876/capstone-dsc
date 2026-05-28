# ADR-016: 텍스트 cell 사전등록 (v5 framework 확장)

- **상태**: Proposed (사전등록 시작, 구현 대기). ADR-015 가중치 freeze 해제 원칙 그대로 적용 — 정의식·메트릭 라인업·합격선·polluter·dataset/모델 후보는 freeze, 가중치는 LLM weight generator의 fallback 용도.
- **결정일**: 2026-05-28
- **선행 결정**:
  - ADR-011 (강한 버전 framework)
  - ADR-013 (dsc_framework 분리)
  - ADR-014 (이미지 cell 사전등록 — 본 ADR이 패턴 미러)
  - ADR-015 (LLM 가중치 생성기 + 튜닝/held-out 분리)
- **결정 근거**: 캡스톤 범위 확장 요구 (2026-05-07 팀 회의), 이미지 cell Phase 1 합격선 통과 (2026-05-22, r=0.45) 후 framework instance 추가 필요. text × classification은 NLP 분야 최대 표준 task로 framework 일반성 주장 강화.
- **후속 ADR**: ADR-017 (텍스트 cell 회귀 트랙 사전등록) — 회귀 데이터셋·target 메트릭 재정의는 별도 ADR로 분기.
- **마스터플랜**: `documents/plans/20260528-01-텍스트-cell-마스터플랜.md` (작성 예정)

---

## 1. 컨텍스트

v5 마스터플랜(2026-04-27, ADR-011)에서 framework는 cell instance가 다양할수록 학술적 일반성 주장이 강해진다. 현재 확정 instance는 다음과 같다.

1. ✅ tabular × classification (v4 완료, r=0.598)
2. 🔨 tabular × regression (Phase 1 노트북 완료, Phase 2 검증 대기)
3. ✅ image × classification (Phase 1 통과, default 가중치 r=0.45 / tuned held-out r=0.57)
4. **이 ADR: text × classification** (사전등록)
5. (후속 ADR-017) text × regression

text × classification은 framework "강한 버전"의 4번째 instance가 되며, 이미지 cell과 동일한 외부 정합성 검증(메트릭 정의식 cell-specific + r ≥ 0.4 합격선)을 통과해야 정식 cell로 승격된다.

## 2. 결정

텍스트 cell의 분류 트랙(text × classification)을 v5 framework의 4번째 instance로 정식 사전등록한다. 본 ADR에서 freeze한 항목은 결과 보고 후 변경 금지(가중치 제외).

## 3. 사전등록 항목 (구현 전 freeze)

### 3-1. 데이터셋 후보 (3개, 튜닝 set)

| 데이터셋 | HuggingFace ID | 클래스 | train | test | 특징 |
|---|---|---:|---:|---:|---|
| **AG News** | `fancyzhx/ag_news` | 4 | 120,000 | 7,600 | 4-class 토픽 분류 (World/Sports/Business/Sci-Tech). 짧은 텍스트(평균 ~50토큰) |
| **IMDB** | `stanfordnlp/imdb` | 2 | 25,000 | 25,000 | binary 감성. 긴 텍스트(평균 ~230토큰, max 13K자) — `max_length=256` truncation 적용 |
| **20 Newsgroups** | `SetFit/20_newsgroups` | 20 | 11,300 | 7,530 | 20-class 토픽. class_balance 변동을 폭넓게 측정 (cf. AG News 4-class) |

**선정 근거**:
- AG News: 표준 텍스트 분류 벤치마크 (Zhang et al. NIPS 2015), 짧은 텍스트로 baseline 모델도 학습 가능
- IMDB: 긴 문서 sentiment, transformer truncation 효과 측정 가능 (Maas et al. ACL 2011)
- 20 Newsgroups: 20-class로 class_balance polluter 신호 폭이 큼 (UCI 표준)

**선정 제외**:
- DBpedia / Yahoo Answers / TREC / SST-2 → **held-out set 후보**로 보존 (ADR-015 분리 원칙)
- 다국어 데이터셋 → English-only 범위 (캡스톤 한계)

**held-out set은 별도 사전등록 문서로 분기**: `documents/plans/20260528-02-텍스트-cell-합격선-heldout-사전등록.md` (작성 예정). 본 ADR에서는 튜닝 set만 freeze.

### 3-2. 모델 후보 (5개)

| 모델 | 출처 | 파라미터 | 학습 방식 | GPU 요구 |
|---|---|---:|---|:---:|
| **LogReg + TF-IDF** | scikit-learn | ~10K-100K | 전통 NLP baseline. unigram+bigram TF-IDF (max_features=20K) → LogisticRegression | CPU |
| **TextCNN** | 직접 정의 | <1M | 3 conv kernel sizes (3/4/5) + max-pool + FC. fastText-class baseline | CPU 가능, T4 권장 |
| **DistilBERT-base-uncased** | `distilbert/distilbert-base-uncased` | 66M | finetune, mean-pool head | T4 |
| **BERT-base-uncased** | `google-bert/bert-base-uncased` | 110M | finetune, [CLS] head | T4 |
| **RoBERTa-base** | `FacebookAI/roberta-base` | 125M | finetune, [CLS] head | T4/L4 |

**선정 근거**:
- 5개 짝수 맞춤 (분류/회귀/이미지 cell과 framework 일관성)
- 아키텍처 다양성: 전통(LogReg) / CNN / Transformer 3종
- 파라미터 폭: 100K → 125M, DSC 신호가 모델 크기와 무관함을 입증
- 모두 학술 표준 (이미지 cell의 ResNet/EfficientNet/MobileNet/ViT 대응)

**하이퍼파라미터 사전등록**:

| 항목 | LogReg+TFIDF | TextCNN | Transformer 3종 |
|---|---|---|---|
| max_length | — (TF-IDF) | 256 | 256 |
| batch_size | — | 64 | 32 |
| epochs | — (closed-form) | 10 | 3 (sanity), 5 (정식) |
| optimizer | — | Adam | AdamW |
| lr | — | 1e-3 | 2e-5 |
| weight_decay | — | 0 | 0.01 |
| pretrained | — | False (random init) | True (HuggingFace `from_pretrained`) |
| random_state | 42 | 42 | 42 |

**TextCNN 사양**: embedding_dim=128 (random init), 3 parallel Conv1d (kernel=3/4/5, filters=100 each), GlobalMaxPool, Dropout(0.5), FC.

### 3-3. Polluter 라인업 (5개)

| Polluter | 정의 | level 의미 | 구현 출처 |
|---|---|---|---|
| **completeness_text** | 단어 일부를 `[MASK]` 토큰으로 치환 | level = 마스킹 비율 (0.1~0.95) | 신규 (텍스트 본문 변형) |
| **noise_injection_text** | character-level typo (랜덤 insert/delete/swap) | level = 문자별 노이즈 확률 (0~0.5) | 신규 (텍스트 본문 변형) |
| **word_shuffle** | 문장 내 어순 무작위 셔플 | level = shuffle 강도 (0=원본, 1=완전 셔플). image blur의 sequential 정보 손상 대응 | 신규 (텍스트 본문 변형) |
| **class_balance** | 클래스별 샘플 수 불균형화 | tabular cell과 동일 | `dq4ai.polluters.ClassBalancePolluter` wrapper |
| **label_swap** | label 무작위 swap | level = swap 비율 | `dq4ai.polluters.TargetAccuracyPolluter` 분류 분기 wrapper |

**구현 방침**:
- **텍스트 본문 변형 3종** (`completeness_text` / `noise_injection_text` / `word_shuffle`): dq4ai의 tabular polluter 인터페이스(`pollute(df: DataFrame) -> DataFrame`)가 list[str] 입력에 부적합 + 텍스트 단위 변형은 컬럼/픽셀 단위 변형과 별개 → `dsc_framework/text_polluters/`에 신규 모듈로 작성. 이미지 cell이 5종 전부 신규 작성한 전례(`image_polluters/`)와 동일 결정.
- **라벨·target 계열 2종** (`class_balance` / `label_swap`): 라벨 조작 자체는 데이터 유형 무관 → dq4ai의 `ClassBalancePolluter` / `TargetAccuracyPolluter` 로직을 그대로 호출. wrapper가 `(texts, labels) → pd.DataFrame({'text': ..., 'label': ...}) → dq4ai polluter → (texts_out, labels_out)` 형태로 변환.

**통일 인터페이스**: `pollute(texts, labels, level, random_state) -> (texts_polluted, labels_polluted)`. `texts`는 list[str], `labels`는 list[int] 또는 1-D ndarray.

**Level grid (Phase 1 학습용)**: `[0.0, 0.1, 0.25, 0.5, 0.75, 0.9]` 6단계. 이미지 cell과 동일.

### 3-4. 메트릭 정의식 (10개, 사전등록 — 가중치는 fallback)

이미지 cell과 동일하게 9 + 1 = 10개 메트릭. 9개는 cell 패턴 유지(정의식 cell-specific), 1개 신설(`sample_quality_text`).

> **ADR-015 원칙 적용**: 본 표의 "가중치 (사전등록)" 컬럼은 LLM weight generator의 *fallback* 가중치로 역할 한정. 운영·검증의 실제 가중치는 LLM 출력. fallback 정의 freeze는 유지 (변경 시 후속 ADR 필요).

| 메트릭 | 정의식 | 가중치 (fallback) |
|---|---|---:|
| `completeness_text` | 1 - (`[MASK]`/`[PAD]`/빈 토큰 비율 평균) | 0.15 |
| `uniqueness` | 1 - (정규화된 텍스트 hash 중복 비율). 정규화 = lower + whitespace 압축 | 0.10 |
| `validity` | 비-empty + UTF-8 디코딩 성공 + 최소 토큰 수(≥1) 비율 | 0.05 |
| `consistency` | 텍스트 길이(token count) 5-bucket 분포 entropy의 보수 | 0.05 |
| `outlier_ratio` | token count의 IQR-based outlier 비율 보수 | 0.05 |
| `class_balance` | tabular cell과 동일 (min_ratio / ideal_ratio) | 0.10 |
| `feature_correlation` | DistilBERT mean-pool embedding(768-d) 차원 간 cosine 상관 고상관(>0.95) 비율 보수 | 0.05 |
| `label_consistency` | k-NN(k=5) DistilBERT embedding 라벨 일관성 (chance 보정) | 0.20 |
| `feature_informativeness` | embedding → label MI / H(Y) | 0.10 |
| **`sample_quality_text`** (NEW) | type-token ratio(TTR) + length_adequacy 결합 점수 | 0.15 |

**가중치 합 = 1.00**.

**메트릭 정의식 상세 (사전등록 freeze)**:

1. **`completeness_text`**: 입력 텍스트마다 `[MASK]`, `[PAD]`, 빈 문자열, 길이 < 1 토큰을 "비완전 토큰"으로 카운트. 텍스트별 비완전 비율 → 전체 평균 → 1.0 - 평균.

2. **`uniqueness`**: 텍스트를 `text.lower().strip()`, multi-space → single-space로 정규화 후 SHA-256 hash. 중복 hash가 있는 경우 `1.0 - (중복 개수 - 고유 개수) / 전체`.

3. **`validity`**: 텍스트가 (a) None/빈 문자열 아니고 (b) UTF-8 디코딩 가능하고 (c) `text.split()` 기준 토큰 수 ≥ 1. 모든 조건 통과 비율.

4. **`consistency`**: 토큰 수를 `[0,10), [10,50), [50,150), [150,500), [500,∞)` 5-bucket으로 binning. 분포의 normalized Shannon entropy → 1.0 - normalized entropy.

5. **`outlier_ratio`**: 토큰 수의 Q1, Q3 계산 → IQR fence(1.5 × IQR) 바깥 비율. 1.0 - outlier_ratio.

6. **`class_balance`**: `min(class_counts) / sum(class_counts) ÷ (1 / n_classes)`, clip to [0, 1].

7. **`feature_correlation`**: DistilBERT-base-uncased mean-pool embedding (768-d) → 768×768 corrcoef 행렬 → 상삼각 |corr| > 0.95 비율 → 1.0 - 비율. embedding 추출은 sample_cap=1000 (메트릭 7-9 공유).

8. **`label_consistency`**: embedding → StandardScaler → k=5 NearestNeighbors. 각 샘플의 k-NN 라벨 중 자기 라벨과 같은 비율 (raw). chance 보정: `chance = sum(class_prop²)`, `score = (raw - chance) / (1 - chance)`, clip to [0, 1]. (이미지 cell과 동일 공식.)

9. **`feature_informativeness`**: `sklearn.feature_selection.mutual_info_classif(embedding, labels)` 합 / H(Y), clip to [0, 1].

10. **`sample_quality_text`** (NEW):
    - `ttr_score = unique_tokens / max(1, total_tokens)` per text, 평균 → min(1, ttr × 2.0)
    - `length_score = min(1, token_count / target_len)` per text, 평균 (target_len=20, Phase 1 보고 후 조정 가능)
    - 결합 = (ttr_score + length_score) / 2

`sample_quality_text`는 텍스트 cell만의 신설 지표 — tabular/image cell 어느 것과도 1:1 매칭이 안 되는 text-intrinsic 품질 측정. (image의 `sample_quality_image`(blur+contrast)와 같은 지위.)

**ADR-011 강한 버전 원칙 준수**: 차원 이름이 같아도 정의식이 cell마다 다름.
- `feature_correlation`: tabular = 컬럼 Pearson, image = ResNet embedding cosine, **text = DistilBERT embedding cosine**
- `outlier_ratio`: tabular = 수치형 IQR, image = mean intensity IQR, **text = token count IQR**
- `consistency`: tabular = format/encoding 일관성, image = mode/size entropy, **text = 길이 bucket entropy**

### 3-5. 평가 메트릭

- 분류 cell·이미지 cell과 동일: accuracy, macro F1
- 다중 클래스 표준: 학습 후 test set 평가
- **합격선**: r(DSC, accuracy) ≥ 0.4 (Pearson + Spearman 모두)
- **Polluter hold-out**: 5개 중 4개 PASS (각 polluter 1개씩 빼고 r 측정 → r ≥ 0.4 유지)
- **모델별 r**: 5/5 양의 r (모델 무관 신호)

## 4. 구현 범위 (캡스톤 한계)

**최소 범위 (반드시)**:
- 3 dataset × 5 model × 5 polluter × 6 level = 450 학습 + 18 baseline (clean) = 468건
- T4 GPU 기준 transformer 1 finetune ≈ 5~20분 (epoch=3, batch=32, max_len=256), CNN/LogReg는 1분 이내
- 총 GPU 시간 추정 = 450 × (3 × 10분 + 2 × 1분) / 5 ≈ 30~50시간. Colab Pro+ 또는 Pay-as-you-go GPU 예산 필요

**Stretch goal (시간 여유 시)**:
- 4번째 데이터셋(Yahoo Answers 등) 추가
- DeBERTa-v3 / ELECTRA 모델 추가
- 다국어 cell (mteb/amazon_reviews_multi 비 영어 subset)

**범위 외 (Limitations 명시)**:
- 다국어 텍스트
- 토큰 시퀀스 길이 > 512 (long-document) — truncation으로 처리
- streaming / online 평가

## 5. dsc_framework 통합

```
dsc_framework/
├── shared_metrics.py            # tabular 6개 (변경 없음)
├── classification_cell.py        # tabular × classification
├── regression_cell.py            # tabular × regression
├── image_cell.py                 # image × classification
├── image_polluters/              # 이미지 polluter 5종
├── text_cell.py                  # text × classification (NEW, 본 ADR)
├── text_polluters/               # 텍스트 polluter 5종 (NEW 디렉토리)
│   ├── __init__.py
│   ├── completeness_text.py      # 신규 (단어 → [MASK])
│   ├── noise_injection_text.py   # 신규 (character-level typo)
│   ├── word_shuffle.py           # 신규 (어순 셔플)
│   ├── class_balance_text.py     # dq4ai.ClassBalancePolluter wrapper
│   └── label_swap_text.py        # dq4ai.TargetAccuracyPolluter wrapper
├── column_detection.py
├── data_type_detection.py        # text vs tabular vs image 감지 확장
└── router.py                     # ('text', 'classification') → text_cell 분기 추가
```

`router.py`의 `select_profile` 확장:
- `('tabular', 'classification')` → classification_cell
- `('tabular', 'regression')` → regression_cell
- `('image', 'classification')` → image_cell
- `('text', 'classification')` → **text_cell (NEW)**
- (ADR-017) `('text', 'regression')` → text_regression_cell

`data_type_detection.py`의 텍스트 감지 규칙(사전등록):
- 입력이 `list[str]` 또는 pandas Series of str
- 평균 토큰 수 ≥ 5 (numeric column이 string으로 저장된 경우 제외)
- 토큰 다양성(TTR) ≥ 0.1 (categorical column 제외)

## 6. 검증 통과 기준 (Phase 2)

이미지 cell과 동일:
- Pearson r(DSC, accuracy) ≥ 0.4 (튜닝 set 각 데이터셋별)
- Spearman ρ ≥ 0.4
- Polluter hold-out 4/5 PASS
- 모델 5/5 양의 r
- POOLED r은 보조 (Heinrich 2018 R1 — cross-dataset 비교 부정당)

**held-out 합격선**: ADR-015 + 별도 plan 문서(작성 예정)에서 LLM 가중치 생성기 출력의 held-out 측정 1회로 최종 평가. 본 ADR의 fallback 가중치는 LLM이 응답 실패 시 대체용.

## 7. 외부 의존성 사전 검증 (CLAUDE.md 준수)

본 ADR 작성 시점(2026-05-28) HuggingFace에서 확인:

| 자원 | 상태 | 비고 |
|---|---|---|
| `fancyzhx/ag_news` | ✅ 접근 가능 | 127.6K rows, 4 classes, Parquet |
| `stanfordnlp/imdb` | ✅ 접근 가능 | 100K rows (50K labeled + 50K unsup), binary |
| `SetFit/20_newsgroups` | ✅ 접근 가능 | 18.8K rows, 20 classes |
| `distilbert/distilbert-base-uncased` | ✅ 접근 가능 | 66M params, 768-d, Apache 2.0 |
| `google-bert/bert-base-uncased` | 사전 확인 권장 | 표준 베이스라인, finetune 예정 |
| `FacebookAI/roberta-base` | 사전 확인 권장 | 표준 베이스라인, finetune 예정 |

BERT/RoBERTa는 워낙 표준이라 큰 위험은 없지만 Phase 1 진입 직전 1회 더 확인.

**Python 패키지 의존성**: `transformers >= 4.30`, `datasets >= 2.10`, `sentence-transformers` (옵션, embedding 빠른 추출용), `torch >= 2.0`. `requirements.txt`에 추가 예정.

## 8. 사전등록 freeze 항목

본 ADR로 다음 항목이 freeze됨 (결과 확인 후 변경 금지):

- 튜닝 데이터셋 3개 (held-out은 별도 plan 문서)
- 모델 5개 (하이퍼파라미터 포함)
- Polluter 5개 (정의 + level grid + 의미)
- 메트릭 10개 (**정의식만 freeze** — 가중치는 LLM 위임, 본 §3-4 표는 fallback 정의로 보존, ADR-015 partial supersede)
- 평가 메트릭 (accuracy + macro F1)
- 합격선 r ≥ 0.4 (Pearson + Spearman 모두)
- Polluter hold-out 4/5 PASS 기준
- 모델별 양의 r 5/5 기준

추가 가중치 조정(fallback 자체 수정)·정의식 수정 시 ADR-016a로 명시적 기록.

## 9. 후속 작업

1. **마스터플랜 작성** (`documents/plans/20260528-01-텍스트-cell-마스터플랜.md`): Phase 1 (인프라, ~7일) / Phase 2 (검증, ~5일) / Phase 3 (통합, ~2일)
2. **held-out 사전등록** (`documents/plans/20260528-02-텍스트-cell-합격선-heldout-사전등록.md`): DBpedia / Yahoo Answers / TREC 3종 후보 검토 및 freeze
3. **ADR-017** (text × regression 사전등록): Yelp Full / Amazon Reviews EN / SST-5 + target_distribution_quality·target_smoothness·mutual_info_regression 메트릭 재정의
4. **구현**: `dsc_framework/text_cell.py`, `text_polluters/` 모듈, 노트북 4종 (Phase 2 검증용)

---

**관련 문서**:
- `documents/decisions/ADR-014-이미지-cell-사전등록.md` (본 ADR이 미러)
- `documents/decisions/ADR-015-LLM-가중치-생성기-튜닝-heldout-분리.md` (가중치 위임 원칙)
- `documents/plans/20260427-02-DSC-Framework-v5-마스터플랜.md` (framework 확장 마스터)
- `documents/plans/20260508-01-이미지-cell-마스터플랜.md` (Phase 의존성 패턴 참고)
- `documents/reports/20260522-01-이미지-cell-합격선-달성-진단.md` (default vs tuned 가중치 갭 분석)
