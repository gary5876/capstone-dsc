# ADR-017: 텍스트 cell 회귀 트랙 사전등록 (v5 framework 확장)

- **상태**: Proposed (사전등록 시작, 구현 대기). ADR-015 가중치 freeze 해제 원칙 그대로 적용.
- **결정일**: 2026-05-28
- **2026-05-28 amendment (당일)**: §3-1 Amazon Reviews EN 데이터셋의 미러를 `mteb/amazon_reviews_multi` (subset `en`) → `SetFit/amazon_reviews_multi_en`으로 교체. 사유: `mteb/...`는 dataset script(.py) 기반이라 신버전 `datasets` 라이브러리에서 "Dataset scripts are no longer supported" RuntimeError. 같은 원본 코퍼스(Amazon Reviews Multi)의 영어 parquet 미러로 데이터 내용 동일. sample_cap(200K train / 5K test) 그대로 적용 가능. 사전 검증 한계로 ADR 작성 당시 인지 못함
- **선행 결정**:
  - ADR-011 (강한 버전 framework — cell마다 정의식 재정의)
  - ADR-012 (Degradation Index 보조 지표, 회귀 cell에 그대로 승계)
  - ADR-015 (LLM 가중치 생성기 + 튜닝/held-out 분리)
  - ADR-016 (텍스트 cell 분류 트랙 사전등록 — 본 ADR과 짝)
- **결정 근거**: text × classification만으로는 framework "(data_type, task) → metric_set" 매핑이 task 축에서 데이터유형 1개에만 검증되는 한계. text × regression을 추가하면 (tabular, image, text) × (classification, regression) 매트릭스에서 text 행이 완성되어 framework 일반성 주장 강화.
- **선행 cell 패턴**: tabular × regression cell 메트릭 재정의(`dq4ai/dsc_engine_regression_v5.py`, 마스터플랜 §3) 그대로 승계 — `class_balance → target_distribution_quality`, `label_consistency → target_smoothness`, `feature_informativeness → mutual_info_regression`.

---

## 1. 컨텍스트

ADR-016에서 text × classification은 사전등록 완료. 본 ADR은 같은 데이터 유형(text)의 회귀 트랙을 추가한다. 회귀 트랙은 다음 두 가지 점에서 분류 트랙과 다르다.

1. **타겟의 자료형이 연속형(또는 ordinal)** — class_balance / label_consistency / feature_informativeness 메트릭의 정의식이 적용 불가
2. **평가 메트릭이 R²** — accuracy 대신 R²(음수 clip to 0)로 합격선 측정

tabular × regression cell이 이미 이 메트릭 재정의 패턴을 검증했으므로 동일 패턴을 text 임베딩 공간에서 그대로 재사용한다.

## 2. 결정

텍스트 cell의 회귀 트랙(text × regression)을 v5 framework의 5번째 instance로 정식 사전등록한다. 본 ADR에서 freeze한 항목은 결과 보고 후 변경 금지(가중치 제외).

## 3. 사전등록 항목 (구현 전 freeze)

### 3-1. 데이터셋 후보 (3개, 튜닝 set)

| 데이터셋 | HuggingFace ID | 타겟 | train | test | 특징 |
|---|---|---|---:|---:|---|
| **Yelp Review Full** | `Yelp/yelp_review_full` | 별점 0-4 (1-5 매핑) | 650,000 | 50,000 | 5-star ordinal regression. 평균 ~120토큰. 클래스당 균일 130K/10K |
| **Amazon Reviews EN** | `SetFit/amazon_reviews_multi_en` | 별점 0-4 | 200,000 | 5,000 (val) + 5,000 (test) | 5-star ordinal regression. Amazon Reviews Multi 코퍼스의 영어 미러. parquet 포맷 (dataset script 비의존). Apache 2.0 |
| **SST-5** | `SetFit/sst5` | 감성 0-4 (very neg → very pos) | 8,540 | 2,210 | 5-class 미세 감성. ordinal. 짧은 텍스트(평균 ~20토큰) |

**선정 근거**:
- Yelp Full: 텍스트 회귀의 사실상 표준 (Zhang et al. NIPS 2015) — 충분한 샘플, 균일 분포로 baseline 강함
- Amazon EN: 다른 도메인(commerce review), Yelp와 직접 비교 가능. domain transfer 신호 측정
- SST-5: 짧은 텍스트 + ordinal — Yelp/Amazon(긴 review)의 length-domain 보완. tree 기반 sentiment 표준

**target ordinal 처리**: 0-4 별점을 float으로 캐스팅 후 regression 학습. R² 계산 시 별점 그대로 사용 (rescale 없음). 이는 ordinal regression의 표준 처리.

**선정 제외 (held-out 후보로 보존)**:
- SemEval-2017 Task 5 (financial sentiment, -1~+1 continuous) — held-out
- STS-B (Semantic Textual Similarity, 0-5, 문장쌍 input) — held-out, pair-input은 단일 입력 cell과 architecture 충돌
- Helpfulness regression (Amazon helpfulness votes)
- IMDB rating (binary 데이터셋이라 회귀 부적합)

**held-out set 사전등록**: 분류 트랙과 마찬가지로 별도 plan 문서로 분기 — `documents/plans/20260528-02-텍스트-cell-합격선-heldout-사전등록.md` (작성 예정).

### 3-2. 모델 후보 (5개)

| 모델 | 출처 | 학습 방식 | GPU 요구 |
|---|---|---|:---:|
| **Ridge + TF-IDF** | scikit-learn | unigram+bigram TF-IDF(max_features=20K) → Ridge(alpha=1.0). 회귀 baseline | CPU |
| **XGBoostReg + TF-IDF** | xgboost | XGBRegressor 위에 TF-IDF feature. tabular regression cell의 XGBR 대응 | CPU |
| **TextCNN-Reg** | 직접 정의 | ADR-016 TextCNN 아키텍처 + linear regression head (no softmax) | CPU 가능 |
| **DistilBERT-Reg** | `distilbert/distilbert-base-uncased` | mean-pool + linear head, MSELoss로 finetune | T4 |
| **BERT-base-Reg** | `google-bert/bert-base-uncased` | [CLS] + linear head, MSELoss로 finetune | T4 |

**선정 근거**:
- 5개 짝수 + 아키텍처 다양성 (전통/tree/CNN/Transformer 2종)
- tabular regression cell의 LinearReg / RandomForestReg / XGBReg / SVR / MLPReg에 대응. text 도메인에서는 RandomForest+TFIDF가 비효율이라 XGBR로 단일화.
- Ridge는 tabular regression의 Linear+L2 대응. RoBERTa는 분류 트랙에 사용하고 회귀에는 BERT-base로 단순화 (모델 수 줄여 학습 시간 절약).

**하이퍼파라미터 사전등록**:

| 항목 | Ridge+TFIDF | XGB+TFIDF | TextCNN-Reg | Transformer 2종 |
|---|---|---|---|---|
| max_length | — | — | 256 | 256 |
| batch_size | — | — | 64 | 32 |
| epochs | — | — | 10 | 3 |
| optimizer | — | — | Adam | AdamW |
| lr | — | — | 1e-3 | 2e-5 |
| loss | — | — | MSE | MSE |
| ridge_alpha | 1.0 | — | — | — |
| xgb_max_depth | — | 6 | — | — |
| xgb_n_estimators | — | 500 | — | — |
| xgb_lr | — | 0.05 | — | — |
| pretrained | — | — | False | True (HF `from_pretrained`) |
| random_state | 42 | 42 | 42 | 42 |

### 3-3. Polluter 라인업 (5개)

분류 트랙(ADR-016)의 polluter 중 `class_balance`·`label_swap` 2개를 회귀용으로 재정의. 나머지 3개(텍스트 본문 변형)는 동일.

| Polluter | 정의 | level 의미 | 구현 출처 |
|---|---|---|---|
| **completeness_text** | 단어 일부를 `[MASK]` 토큰으로 치환 | level = 마스킹 비율 (0.1~0.95) | 신규 (ADR-016 공유) |
| **noise_injection_text** | character-level typo | level = 문자별 노이즈 확률 | 신규 (ADR-016 공유) |
| **word_shuffle** | 어순 무작위 셔플 | level = shuffle 강도 | 신규 (ADR-016 공유) |
| **target_distribution_skew** | target 분포를 한쪽으로 편향 (Q3 이상 제거) | level = 제거 비율 (편향 정도) | `dq4ai.polluters.TargetDistributionSkewPolluter` wrapper |
| **target_noise** | target에 Gaussian noise 추가 (회귀 전용) | level = noise σ / target_std | `dq4ai.polluters.TargetAccuracyPolluter` 회귀 분기 wrapper |

**구현 방침**:
- 텍스트 본문 변형 3종은 ADR-016의 신규 모듈을 그대로 import해 공유
- target 계열 2종은 dq4ai polluter를 `(texts, labels) → pd.DataFrame → polluter → (texts_out, targets_out)` 형태로 wrap. target은 float으로 캐스팅. wrapper 파일 위치는 `dsc_framework/text_polluters/target_distribution_skew.py`, `target_noise.py`

### 3-4. 메트릭 정의식 (10개, 사전등록 — 가중치는 fallback)

분류 트랙(ADR-016) 10개 메트릭 중 3개를 회귀용으로 재정의. 나머지 7개는 정의식 동일.

| 메트릭 | 정의식 | 가중치 (fallback) | vs 분류 트랙 |
|---|---|---:|---|
| `completeness_text` | 동일 | 0.15 | 동일 |
| `uniqueness` | 동일 | 0.10 | 동일 |
| `validity` | 동일 | 0.05 | 동일 |
| `consistency` | 동일 (token count 5-bucket entropy 보수) | 0.05 | 동일 |
| `outlier_ratio` | 동일 (token count IQR) | 0.05 | 동일 |
| **`target_distribution_quality`** (재정의) | target 값을 10-bin equal-width binning → 분포의 normalized Shannon entropy | 0.10 | `class_balance` 대응 |
| `feature_correlation` | 동일 (DistilBERT 768-d cosine 상관) | 0.05 | 동일 |
| **`target_smoothness`** (재정의) | k-NN(k=5, embedding 공간) → 이웃 target 값의 std / target_std 보수 (1 - clip) | 0.20 | `label_consistency` 대응 |
| **`feature_informativeness_reg`** (재정의) | `mutual_info_regression(embedding, target)` 합 / log(target 10-bin) 정규화, clip to [0, 1] | 0.10 | `feature_informativeness` 대응 |
| `sample_quality_text` | 동일 (TTR + length_adequacy 결합) | 0.15 | 동일 |

**가중치 합 = 1.00**.

**재정의 메트릭 상세 (사전등록 freeze)**:

1. **`target_distribution_quality`**:
   ```
   bin_edges = linspace(target.min(), target.max(), 11)  # 10 bins
   counts, _ = histogram(target, bins=bin_edges)
   probs = counts / counts.sum()
   probs = probs[probs > 0]
   ent = -(probs * log(probs)).sum()
   score = ent / log(len(probs)) if len(probs) > 1 else 1.0
   ```
   균일 분포 = 1.0, 편향 = 0에 가까움. tabular regression cell의 정의와 동일 공식.

2. **`target_smoothness`**:
   ```
   nn = NearestNeighbors(n_neighbors=k+1).fit(embedding_std)
   _, idx = nn.kneighbors(embedding_std)
   neighbor_targets = target[idx[:, 1:]]  # 자기 제외
   local_std = neighbor_targets.std(axis=1).mean()
   score = 1 - clip(local_std / target.std(), 0, 1)
   ```
   유사한 텍스트끼리 target도 유사 = high smoothness (≈1). tabular regression cell의 정의와 동일 공식.

3. **`feature_informativeness_reg`**:
   ```
   from sklearn.feature_selection import mutual_info_regression
   mi = mutual_info_regression(embedding, target, discrete_features=False)
   target_h = log(n_unique_bins_of_target)  # 10
   score = clip(mi.sum() / target_h, 0, 1)
   ```
   tabular regression cell과 동일.

**ADR-011 강한 버전 원칙 준수 (회귀 트랙 추가 사례)**:
- `target_smoothness`: tabular = 수치형 컬럼 k-NN 라벨 거리, **text = DistilBERT 임베딩 공간 k-NN 라벨 거리**
- `feature_informativeness_reg`: tabular = 수치형 컬럼 MI, image = embedding MI, **text = DistilBERT embedding MI**

### 3-5. 평가 메트릭

- tabular × regression cell과 동일: R² (음수 clip to 0)
- **합격선**: r(DSC, R²) ≥ 0.4 (Pearson + Spearman 모두)
- **Polluter hold-out**: 5개 중 4개 PASS
- **모델별 r**: 5/5 양의 r
- **보조 지표**: ADR-012 Degradation Index — `m_deg = max(0, 1 − R²_polluted / R²_clean)`, 약신호 데이터셋(특히 SST-5 short text) floor effect 회피용. tabular regression cell의 `compute_dsc_degradation` 함수 재사용.

## 4. 구현 범위 (캡스톤 한계)

**최소 범위 (반드시)**:
- 3 dataset × 5 model × 5 polluter × 6 level = 450 학습 + 18 baseline = 468건
- T4 GPU 기준 transformer finetune ≈ 5~20분 (epoch=3, batch=32, max_len=256)
- 총 GPU 시간 추정 ≈ 30~50시간 (분류 트랙과 동급)
- Yelp Full 650K는 sample_cap=50K로 다운샘플링 (메모리 + 시간 절약, 사전등록)

**Yelp Full / Amazon EN 샘플링 사전등록**:
- train sample_cap = 50,000 (각 별점 균등, stratified)
- test sample_cap = 5,000 (균등 stratified)
- random_state = 42

이 샘플링은 결과 재현용 freeze (변경 시 ADR-017a).

**Stretch goal (시간 여유 시)**:
- STS-B 추가 (pair-input wrapping)
- DeBERTa-v3 regression head 추가
- 다국어 cell (Amazon Reviews 다른 언어 subset)

**범위 외 (Limitations 명시)**:
- ranking metric (NDCG, MAP)
- non-ordinal continuous target (e.g., 임의 실수 score)
- 토큰 시퀀스 길이 > 512

## 5. dsc_framework 통합

```
dsc_framework/
├── text_cell.py                    # text × classification (ADR-016)
├── text_cell_regression.py         # text × regression (NEW, 본 ADR)
├── text_polluters/
│   ├── completeness_text.py        # 신규 (ADR-016 공유)
│   ├── noise_injection_text.py     # 신규 (ADR-016 공유)
│   ├── word_shuffle.py             # 신규 (ADR-016 공유)
│   ├── target_distribution_skew.py # dq4ai.TargetDistributionSkewPolluter wrapper (회귀 전용)
│   ├── target_noise.py             # dq4ai.TargetAccuracyPolluter 회귀 분기 wrapper
│   ├── class_balance_text.py       # dq4ai.ClassBalancePolluter wrapper (ADR-016 분류 전용)
│   └── label_swap_text.py          # dq4ai.TargetAccuracyPolluter 분류 분기 wrapper (ADR-016)
└── router.py                       # ('text', 'regression') 분기 추가
```

`router.py`:
- `('text', 'classification')` → text_cell (ADR-016)
- `('text', 'regression')` → text_cell_regression (본 ADR)

text_cell.py와 text_cell_regression.py는 embedding 추출 함수(`_extract_features_text`)를 공유. shared helper로 분리 권장.

## 6. 검증 통과 기준 (Phase 2)

분류 트랙과 동일:
- Pearson r(DSC, R²) ≥ 0.4 (튜닝 set 각 데이터셋별)
- Spearman ρ ≥ 0.4
- Polluter hold-out 4/5 PASS
- 모델 5/5 양의 r
- POOLED는 보조 (Heinrich 2018 R1)

**Degradation 지표 보고**: ADR-012에 따라 absolute r(DSC, R²) + preservation r(DSC_deg, R²_deg/R²_clean) 두 값 모두 보고. SST-5처럼 baseline R²가 낮은 경우 preservation이 주신호가 될 수 있음.

## 7. 외부 의존성 사전 검증 (CLAUDE.md 준수)

본 ADR 작성 시점(2026-05-28) HuggingFace에서 확인:

| 자원 | 상태 | 비고 |
|---|---|---|
| `Yelp/yelp_review_full` | ✅ 접근 가능 | 700K rows, 5-star, Parquet, 라이선스 Yelp Dataset Agreement (학술 사용 OK) |
| `SetFit/amazon_reviews_multi_en` | ✅ 접근 가능, parquet | 210K English subset (Amazon Reviews Multi 코퍼스). Apache 2.0. 2026-05-28 Phase 2 진입 시 `mteb/amazon_reviews_multi`가 dataset script(.py) 기반이라 신버전 `datasets` 라이브러리에서 차단됨이 확인되어 본 미러로 교체 |
| `SetFit/sst5` | ✅ 접근 가능 | 11.9K rows, 5-class ordinal |
| `distilbert/distilbert-base-uncased` | ✅ 접근 가능 | ADR-016과 공유 |
| `google-bert/bert-base-uncased` | 사전 확인 권장 | Phase 1 진입 직전 확인 |

**Python 패키지 추가**: `xgboost >= 1.7`, `scikit-learn >= 1.2`. transformers/datasets/torch는 ADR-016과 공유.

## 8. 사전등록 freeze 항목

본 ADR로 다음 항목이 freeze됨:

- 튜닝 데이터셋 3개 + 샘플링 cap (Yelp/Amazon 50K train / 5K test)
- 모델 5개 (하이퍼파라미터 포함)
- Polluter 5개 (정의 + level grid + 의미)
- 메트릭 10개 (정의식만 freeze, 가중치는 LLM 위임)
- 평가 메트릭 (R², 음수 clip to 0)
- 합격선 r ≥ 0.4 (Pearson + Spearman 모두)
- Polluter hold-out 4/5 PASS 기준
- 모델별 양의 r 5/5 기준
- ADR-012 Degradation Index 보조 보고 의무

추가 변경 시 ADR-017a.

## 9. 후속 작업

1. **마스터플랜** (`documents/plans/20260528-01-텍스트-cell-마스터플랜.md`): ADR-016과 합본 — 분류/회귀 두 트랙의 Phase 1/2/3 일정
2. **held-out 사전등록** (`documents/plans/20260528-02-텍스트-cell-합격선-heldout-사전등록.md`): 분류·회귀 각 3종 held-out 데이터셋 freeze
3. **구현**:
   - `dsc_framework/text_cell.py` (분류, ADR-016)
   - `dsc_framework/text_cell_regression.py` (회귀, 본 ADR)
   - `text_polluters/` 7개 모듈
   - Phase 2 검증 노트북 4종 (분류 + 회귀 별도 또는 합본)
4. **LLM weight generator prompt 확장**: 현재 prompt가 tabular/image만 인지 → text 케이스 추가 (`dsc_framework/prompts/weight_generator_v2.txt`)

---

**관련 문서**:
- `documents/decisions/ADR-016-텍스트-cell-사전등록.md` (분류 트랙, 본 ADR의 짝)
- `documents/decisions/ADR-011-Task-conditional-Framework-강한버전채택.md` (강한 버전 원칙)
- `documents/decisions/ADR-012-Degradation-Index-보조지표-도입.md` (회귀 보조 지표)
- `documents/decisions/ADR-015-LLM-가중치-생성기-튜닝-heldout-분리.md` (가중치 위임 원칙)
- `dq4ai/dsc_engine_regression_v5.py` (tabular regression cell — 메트릭 재정의 패턴 출처)
