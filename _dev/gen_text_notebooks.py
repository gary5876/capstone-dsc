"""Generate 4 Phase-2 text cell notebook stubs.

이미지 cell 노트북 4종(01~04) 패턴 미러. 분류+회귀 합본 텍스트 노트북 stub 생성.
실행 셀은 GPU 의존이라 작성만 — 사용자가 Colab에서 실행.

Usage:
    python _dev/gen_text_notebooks.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / 'notebooks'


def md(text):
    return {
        'cell_type': 'markdown',
        'metadata': {},
        'source': text if isinstance(text, list) else text.splitlines(keepends=True),
    }


def code(src):
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': src if isinstance(src, list) else src.splitlines(keepends=True),
    }


def notebook(cells):
    return {
        'cells': cells,
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.10'},
        },
        'nbformat': 4,
        'nbformat_minor': 5,
    }


# ====================================================================
# 01_setup_and_baseline_text.ipynb
# ====================================================================
NB_01 = notebook([
    md("""# 01. Setup & Baseline (Text Cell)

**Phase 1 → Phase 2 진입**: HuggingFace datasets 로드 → DSC 베이스라인 (clean) → 모델 5종 베이스라인 metric 수집

DSC v5 framework — text × classification (ADR-016) + text × regression (ADR-017) 사전등록.

분류 튜닝 dataset 3종: `fancyzhx/ag_news` / `stanfordnlp/imdb` / `SetFit/20_newsgroups`
회귀 튜닝 dataset 3종: `Yelp/yelp_review_full` (50K) / `mteb/amazon_reviews_multi` 'en' (200K) / `SetFit/sst5`

---
"""),
    md("## 0. 환경 설정\n\nColab T4 GPU 가정. 학교 카드 결제 활성화 + HuggingFace 토큰 발급(필요 시) 완료 전제.\n"),
    code("""# requirements-text.txt 패키지 설치 (Colab 환경)
!pip install -q transformers>=4.30 datasets>=2.10 xgboost>=1.7 sentence-transformers

import sys, os, json
ROOT = '/content/drive/MyDrive/capstone/dsc'  # Colab Google Drive 마운트 경로 (조정)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset

from dsc_framework.text_cell import compute_dsc_text
from dsc_framework.text_cell_regression import compute_dsc_text_regression

print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')
"""),
    md("## 1. 튜닝 dataset 로드\n\nADR-016 §3-1 / ADR-017 §3-1 freeze.\n"),
    code("""# 분류 트랙
ag_news = load_dataset('fancyzhx/ag_news')
imdb    = load_dataset('stanfordnlp/imdb')
news20  = load_dataset('SetFit/20_newsgroups')

# 회귀 트랙 (Yelp/Amazon은 sample_cap 적용)
yelp    = load_dataset('Yelp/yelp_review_full')
amazon  = load_dataset('mteb/amazon_reviews_multi', 'en')
sst5    = load_dataset('SetFit/sst5')

print('sizes:',
    {'ag_news': len(ag_news['train']), 'imdb': len(imdb['train']),
     '20news': len(news20['train']), 'yelp': len(yelp['train']),
     'amazon': len(amazon['train']), 'sst5': len(sst5['train'])})
"""),
    md("""## 2. 샘플링 (ADR-017 §4 freeze)

Yelp / Amazon은 stratified random_state=42, train 50K (Yelp) / 200K (Amazon) / test 5K.
"""),
    code("""def stratified_sample(ds, label_key, n_per_split, seed=42):
    rng = np.random.RandomState(seed)
    df = ds.to_pandas()
    parts = [g.sample(min(len(g), n_per_split // df[label_key].nunique()), random_state=seed)
             for _, g in df.groupby(label_key)]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)

yelp_train_s  = stratified_sample(yelp['train'],   'label',  50000)
yelp_test_s   = stratified_sample(yelp['test'],    'label',   5000)
amazon_train_s= stratified_sample(amazon['train'], 'label', 200000)
amazon_test_s = stratified_sample(amazon['test'],  'label',   5000)
print('Yelp:', len(yelp_train_s), len(yelp_test_s), '| Amazon:', len(amazon_train_s), len(amazon_test_s))
"""),
    md("""## 3. DSC 베이스라인 (clean DSC, default 가중치)

ADR-015 원칙: 본 단계의 default DSC는 합격선 출발점 측정. 운영 가중치는 Phase 4 LLM 호출 결과 사용.
"""),
    code("""def df_to_text_label(ds, text_key='text', label_key='label'):
    if isinstance(ds, pd.DataFrame):
        return ds[text_key].tolist(), ds[label_key].tolist()
    return ds[text_key], ds[label_key]

# 분류 cell — clean baseline
for name, ds in [('ag_news', ag_news['train']),
                  ('imdb',   imdb['train']),
                  ('20news', news20['train'])]:
    texts, labels = df_to_text_label(ds)
    r = compute_dsc_text(texts[:5000], labels[:5000],
                         use_embeddings=True, sample_cap=1000, random_state=42)
    print(f"{name:8s}  DSC={r['score']}  grade={r['grade']}")

# 회귀 cell — clean baseline (target=label cast to float)
for name, ds in [('yelp_50k',   yelp_train_s),
                  ('amazon_200k', amazon_train_s),
                  ('sst5',       pd.DataFrame(sst5['train']))]:
    texts, lbs = df_to_text_label(ds)
    targets = [float(x) for x in lbs]
    r = compute_dsc_text_regression(texts[:5000], targets[:5000],
                                    use_embeddings=True, sample_cap=1000, random_state=42)
    print(f"{name:12s} DSC={r['score']}  grade={r['grade']}")
"""),
    md("""## 4. 모델 베이스라인 — clean 학습

5 모델 × 6 dataset baseline accuracy/R² 수집. 학습 시간 절약 위해 각 dataset의 train_sub(예: 5000)로 sanity. Phase 2 정식 학습은 03 노트북에서 풀 train으로 재실행.

다음 셀들은 GPU 환경에서 실행 — 본 노트북 stub은 함수 정의만 둠.
"""),
    code("""# Transformer head 분류 학습 (DistilBERT/BERT/RoBERTa 공유)
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
# 함수 정의는 03_training_text.ipynb에 풀 버전 작성
"""),
    code("""# TextCNN 분류 (random init embedding + 3-kernel conv)
# class TextCNN(torch.nn.Module): ...  # 03_training_text.ipynb 참조
"""),
    code("""# LogReg + TF-IDF baseline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

def logreg_tfidf_accuracy(train_texts, train_labels, test_texts, test_labels,
                          max_features=20000, ngram_range=(1, 2)):
    vec = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    Xtr = vec.fit_transform(train_texts)
    Xte = vec.transform(test_texts)
    clf = LogisticRegression(max_iter=2000, n_jobs=-1, random_state=42).fit(Xtr, train_labels)
    p = clf.predict(Xte)
    return accuracy_score(test_labels, p), f1_score(test_labels, p, average='macro')
"""),
    md("---\n\n다음: `02_pollution_and_dsc_text.ipynb` — 7 polluter × 6 level × 6 dataset 스윕.\n"),
])

# ====================================================================
# 02_pollution_and_dsc_text.ipynb
# ====================================================================
NB_02 = notebook([
    md("""# 02. Pollution & DSC (Text Cell)

ADR-016 §3-3 / ADR-017 §3-3 polluter 7종 × 6 level × 6 dataset 스윕 → DSC 점수 + polluter quality measure 수집.

Output: `results/text_cell_dsc_sweep.csv` (rows = dataset × polluter × level × seed)

---
"""),
    md("## 0. import + 데이터 (01에서 캐싱)\n"),
    code("""import sys, os
ROOT = '/content/drive/MyDrive/capstone/dsc'
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import json

from dsc_framework.text_cell import compute_dsc_text
from dsc_framework.text_cell_regression import compute_dsc_text_regression
from dsc_framework.text_polluters import (
    CompletenessTextPolluter, NoiseInjectionTextPolluter, WordShufflePolluter,
    ClassBalanceTextPolluter, LabelSwapTextPolluter,
    TargetDistributionSkewTextPolluter, TargetNoiseTextPolluter,
)
"""),
    md("## 1. 스윕 설정 (ADR-016/017 freeze)\n"),
    code("""LEVEL_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9]  # 6 단계, ADR-016 §3-3
SEEDS = [42, 7, 99]                                  # 3 seed, polluter hold-out 분석에 사용

CLASSIFICATION_POLLUTERS = {
    'completeness_text':    CompletenessTextPolluter,
    'noise_injection_text': NoiseInjectionTextPolluter,
    'word_shuffle':         WordShufflePolluter,
    'class_balance':        ClassBalanceTextPolluter,
    'label_swap':           LabelSwapTextPolluter,
}
REGRESSION_POLLUTERS = {
    'completeness_text':         CompletenessTextPolluter,
    'noise_injection_text':      NoiseInjectionTextPolluter,
    'word_shuffle':              WordShufflePolluter,
    'target_distribution_skew':  TargetDistributionSkewTextPolluter,
    'target_noise':              TargetNoiseTextPolluter,
}
"""),
    md("## 2. 분류 트랙 스윕\n"),
    code("""# datasets은 01 노트북에서 로드된 변수 그대로 사용 (Colab 메모리 공유)
def sweep_classification(texts, labels, dataset_name, polluters, seeds=SEEDS, levels=LEVEL_GRID):
    rows = []
    for pol_name, pol_cls in polluters.items():
        for seed in seeds:
            for lvl in levels:
                pol = pol_cls(lvl, random_seed=seed)
                t_p, l_p = pol.pollute(texts, labels)
                r = compute_dsc_text(t_p, l_p, use_embeddings=True,
                                     sample_cap=1000, random_state=seed)
                rows.append({
                    'dataset': dataset_name, 'polluter': pol_name,
                    'level': lvl, 'seed': seed,
                    'dsc_score': r['score'], **{k: r[k] for k in r if k not in ('score', 'grade')}
                })
    return pd.DataFrame(rows)
"""),
    code("""# 실제 sweep — Colab GPU에서 실행. dataset 당 ~10분 (DistilBERT embedding 추출 포함)
# results_cls = []
# for name, ds in [('ag_news', ag_news['train']), ('imdb', imdb['train']), ('20news', news20['train'])]:
#     texts, labels = ds['text'][:5000], ds['label'][:5000]
#     df = sweep_classification(texts, labels, name, CLASSIFICATION_POLLUTERS)
#     results_cls.append(df)
# df_cls = pd.concat(results_cls); df_cls.to_csv('results/text_cls_dsc_sweep.csv', index=False)
"""),
    md("## 3. 회귀 트랙 스윕\n"),
    code("""def sweep_regression(texts, targets, dataset_name, polluters, seeds=SEEDS, levels=LEVEL_GRID):
    rows = []
    for pol_name, pol_cls in polluters.items():
        for seed in seeds:
            for lvl in levels:
                pol = pol_cls(lvl, random_seed=seed)
                t_p, tg_p = pol.pollute(texts, targets)
                r = compute_dsc_text_regression(t_p, tg_p, use_embeddings=True,
                                                sample_cap=1000, random_state=seed)
                rows.append({
                    'dataset': dataset_name, 'polluter': pol_name,
                    'level': lvl, 'seed': seed,
                    'dsc_score': r['score'], **{k: r[k] for k in r if k not in ('score', 'grade')}
                })
    return pd.DataFrame(rows)
"""),
    code("""# 실제 sweep
# results_reg = []
# for name, ds in [('yelp_50k', yelp_train_s), ('amazon_200k', amazon_train_s),
#                  ('sst5', pd.DataFrame(sst5['train']))]:
#     texts = ds['text'].tolist(); targets = [float(x) for x in ds['label'].tolist()]
#     df = sweep_regression(texts[:5000], targets[:5000], name, REGRESSION_POLLUTERS)
#     results_reg.append(df)
# df_reg = pd.concat(results_reg); df_reg.to_csv('results/text_reg_dsc_sweep.csv', index=False)
"""),
    md("---\n\n다음: `03_training_text.ipynb` — 모델 학습 + accuracy/R² 수집.\n"),
])

# ====================================================================
# 03_training_text.ipynb
# ====================================================================
NB_03 = notebook([
    md("""# 03. Training (Text Cell)

ADR-016 §3-2 / ADR-017 §3-2 모델 5종 × polluter 5종 × level 6 × dataset 6 학습.

분류 트랙: LogReg+TFIDF / TextCNN / DistilBERT / BERT-base / RoBERTa-base
회귀 트랙: Ridge+TFIDF / XGBoost+TFIDF / TextCNN-Reg / DistilBERT-Reg / BERT-base-Reg

총 학습 건수: 6 dataset × 5 model × 5 polluter × 6 level + 6×5 baseline = 930건
T4 기준 transformer ≈ 5~20분 → 30~50시간 (분류) + 30~50시간 (회귀)

Output: `results/text_train_metrics.csv` (rows = dataset × model × polluter × level × seed)

---
"""),
    md("## 0. import + 모델 정의\n"),
    code("""import sys, os
ROOT = '/content/drive/MyDrive/capstone/dsc'
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, r2_score
"""),
    md("## 1. 모델 학습 함수\n\n각 모델은 (train_texts, train_labels, test_texts, test_labels) → metric 반환.\n"),
    code("""def train_logreg_tfidf(tr_t, tr_y, te_t, te_y, max_features=20000):
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    Xtr = vec.fit_transform(tr_t); Xte = vec.transform(te_t)
    clf = LogisticRegression(max_iter=2000, n_jobs=-1, random_state=42).fit(Xtr, tr_y)
    return accuracy_score(te_y, clf.predict(Xte))


def train_ridge_tfidf(tr_t, tr_y, te_t, te_y, max_features=20000, alpha=1.0):
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    Xtr = vec.fit_transform(tr_t); Xte = vec.transform(te_t)
    clf = Ridge(alpha=alpha, random_state=42).fit(Xtr, tr_y)
    return r2_score(te_y, clf.predict(Xte))


def train_xgb_tfidf(tr_t, tr_y, te_t, te_y, max_features=20000):
    from xgboost import XGBRegressor
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    Xtr = vec.fit_transform(tr_t); Xte = vec.transform(te_t)
    clf = XGBRegressor(max_depth=6, n_estimators=500, learning_rate=0.05,
                        random_state=42, n_jobs=-1).fit(Xtr, tr_y)
    return r2_score(te_y, clf.predict(Xte))
"""),
    code("""# TextCNN — 분류/회귀 head 교체
class TextCNN(nn.Module):
    def __init__(self, vocab_size, n_class, emb=128, kernels=(3, 4, 5), filters=100,
                 dropout=0.5, regression=False):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(emb, filters, k, padding=k // 2) for k in kernels])
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(filters * len(kernels), 1 if regression else n_class)
        self.regression = regression

    def forward(self, x):
        x = self.emb(x).transpose(1, 2)
        x = torch.cat([torch.max(torch.relu(c(x)), dim=2).values for c in self.convs], dim=1)
        x = self.drop(x)
        return self.fc(x).squeeze(-1) if self.regression else self.fc(x)


# 실제 학습 루프는 길어 별도 함수로
def train_textcnn(tr_t, tr_y, te_t, te_y, regression=False, epochs=10, batch=64, lr=1e-3):
    # tokenizer build, padding, train loop, eval
    # ... (구현 예정, GPU 환경에서)
    pass
"""),
    code("""# Transformer (DistilBERT/BERT/RoBERTa) — head 교체로 분류/회귀 둘 다 처리
def train_transformer(model_id, tr_t, tr_y, te_t, te_y, regression=False,
                       max_len=256, epochs=3, batch=32, lr=2e-5):
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        Trainer, TrainingArguments,
    )
    n_label = 1 if regression else len(set(tr_y))
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=n_label,
        problem_type='regression' if regression else 'single_label_classification')
    # tokenize, Trainer, train, eval
    # ... (구현 예정)
    pass
"""),
    md("## 2. 학습 스윕\n\n02의 sweep 결과(polluted texts/labels)를 직접 메모리에서 사용하거나 ↓ 처럼 csv에서 재구성.\n"),
    code("""# 학습 루프 — Colab GPU 환경에서 큐로 백그라운드 실행
# results = []
# for ds_name, (tr_texts, tr_y, te_texts, te_y) in datasets.items():
#     for pol_name, pol_cls in CLASSIFICATION_POLLUTERS.items():
#         for lvl in LEVEL_GRID:
#             pol = pol_cls(lvl, random_seed=42)
#             tr_p, tr_yp = pol.pollute(tr_texts, tr_y)
#             for model_name, train_fn in MODELS.items():
#                 metric = train_fn(tr_p, tr_yp, te_texts, te_y)
#                 results.append({...})
# pd.DataFrame(results).to_csv('results/text_train_metrics.csv', index=False)
"""),
    md("---\n\n다음: `04_scoreboard_text.ipynb` — r 분석, polluter hold-out, default vs tuned 가중치.\n"),
])

# ====================================================================
# 04_scoreboard_text.ipynb
# ====================================================================
NB_04 = notebook([
    md("""# 04. Scoreboard (Text Cell)

ADR-016 §6 / ADR-017 §6 검증 — 02의 DSC + 03의 train metric을 join → r 분석, polluter hold-out, default vs tuned 가중치 grid search, LLM weight generator sanity.

합격 기준:
- Pearson r(DSC, accuracy/R²) ≥ 0.4 (튜닝 set 각 dataset 별)
- Spearman ρ ≥ 0.4
- Polluter hold-out 4/5 PASS
- 모델 5/5 양의 r

회귀 트랙은 ADR-012 Degradation Index 보조 보고 추가.

---
"""),
    md("## 0. import + 결과 csv 로드\n"),
    code("""import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

dsc_cls   = pd.read_csv('results/text_cls_dsc_sweep.csv')
dsc_reg   = pd.read_csv('results/text_reg_dsc_sweep.csv')
train_m   = pd.read_csv('results/text_train_metrics.csv')

# join key: (dataset, polluter, level, seed)
JOIN_KEYS = ['dataset', 'polluter', 'level', 'seed']
"""),
    md("## 1. dataset별 r(DSC, metric) 측정 — 합격선 검토\n"),
    code("""def compute_r_per_dataset(dsc_df, train_df, metric_col):
    j = dsc_df.merge(train_df, on=JOIN_KEYS)
    rows = []
    for ds_name in j['dataset'].unique():
        sub = j[j['dataset'] == ds_name]
        r_p, p_p = pearsonr(sub['dsc_score'], sub[metric_col])
        r_s, p_s = spearmanr(sub['dsc_score'], sub[metric_col])
        rows.append({
            'dataset': ds_name, 'n': len(sub),
            'pearson': r_p, 'p_pearson': p_p,
            'spearman': r_s, 'p_spearman': p_s,
            'pass_r040': bool(r_p >= 0.4 and r_s >= 0.4),
        })
    return pd.DataFrame(rows)

# 분류
print('=== TEXT × CLASSIFICATION ===')
print(compute_r_per_dataset(dsc_cls, train_m[train_m['task'] == 'classification'], 'accuracy'))
# 회귀
print('=== TEXT × REGRESSION ===')
print(compute_r_per_dataset(dsc_reg, train_m[train_m['task'] == 'regression'], 'r2'))
"""),
    md("## 2. polluter hold-out (4/5 PASS)\n"),
    code("""def polluter_holdout(dsc_df, train_df, metric_col):
    j = dsc_df.merge(train_df, on=JOIN_KEYS)
    polluters = j['polluter'].unique()
    rows = []
    for held in polluters:
        sub = j[j['polluter'] != held]
        for ds_name in sub['dataset'].unique():
            sd = sub[sub['dataset'] == ds_name]
            r, _ = pearsonr(sd['dsc_score'], sd[metric_col])
            rows.append({'held_out': held, 'dataset': ds_name, 'pearson': r, 'pass': r >= 0.4})
    return pd.DataFrame(rows)

# hold-out summary: dataset × polluter
"""),
    md("## 3. 모델별 r (5/5 양의 r)\n"),
    code("""def per_model_r(train_df, dsc_df, metric_col):
    j = dsc_df.merge(train_df, on=JOIN_KEYS)
    rows = []
    for m in j['model'].unique():
        sub = j[j['model'] == m]
        r, p = pearsonr(sub['dsc_score'], sub[metric_col])
        rows.append({'model': m, 'pearson': r, 'p': p, 'positive': r > 0})
    return pd.DataFrame(rows)
"""),
    md("""## 4. Default → tuned 가중치 grid search

이미지 cell의 04 노트북 패턴 미러. dead/live 메트릭 분포 진단 + grid search로 더 높은 r 달성 가능한 가중치 찾기.

ADR-015 원칙: 본 결과는 fallback 가중치의 **갭 분석** 용도. 운영·검증 정식 가중치는 Phase 4의 LLM 호출 결과 사용.
"""),
    code("""# dead 메트릭 진단 — std < 0.01인 메트릭은 r에 영향 거의 없음
METRIC_KEYS = [
    'completeness_text', 'uniqueness', 'validity', 'consistency', 'outlier_ratio',
    'class_balance', 'feature_correlation', 'label_consistency',
    'feature_informativeness', 'sample_quality_text',
]
for k in METRIC_KEYS:
    if k in dsc_cls.columns:
        print(f"{k:30s} std={dsc_cls[k].std():.4f}  range=[{dsc_cls[k].min():.3f}, {dsc_cls[k].max():.3f}]")
"""),
    code("""# 가중치 grid search — live 메트릭만 자유 변수, dead는 default 유지
# (이미지 cell의 score_image_v2.py 패턴)
# Scipy.optimize.minimize로 -r 최소화 → tuned 가중치 찾기

from scipy.optimize import minimize

def grid_search_weights(dsc_df, train_df, metric_col, live_keys, default_weights):
    # ... 구현 예정 — 이미지 cell의 _dev/score_image_v2.py를 텍스트로 이식
    pass
"""),
    md("## 5. LLM weight generator sanity\n\nPhase 4 진입 직전 sanity (튜닝 set 1개로 5회 호출 → CV 측정).\n"),
    code("""from dsc_framework.llm_weight_generator import AnthropicWeightGenerator  # 미구현 시 구현 필요

# gen = AnthropicWeightGenerator(data_type='text', task='classification')
# results = [gen.generate({'schema': ...}) for _ in range(5)]
# cv = max(np.std([r.weights[k] for r in results]) / (np.mean([r.weights[k] for r in results]) + 1e-9)
#          for k in DEFAULT_WEIGHTS_TEXT.keys())
# print('weight CV:', cv, 'fallback rate:', sum(r.used_fallback for r in results) / 5)
"""),
    md("## 6. ADR-012 Degradation Index (회귀 트랙 보조 보고)\n"),
    code("""# from dsc_framework import compute_dsc_degradation
# m_deg = compute_dsc_degradation(polluted_dsc_dict, clean_dsc_dict)
# 회귀 트랙은 r(DSC, R²) absolute + r(DSC_deg, R²_deg/R²_clean) preservation 두 값 보고
"""),
    md("---\n\n**다음 단계**: 합격선 통과 시 Phase 4 — held-out 측정 (`plans/20260528-02-텍스트-cell-합격선-heldout-사전등록.md`).\n"),
])


for name, nb in [('01_setup_and_baseline_text', NB_01),
                  ('02_pollution_and_dsc_text', NB_02),
                  ('03_training_text', NB_03),
                  ('04_scoreboard_text', NB_04)]:
    path = NB_DIR / f'{name}.ipynb'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f'wrote {path}')

print('done.')
