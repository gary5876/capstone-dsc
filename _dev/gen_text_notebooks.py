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
    md("## 0. 환경 설정\n\nColab T4 GPU 가정. dsc/ 위치는 자동 검색 (G드라이브 어디에 있어도 OK).\n"),
    code("""# ============================================================
# 0-1. Drive 마운트 + dsc/ 자동 검색 + sys.path 등록
# ============================================================
from google.colab import drive
drive.mount('/content/drive')

import os, sys, glob, json
import numpy as np
import pandas as pd
import torch


def _find_dsc_base():
    \"\"\"G드라이브에서 dsc_framework/__init__.py가 들어있는 디렉토리 찾기.

    1) 알려진 후보 우선 (빠른 경로)
    2) 깊이 3까지 명시 glob (recursive=True는 큰 Drive에서 느림)
    Returns: 찾은 경로 또는 None.
    \"\"\"
    root = '/content/drive/MyDrive'
    if not os.path.isdir(root):
        return None
    fast = [
        f'{root}/capstone/dsc',
        f'{root}/dsc',
        f'{root}/capstone-dsc',
    ]
    for c in fast:
        if os.path.isfile(f'{c}/dsc_framework/__init__.py'):
            return c
    for pattern in [
        f'{root}/*/dsc_framework/__init__.py',
        f'{root}/*/*/dsc_framework/__init__.py',
        f'{root}/*/*/*/dsc_framework/__init__.py',
    ]:
        for hit in glob.glob(pattern):
            return os.path.dirname(os.path.dirname(hit))
    return None


BASE = _find_dsc_base()
if BASE is None:
    drive_root = '/content/drive/MyDrive'
    listing = os.listdir(drive_root) if os.path.isdir(drive_root) else []
    raise RuntimeError(
        'dsc_framework/ 폴더를 G드라이브에서 못 찾음.\\n'
        '확인 사항:\\n'
        '  1) G드라이브 클라이언트가 sync 완료 상태인지 (commit 직후면 잠시 대기 후 재시도).\\n'
        '  2) Drive 마운트가 됐는지 — `!ls /content/drive/MyDrive` 출력 확인.\\n'
        f'  현재 /content/drive/MyDrive 안 내용: {listing[:20]}\\n'
        '  3) 위 1~2 모두 OK인데도 실패면 사용자 Drive에 dsc_framework 폴더 자체가 없음 → push/sync 재확인.'
    )

RESULTS_DIR = f'{BASE}/results'
DATA_DIR = f'{BASE}/data/text'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

if BASE not in sys.path:
    sys.path.insert(0, BASE)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'BASE: {BASE}')
fw_contents = sorted(f for f in os.listdir(f'{BASE}/dsc_framework') if not f.startswith('_'))
print(f'  dsc_framework 폴더 ({len(fw_contents)}개): {fw_contents}')

REQUIRED_FILES = [
    'shared_metrics.py', 'classification_cell.py', 'regression_cell.py',
    'image_cell.py', 'text_cell.py', 'text_cell_regression.py',
    'column_detection.py', 'data_type_detection.py', 'router.py',
    'text_polluters', 'image_polluters',
]
missing = [f for f in REQUIRED_FILES if not os.path.exists(f'{BASE}/dsc_framework/{f}')]
if missing:
    raise RuntimeError(
        f'dsc_framework/ 폴더는 있지만 다음 파일들이 sync 안 됨: {missing}\\n'
        '→ G드라이브 클라이언트에서 sync 완료될 때까지 대기 후 재실행.\\n'
        '→ 강제 sync: G드라이브 폴더 열고 새로고침 또는 클라이언트 재시작.\\n'
        '→ Colab Drive view가 stale일 수 있음: drive.flush_and_unmount() 후 재마운트.'
    )
print(f'device: {device}, torch: {torch.__version__}')
"""),
    code("""# ============================================================
# 0-2. 의존성 설치 (Colab 환경) — 패키지 설치 후 다음 셀부터 import
# ============================================================
%pip install -q transformers>=4.30 datasets>=2.10 xgboost>=1.7
"""),
    code("""# ============================================================
# 0-3. dsc_framework / datasets import — 위 셀 실행 완료 후
# ============================================================
import warnings
warnings.simplefilter('default')

from datasets import load_dataset

# dsc_framework는 robust __init__.py라 일부 cell 누락 시 warning만 발생.
# text_cell이 진짜 못 import되면 ModuleNotFoundError raise — 그 경우 진단.
try:
    from dsc_framework.text_cell import compute_dsc_text
    from dsc_framework.text_cell_regression import compute_dsc_text_regression
except ModuleNotFoundError as e:
    fw = f'{BASE}/dsc_framework'
    have = sorted(os.listdir(fw)) if os.path.isdir(fw) else []
    raise RuntimeError(
        f'text_cell import 실패: {e}\\n'
        f'dsc_framework/ 안 파일: {have}\\n'
        '→ shared_metrics.py / text_cell.py 등 의존 파일이 Drive에 있는지 확인. '
        'sync 미완료면 잠시 대기 후 재실행.'
    ) from e

print('dsc_framework import OK.')
print('compute_dsc_text:', compute_dsc_text)
print('compute_dsc_text_regression:', compute_dsc_text_regression)
"""),
    md("## 1. 튜닝 dataset 로드\n\nADR-016 §3-1 / ADR-017 §3-1 freeze.\n"),
    code("""# 분류 트랙
ag_news = load_dataset('fancyzhx/ag_news')
imdb    = load_dataset('stanfordnlp/imdb')
news20  = load_dataset('SetFit/20_newsgroups')

# 회귀 트랙 (Yelp/Amazon은 sample_cap 적용)
yelp    = load_dataset('Yelp/yelp_review_full')
amazon  = load_dataset('SetFit/amazon_reviews_multi_en')  # ADR-017 §3-1 미러 (mteb는 dataset script 차단됨)
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
    md("## 0. import + 데이터 (01에서 캐싱)\n\n새 세션이면 drive 재마운트 필요. 같은 런타임에서 01 다음으로 실행이면 이 셀 skip 가능.\n"),
    code("""from google.colab import drive
drive.mount('/content/drive', force_remount=False)

import sys, os, glob, json
import numpy as np
import pandas as pd


def _find_dsc_base():
    root = '/content/drive/MyDrive'
    if not os.path.isdir(root):
        return None
    for c in [f'{root}/capstone/dsc', f'{root}/dsc', f'{root}/capstone-dsc']:
        if os.path.isfile(f'{c}/dsc_framework/__init__.py'):
            return c
    for pat in [f'{root}/*/dsc_framework/__init__.py',
                f'{root}/*/*/dsc_framework/__init__.py',
                f'{root}/*/*/*/dsc_framework/__init__.py']:
        for hit in glob.glob(pat):
            return os.path.dirname(os.path.dirname(hit))
    return None


BASE = _find_dsc_base()
if BASE is None:
    raise RuntimeError(
        'dsc_framework 폴더 못 찾음. 01 노트북 0-1 셀의 진단 메시지 + '
        'G드라이브 sync 상태 확인.'
    )
if BASE not in sys.path:
    sys.path.insert(0, BASE)
RESULTS_DIR = f'{BASE}/results'

from dsc_framework.text_cell import compute_dsc_text
from dsc_framework.text_cell_regression import compute_dsc_text_regression
from dsc_framework.text_polluters import (
    CompletenessTextPolluter, NoiseInjectionTextPolluter, WordShufflePolluter,
    ClassBalanceTextPolluter, LabelSwapTextPolluter,
    TargetDistributionSkewTextPolluter, TargetNoiseTextPolluter,
)
print(f'BASE: {BASE}')
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
    code("""from google.colab import drive
drive.mount('/content/drive', force_remount=False)

import sys, os, glob, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, r2_score


def _find_dsc_base():
    root = '/content/drive/MyDrive'
    if not os.path.isdir(root):
        return None
    for c in [f'{root}/capstone/dsc', f'{root}/dsc', f'{root}/capstone-dsc']:
        if os.path.isfile(f'{c}/dsc_framework/__init__.py'):
            return c
    for pat in [f'{root}/*/dsc_framework/__init__.py',
                f'{root}/*/*/dsc_framework/__init__.py',
                f'{root}/*/*/*/dsc_framework/__init__.py']:
        for hit in glob.glob(pat):
            return os.path.dirname(os.path.dirname(hit))
    return None


BASE = _find_dsc_base()
if BASE is None:
    raise RuntimeError('dsc_framework 폴더 못 찾음. 01 노트북 0-1 셀 진단 참조.')
if BASE not in sys.path:
    sys.path.insert(0, BASE)
RESULTS_DIR = f'{BASE}/results'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'BASE: {BASE}, device: {device}')
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
    code("""# ============================================================
# TextCNN — 분류/회귀 head 교체 (ADR-016/017 §3-2 사전등록)
# ============================================================
class _SimpleTokenizer:
    \"\"\"random init embedding용 whitespace 토큰화 + 단순 vocab.

    DistilBERT BPE와 다르지만 TextCNN baseline에서는 충분.
    UNK=1, PAD=0 고정. max_len truncation.
    \"\"\"
    PAD, UNK = 0, 1

    def __init__(self, max_vocab=20000):
        self.max_vocab = max_vocab
        self.itos = ['[PAD]', '[UNK]']
        self.stoi = {'[PAD]': 0, '[UNK]': 1}

    def fit(self, texts):
        from collections import Counter
        cnt = Counter()
        for t in texts:
            cnt.update(t.split())
        for tok, _ in cnt.most_common(self.max_vocab - 2):
            self.stoi[tok] = len(self.itos)
            self.itos.append(tok)
        return self

    def encode(self, text, max_len):
        ids = [self.stoi.get(tok, self.UNK) for tok in text.split()[:max_len]]
        if len(ids) < max_len:
            ids = ids + [self.PAD] * (max_len - len(ids))
        return ids

    def __len__(self):
        return len(self.itos)


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
        out = self.fc(x)
        return out.squeeze(-1) if self.regression else out


def train_textcnn(tr_t, tr_y, te_t, te_y, regression=False, epochs=10,
                  batch=64, lr=1e-3, max_len=256, emb=128, filters=100,
                  device=None):
    \"\"\"TextCNN finetune. 분류 → accuracy, 회귀 → R².\"\"\"
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    tok = _SimpleTokenizer().fit(tr_t)

    def to_tensor(texts, ys):
        X = torch.tensor([tok.encode(t, max_len) for t in texts], dtype=torch.long)
        if regression:
            y = torch.tensor(list(ys), dtype=torch.float32)
        else:
            y = torch.tensor(list(ys), dtype=torch.long)
        return X, y

    Xtr, ytr = to_tensor(tr_t, tr_y)
    Xte, yte = to_tensor(te_t, te_y)

    n_class = 1 if regression else int(max(int(max(tr_y)), int(max(te_y))) + 1)
    model = TextCNN(len(tok), n_class, emb=emb, filters=filters, regression=regression).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss() if regression else nn.CrossEntropyLoss()

    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch, shuffle=True)
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, len(Xte), batch):
            xb = Xte[i:i + batch].to(device)
            out = model(xb)
            if regression:
                preds.append(out.cpu().numpy())
            else:
                preds.append(out.argmax(dim=1).cpu().numpy())
        preds = np.concatenate(preds)

    if regression:
        return float(r2_score(yte.numpy(), preds))
    return float(accuracy_score(yte.numpy(), preds))
"""),
    code("""# ============================================================
# Transformer (DistilBERT/BERT/RoBERTa) — head 교체로 분류/회귀 둘 다 처리
# ADR-016/017 §3-2 사전등록 (max_len=256, batch=32, epoch=3, lr=2e-5, AdamW)
# ============================================================
def train_transformer(model_id, tr_t, tr_y, te_t, te_y, regression=False,
                      max_len=256, batch=32, epochs=3, lr=2e-5,
                      weight_decay=0.01, output_dir=None):
    \"\"\"HuggingFace AutoModelForSequenceClassification finetune.\"\"\"
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        Trainer, TrainingArguments, DataCollatorWithPadding,
    )
    from datasets import Dataset

    n_label = 1 if regression else int(max(int(max(tr_y)), int(max(te_y))) + 1)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=n_label,
        problem_type='regression' if regression else 'single_label_classification')

    def to_hf(texts, ys):
        return Dataset.from_dict({
            'text': list(texts),
            'labels': [float(y) for y in ys] if regression else [int(y) for y in ys],
        })

    def tokenize(batch):
        return tok(batch['text'], truncation=True, max_length=max_len)

    ds_tr = to_hf(tr_t, tr_y).map(tokenize, batched=True, remove_columns=['text'])
    ds_te = to_hf(te_t, te_y).map(tokenize, batched=True, remove_columns=['text'])

    args = TrainingArguments(
        output_dir=output_dir or f'./_tmp_{os.getpid()}',
        num_train_epochs=epochs,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        learning_rate=lr,
        weight_decay=weight_decay,
        logging_strategy='no',
        save_strategy='no',
        report_to='none',
        seed=42,
    )

    def metric_fn(eval_pred):
        preds, labels = eval_pred
        if regression:
            preds = preds.squeeze(-1) if preds.ndim > 1 else preds
            return {'r2': float(r2_score(labels, preds))}
        return {'accuracy': float(accuracy_score(labels, preds.argmax(axis=-1)))}

    trainer = Trainer(
        model=model, args=args,
        train_dataset=ds_tr, eval_dataset=ds_te,
        tokenizer=tok,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=metric_fn,
    )
    trainer.train()
    metrics = trainer.evaluate()
    return metrics.get('eval_r2' if regression else 'eval_accuracy', float('nan'))
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
    code("""from google.colab import drive
drive.mount('/content/drive', force_remount=False)

import sys, os, glob
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr


def _find_dsc_base():
    root = '/content/drive/MyDrive'
    if not os.path.isdir(root):
        return None
    for c in [f'{root}/capstone/dsc', f'{root}/dsc', f'{root}/capstone-dsc']:
        if os.path.isfile(f'{c}/dsc_framework/__init__.py'):
            return c
    for pat in [f'{root}/*/dsc_framework/__init__.py',
                f'{root}/*/*/dsc_framework/__init__.py',
                f'{root}/*/*/*/dsc_framework/__init__.py']:
        for hit in glob.glob(pat):
            return os.path.dirname(os.path.dirname(hit))
    return None


BASE = _find_dsc_base()
if BASE is None:
    raise RuntimeError('dsc_framework 폴더 못 찾음. 01 노트북 0-1 셀 진단 참조.')
if BASE not in sys.path:
    sys.path.insert(0, BASE)
RESULTS_DIR = f'{BASE}/results'

dsc_cls = pd.read_csv(f'{RESULTS_DIR}/text_cls_dsc_sweep.csv')
dsc_reg = pd.read_csv(f'{RESULTS_DIR}/text_reg_dsc_sweep.csv')
train_m = pd.read_csv(f'{RESULTS_DIR}/text_train_metrics.csv')

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
