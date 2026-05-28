"""Generate 4 Phase-2 text cell notebook stubs (self-contained).

각 노트북이 독립 실행 가능:
- 01: 데이터 로드 + clean DSC baseline + train function sanity
- 02: polluter × level × dataset sweep → DSC csv 저장
- 03: model × polluter × level × dataset 학습 sweep → metric csv 저장
- 04: csv read → r·hold-out·모델별 r·default vs tuned 분석

공통 0-셀: drive.mount + dsc/ auto-discovery + 누락 파일 진단.
train function은 dsc_framework.text_trainers에서 import (검증 완료).

Usage:
    python _dev/gen_text_notebooks.py
"""
from __future__ import annotations

import json
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


# ===================================================================
# 공통 0번 셀: drive.mount + auto-discovery + 누락 파일 진단
# ===================================================================

SETUP_CELL = """# ============================================================
# 0. Drive 마운트 + dsc/ 자동 검색 + sys.path 등록
# ============================================================
from google.colab import drive
drive.mount('/content/drive', force_remount=False)

import os, sys, glob, json
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
    drive_root = '/content/drive/MyDrive'
    listing = os.listdir(drive_root) if os.path.isdir(drive_root) else []
    raise RuntimeError(
        'dsc_framework/ 폴더를 G드라이브에서 못 찾음.\\n'
        '  1) G드라이브 클라이언트 sync 완료 확인 (commit 직후면 잠시 대기 후 재시도)\\n'
        '  2) Drive 마운트 확인 — !ls /content/drive/MyDrive\\n'
        f'  현재 Drive 내용: {listing[:20]}'
    )

# 누락 파일 진단 — partial sync 시 빠른 실패
REQUIRED = ['shared_metrics.py', 'classification_cell.py', 'regression_cell.py',
            'image_cell.py', 'text_cell.py', 'text_cell_regression.py',
            'text_trainers.py', 'data_type_detection.py', 'router.py',
            'text_polluters', 'image_polluters']
missing = [f for f in REQUIRED if not os.path.exists(f'{BASE}/dsc_framework/{f}')]
if missing:
    raise RuntimeError(
        f'dsc_framework/ 파일 누락: {missing}\\n'
        '→ G드라이브 sync 미완료. 잠시 대기 후 재실행.\\n'
        '→ Colab Drive view stale 시: drive.flush_and_unmount() 후 재마운트.'
    )

RESULTS_DIR = f'{BASE}/results'
DATA_DIR = f'{BASE}/data/text'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

if BASE not in sys.path:
    sys.path.insert(0, BASE)

print(f'BASE: {BASE}')
print(f'dsc_framework 파일: {sorted(f for f in os.listdir(f"{BASE}/dsc_framework") if not f.startswith("_"))}')
"""

# pip install — Colab 환경
INSTALL_CELL = """# ============================================================
# 의존성 설치 (Colab 1회 실행 후 다음 셀)
# ============================================================
%pip install -q 'transformers>=4.30' 'datasets>=2.10' 'xgboost>=1.7' 'accelerate>=1.1.0'
"""

# 데이터 로드 helper (공통)
DATA_LOAD_CELL = """# ============================================================
# 데이터셋 로드 (ADR-016 분류 3종 + ADR-017 회귀 3종)
#   - Phase 2 정식 실행 시 N_TRAIN/N_TEST를 ADR §4 sample_cap으로 키울 것
#   - sanity/dev에선 작은 sample로 시작
# ============================================================
from datasets import load_dataset
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {device}')

# 사용자가 sample size 조정. ADR §4 정식 = train 50000~200000 / test 5000~50000
N_TRAIN = 3000  # 분류·SST5는 자동 cap, 빈 dataset이면 자체 train size
N_TEST  = 500


def _slice(ds_dict, split, n, seed=42):
    ds = ds_dict[split] if split in ds_dict else ds_dict['train']
    if n is None or len(ds) <= n:
        return ds
    return ds.shuffle(seed=seed).select(range(n))


def load_all():
    \"\"\"분류 3 + 회귀 3 = 6 dataset을 (tr_texts, tr_y, te_texts, te_y, task)로 반환.\"\"\"
    out = {}

    # 분류
    ag = load_dataset('fancyzhx/ag_news')
    out['ag_news'] = (_slice(ag, 'train', N_TRAIN), _slice(ag, 'test', N_TEST), 'classification')

    imdb = load_dataset('stanfordnlp/imdb')
    out['imdb'] = (_slice(imdb, 'train', N_TRAIN), _slice(imdb, 'test', N_TEST), 'classification')

    news20 = load_dataset('SetFit/20_newsgroups')
    out['20news'] = (_slice(news20, 'train', N_TRAIN), _slice(news20, 'test', N_TEST), 'classification')

    # 회귀 (label = star/sentiment를 float)
    yelp = load_dataset('Yelp/yelp_review_full')
    out['yelp_full'] = (_slice(yelp, 'train', N_TRAIN), _slice(yelp, 'test', N_TEST), 'regression')

    amazon = load_dataset('SetFit/amazon_reviews_multi_en')  # ADR-017 미러
    out['amazon_en'] = (_slice(amazon, 'train', N_TRAIN), _slice(amazon, 'test', N_TEST), 'regression')

    sst = load_dataset('SetFit/sst5')
    out['sst5'] = (_slice(sst, 'train', N_TRAIN), _slice(sst, 'test', N_TEST), 'regression')

    return out


def to_lists(ds_split, task):
    texts = ds_split['text']
    labels = ds_split['label']
    if task == 'regression':
        labels = [float(y) for y in labels]
    return list(texts), list(labels)


print('load_dataset OK — load_all() 호출하면 6 dataset 로드 시작.')
"""


# ===================================================================
# NB_01 — Setup & Baseline
# ===================================================================

NB_01 = notebook([
    md("""# 01. Setup & Baseline (Text Cell)

ADR-016 (분류) + ADR-017 (회귀) Phase 2 진입.

- 6 dataset 로드 + clean DSC baseline 측정 (default 가중치, ADR-015 fallback)
- 결과: `results/text_baseline_dsc.csv`

Phase 2 후속 노트북:
- 02 polluter sweep, 03 model train, 04 분석
"""),
    code(SETUP_CELL),
    code(INSTALL_CELL),
    code("""# ============================================================
# Restart 권장 — pip install 후 import 안정성 위해 런타임 한 번 재시작
# 또는 force reimport.
# ============================================================
from dsc_framework.text_cell import compute_dsc_text
from dsc_framework.text_cell_regression import compute_dsc_text_regression
print('text cell modules OK')
"""),
    code(DATA_LOAD_CELL),
    code("""# ============================================================
# 6 dataset 로드 + clean DSC baseline 측정
# ============================================================
datasets = load_all()

rows = []
for name, (tr_ds, te_ds, task) in datasets.items():
    texts, labels = to_lists(tr_ds, task)
    print(f'\\n=== {name} ({task}, n={len(texts)}) ===')
    if task == 'classification':
        r = compute_dsc_text(texts, labels, use_embeddings=True,
                             sample_cap=1000, random_state=42)
    else:
        r = compute_dsc_text_regression(texts, labels, use_embeddings=True,
                                        sample_cap=1000, random_state=42)
    print(f'  DSC={r["score"]} ({r["grade"]})')
    row = {'dataset': name, 'task': task, 'n': len(texts), **{k: v for k, v in r.items()}}
    rows.append(row)

baseline = pd.DataFrame(rows)
out_path = f'{RESULTS_DIR}/text_baseline_dsc.csv'
baseline.to_csv(out_path, index=False)
print(f'\\nbaseline saved: {out_path}')
baseline
"""),
    md("""---

다음: `02_pollution_and_dsc_text.ipynb` — polluter 스윕.
"""),
])


# ===================================================================
# NB_02 — Pollution × DSC
# ===================================================================

NB_02 = notebook([
    md("""# 02. Pollution × DSC (Text Cell)

ADR-016/017 §3-3 polluter × level × dataset 스윕 → DSC csv.

Output: `results/text_dsc_sweep.csv` (분류 + 회귀 합본)
"""),
    code(SETUP_CELL),
    code(INSTALL_CELL),
    code("""# ============================================================
# imports
# ============================================================
from dsc_framework.text_cell import compute_dsc_text
from dsc_framework.text_cell_regression import compute_dsc_text_regression
from dsc_framework.text_polluters import (
    CompletenessTextPolluter, NoiseInjectionTextPolluter, WordShufflePolluter,
    ClassBalanceTextPolluter, LabelSwapTextPolluter,
    TargetDistributionSkewTextPolluter, TargetNoiseTextPolluter,
)
print('polluters OK')
"""),
    code(DATA_LOAD_CELL),
    code("""# ============================================================
# Sweep 설정 (ADR-016/017 §3-3 사전등록)
# ============================================================
LEVEL_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9]
SEEDS = [42, 7, 99]

POLLUTERS_CLS = {
    'completeness_text':    CompletenessTextPolluter,
    'noise_injection_text': NoiseInjectionTextPolluter,
    'word_shuffle':         WordShufflePolluter,
    'class_balance':        ClassBalanceTextPolluter,
    'label_swap':           LabelSwapTextPolluter,
}
POLLUTERS_REG = {
    'completeness_text':         CompletenessTextPolluter,
    'noise_injection_text':      NoiseInjectionTextPolluter,
    'word_shuffle':              WordShufflePolluter,
    'target_distribution_skew':  TargetDistributionSkewTextPolluter,
    'target_noise':              TargetNoiseTextPolluter,
}


def sweep_one(name, tr_ds, task):
    texts, labels = to_lists(tr_ds, task)
    polluters = POLLUTERS_CLS if task == 'classification' else POLLUTERS_REG
    compute_fn = compute_dsc_text if task == 'classification' else compute_dsc_text_regression
    rows = []
    for pol_name, pol_cls in polluters.items():
        for seed in SEEDS:
            for lvl in LEVEL_GRID:
                pol = pol_cls(lvl, random_seed=seed)
                t_p, l_p = pol.pollute(texts, labels)
                r = compute_fn(t_p, l_p, use_embeddings=True,
                               sample_cap=500, random_state=seed)
                rows.append({
                    'dataset': name, 'task': task, 'polluter': pol_name,
                    'level': lvl, 'seed': seed,
                    'dsc_score': r['score'],
                    **{k: v for k, v in r.items() if k not in ('score', 'grade')},
                })
    return rows
"""),
    code("""# ============================================================
# 전체 스윕 — Colab T4에서 dataset 당 ~10분. 6 dataset ≈ 1시간
# ============================================================
datasets = load_all()
all_rows = []
for name, (tr_ds, te_ds, task) in datasets.items():
    print(f'\\n=== {name} ({task}) sweep 시작 ===')
    rows = sweep_one(name, tr_ds, task)
    all_rows.extend(rows)
    print(f'  rows: {len(rows)}')

sweep = pd.DataFrame(all_rows)
out_path = f'{RESULTS_DIR}/text_dsc_sweep.csv'
sweep.to_csv(out_path, index=False)
print(f'\\nsweep saved: {out_path} ({len(sweep)} rows)')
sweep.head()
"""),
    md("---\n\n다음: `03_training_text.ipynb` — 모델 학습 스윕.\n"),
])


# ===================================================================
# NB_03 — Model Training
# ===================================================================

NB_03 = notebook([
    md("""# 03. Training (Text Cell)

ADR-016/017 §3-2 model × polluter × level × dataset 학습 → metric csv.

분류 5종: LogReg+TFIDF / TextCNN / DistilBERT / BERT / RoBERTa
회귀 5종: Ridge+TFIDF / XGBoost+TFIDF / TextCNN-Reg / DistilBERT-Reg / BERT-Reg

학습 함수는 `dsc_framework.text_trainers`에서 import (검증 완료).

GPU 시간: 분류 30~50시간 + 회귀 30~50시간 = 60~100시간 (T4 가정).
Colab Pro+ 또는 Pay-as-you-go.

Output: `results/text_train_metrics.csv`
"""),
    code(SETUP_CELL),
    code(INSTALL_CELL),
    code("""# ============================================================
# imports
# ============================================================
from dsc_framework.text_trainers import (
    CLASSIFICATION_MODELS, REGRESSION_MODELS,
)
from dsc_framework.text_polluters import (
    CompletenessTextPolluter, NoiseInjectionTextPolluter, WordShufflePolluter,
    ClassBalanceTextPolluter, LabelSwapTextPolluter,
    TargetDistributionSkewTextPolluter, TargetNoiseTextPolluter,
)
print('trainers OK:', list(CLASSIFICATION_MODELS.keys()), list(REGRESSION_MODELS.keys()))
"""),
    code(DATA_LOAD_CELL),
    code("""# ============================================================
# 학습 sweep 설정
# ============================================================
LEVEL_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9]
SEED = 42  # 학습은 1 seed (NB_02 sweep과 메모리 절약)

POLLUTERS_CLS = {
    'completeness_text':    CompletenessTextPolluter,
    'noise_injection_text': NoiseInjectionTextPolluter,
    'word_shuffle':         WordShufflePolluter,
    'class_balance':        ClassBalanceTextPolluter,
    'label_swap':           LabelSwapTextPolluter,
}
POLLUTERS_REG = {
    'completeness_text':         CompletenessTextPolluter,
    'noise_injection_text':      NoiseInjectionTextPolluter,
    'word_shuffle':              WordShufflePolluter,
    'target_distribution_skew':  TargetDistributionSkewTextPolluter,
    'target_noise':              TargetNoiseTextPolluter,
}


def train_one_combo(model_name, train_fn, tr_t, tr_y, te_t, te_y, task, **kw):
    try:
        metric = train_fn(tr_t, tr_y, te_t, te_y, **kw)
        return {'metric': float(metric), 'error': None}
    except Exception as e:
        return {'metric': float('nan'), 'error': f'{type(e).__name__}: {e}'}


def model_sweep(name, tr_ds, te_ds, task):
    tr_texts, tr_y = to_lists(tr_ds, task)
    te_texts, te_y = to_lists(te_ds, task)
    polluters = POLLUTERS_CLS if task == 'classification' else POLLUTERS_REG
    models = CLASSIFICATION_MODELS if task == 'classification' else REGRESSION_MODELS
    rows = []
    for pol_name, pol_cls in polluters.items():
        for lvl in LEVEL_GRID:
            pol = pol_cls(lvl, random_seed=SEED)
            tr_p, tr_yp = pol.pollute(tr_texts, tr_y)
            for model_name, train_fn in models.items():
                t0 = __import__('time').time()
                res = train_one_combo(model_name, train_fn, tr_p, tr_yp,
                                       te_texts, te_y, task)
                rows.append({
                    'dataset': name, 'task': task, 'model': model_name,
                    'polluter': pol_name, 'level': lvl, 'seed': SEED,
                    'metric': res['metric'], 'error': res['error'],
                    'elapsed_s': round(__import__('time').time() - t0, 2),
                })
                print(f'  {model_name:14s} pol={pol_name:24s} lvl={lvl:.2f} metric={res["metric"]:.3f}')
    return rows
"""),
    code("""# ============================================================
# 전체 학습 sweep — 매우 long-running. 6 dataset × 5 model × 5 pol × 6 lvl
# = 900 학습. GPU 큐 백그라운드 권장. 체크포인트 csv 저장.
# ============================================================
import time
datasets = load_all()

ckpt = f'{RESULTS_DIR}/text_train_metrics.csv'
all_rows = []

for name, (tr_ds, te_ds, task) in datasets.items():
    print(f'\\n=== {name} ({task}) 학습 시작 ===')
    t0 = time.time()
    rows = model_sweep(name, tr_ds, te_ds, task)
    all_rows.extend(rows)
    pd.DataFrame(all_rows).to_csv(ckpt, index=False)  # 체크포인트
    print(f'  {len(rows)} rows, {(time.time()-t0)/60:.1f}분')

print(f'\\nfinal: {ckpt} ({len(all_rows)} rows)')
pd.DataFrame(all_rows).head()
"""),
    md("---\n\n다음: `04_scoreboard_text.ipynb` — r·hold-out·default vs tuned.\n"),
])


# ===================================================================
# NB_04 — Scoreboard
# ===================================================================

NB_04 = notebook([
    md("""# 04. Scoreboard (Text Cell)

ADR-016/017 §6 검증. 02의 DSC + 03의 train metric을 join → r 분석.

합격 기준:
- Pearson r(DSC, accuracy/R²) ≥ 0.4
- Spearman ρ ≥ 0.4
- Polluter hold-out 4/5 PASS
- 모델 5/5 양의 r

회귀 트랙은 ADR-012 Degradation Index 보조 보고.
"""),
    code(SETUP_CELL),
    code("""# ============================================================
# results csv 로드
# ============================================================
from scipy.stats import pearsonr, spearmanr

dsc_sweep = pd.read_csv(f'{RESULTS_DIR}/text_dsc_sweep.csv')
train_m = pd.read_csv(f'{RESULTS_DIR}/text_train_metrics.csv')

print('dsc_sweep:', dsc_sweep.shape, dsc_sweep['task'].value_counts().to_dict())
print('train_m  :', train_m.shape, train_m['task'].value_counts().to_dict())

JOIN_KEYS = ['dataset', 'polluter', 'level', 'seed']
# DSC는 multi-seed, train은 single-seed → train의 seed로 join
train_seed = train_m['seed'].iloc[0]
dsc_for_join = dsc_sweep[dsc_sweep['seed'] == train_seed]
print(f'join seed={train_seed}, dsc rows for join={len(dsc_for_join)}')
"""),
    md("## 1. dataset별 r(DSC, metric) — 합격선 검토\n"),
    code("""def r_per_dataset(dsc_df, train_df):
    j = dsc_df.merge(train_df, on=JOIN_KEYS)
    rows = []
    for ds_name in j['dataset'].unique():
        sub = j[j['dataset'] == ds_name]
        if len(sub) < 4 or sub['metric'].isna().all():
            rows.append({'dataset': ds_name, 'n': len(sub), 'note': 'insufficient'})
            continue
        sub_clean = sub.dropna(subset=['metric'])
        r_p, p_p = pearsonr(sub_clean['dsc_score'], sub_clean['metric'])
        r_s, p_s = spearmanr(sub_clean['dsc_score'], sub_clean['metric'])
        rows.append({
            'dataset': ds_name, 'n': len(sub_clean),
            'pearson': round(r_p, 4), 'p_pearson': p_p,
            'spearman': round(r_s, 4), 'p_spearman': p_s,
            'pass_r040': bool(r_p >= 0.4 and r_s >= 0.4),
        })
    return pd.DataFrame(rows)

print('=== 분류 트랙 ===')
print(r_per_dataset(dsc_for_join[dsc_for_join['task'] == 'classification'],
                   train_m[train_m['task'] == 'classification']))
print('\\n=== 회귀 트랙 ===')
print(r_per_dataset(dsc_for_join[dsc_for_join['task'] == 'regression'],
                   train_m[train_m['task'] == 'regression']))
"""),
    md("## 2. Polluter hold-out (4/5 PASS)\n"),
    code("""def polluter_holdout(dsc_df, train_df):
    j = dsc_df.merge(train_df, on=JOIN_KEYS).dropna(subset=['metric'])
    polluters = j['polluter'].unique()
    rows = []
    for held in polluters:
        for ds_name in j['dataset'].unique():
            sub = j[(j['polluter'] != held) & (j['dataset'] == ds_name)]
            if len(sub) < 4:
                continue
            r, _ = pearsonr(sub['dsc_score'], sub['metric'])
            rows.append({'held_out': held, 'dataset': ds_name,
                         'pearson': round(r, 4), 'pass': r >= 0.4})
    return pd.DataFrame(rows)

ho_cls = polluter_holdout(dsc_for_join[dsc_for_join['task'] == 'classification'],
                          train_m[train_m['task'] == 'classification'])
ho_reg = polluter_holdout(dsc_for_join[dsc_for_join['task'] == 'regression'],
                          train_m[train_m['task'] == 'regression'])
print('=== 분류 hold-out ===')
print(ho_cls.groupby(['dataset']).agg(pass_count=('pass', 'sum'), total=('pass', 'count')))
print('\\n=== 회귀 hold-out ===')
print(ho_reg.groupby(['dataset']).agg(pass_count=('pass', 'sum'), total=('pass', 'count')))
"""),
    md("## 3. 모델별 r (5/5 양의 r)\n"),
    code("""def r_per_model(dsc_df, train_df):
    j = dsc_df.merge(train_df, on=JOIN_KEYS).dropna(subset=['metric'])
    rows = []
    for m in j['model'].unique():
        sub = j[j['model'] == m]
        if len(sub) < 4:
            continue
        r, p = pearsonr(sub['dsc_score'], sub['metric'])
        rows.append({'model': m, 'pearson': round(r, 4), 'p': p, 'positive': r > 0})
    return pd.DataFrame(rows)

print('=== 분류 모델별 r ===')
print(r_per_model(dsc_for_join[dsc_for_join['task'] == 'classification'],
                  train_m[train_m['task'] == 'classification']))
print('\\n=== 회귀 모델별 r ===')
print(r_per_model(dsc_for_join[dsc_for_join['task'] == 'regression'],
                  train_m[train_m['task'] == 'regression']))
"""),
    md("""## 4. Default vs tuned 가중치 grid search

이미지 cell `04` 패턴 미러. dead/live 메트릭 진단 + grid로 r 상승 가중치 탐색.

ADR-015 원칙: 본 grid 결과는 fallback 가중치 갭 분석. 운영·검증 정식 가중치는
Phase 4의 LLM weight generator 출력.
"""),
    code("""# dead 메트릭 진단 — std < 0.01 (변동 없음 → r에 영향 없음)
METRIC_KEYS_CLS = ['completeness_text', 'uniqueness', 'validity', 'consistency',
                   'outlier_ratio', 'class_balance', 'feature_correlation',
                   'label_consistency', 'feature_informativeness', 'sample_quality_text']
METRIC_KEYS_REG = ['completeness_text', 'uniqueness', 'validity', 'consistency',
                   'outlier_ratio', 'target_distribution_quality',
                   'feature_correlation', 'target_smoothness',
                   'feature_informativeness_reg', 'sample_quality_text']

for task, keys in [('classification', METRIC_KEYS_CLS), ('regression', METRIC_KEYS_REG)]:
    sub = dsc_sweep[dsc_sweep['task'] == task]
    print(f'\\n=== {task} ===')
    for k in keys:
        if k in sub.columns:
            print(f'  {k:30s} std={sub[k].std():.4f}  range=[{sub[k].min():.3f}, {sub[k].max():.3f}]')
"""),
    code("""# 가중치 grid search — image cell의 score_image_v2.py 패턴 이식 예정.
# 본 셀은 stub: scipy.optimize.minimize로 -r 최소화 → tuned 가중치.
# from scipy.optimize import minimize
# (구현은 결과 분포 보고 결정)
print('grid search: results/text_dsc_sweep.csv 분석 후 구현')
"""),
    md("---\n\n**Phase 4 진입 전제**: 위 1~3 합격 시 plan 20260528-02 LLM prompt freeze + held-out 측정.\n"),
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
