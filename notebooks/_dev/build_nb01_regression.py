"""노트북 01 회귀 버전(01_setup_and_baseline_regression.ipynb) 생성 스크립트.

분류 노트북 01의 구조를 그대로 따르되 다음만 변경:
- 데이터셋: SouthGermanCredit/Telco/letter → CaliforniaHousing/BikeSharing/WineQuality
- DSC 엔진 셀: NEW_DSC_ENGINE 코드 → dsc_framework import
- 모델: 분류 5개 → 회귀 5개 (LinearReg/RFReg/XGBReg/SVR/MLPReg)
- 평가 메트릭: accuracy/F1/AUC → R² (음수 clip to 0)
- 분리된 결과 파일: dsc_scores_regression.csv, model_performance_regression.csv

사전등록 (ADR-011, v5 마스터플랜 sect 3-4, 3-5, 3-6):
- 데이터셋 후보 + target 컬럼
- 모델 5개 + 하이퍼파라미터
- 평가 R² (음수 clip)
"""
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NB_OUT = REPO / 'notebooks' / '01_setup_and_baseline_regression.ipynb'


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)}


def code(text):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {},
            'outputs': [], 'source': text.splitlines(keepends=True)}


CELLS = []

CELLS.append(md("""# 01. Setup & Baseline (Regression Cell)

**Phase 1**: 환경 세팅 → DSC framework import → 회귀 데이터 로드 → 베이스라인 DSC & 모델 R²

DSC v5 framework — regression cell instance (ADR-011, v5 마스터플랜).
정의식·가중치는 `dsc_framework/regression_cell.py`에 사전등록되어 freeze.

---"""))

CELLS.append(md("""## 0. 환경 설정"""))

CELLS.append(code("""# ============================================================
# 0-1. Google Drive 마운트 + 경로 설정
# ============================================================
from google.colab import drive
drive.mount('/content/drive')

import os
import sys

BASE = '/content/drive/MyDrive/capstone/dsc'
RAW_DIR = f'{BASE}/data/raw'
POLLUTED_DIR = f'{BASE}/data/polluted'
RESULTS_DIR = f'{BASE}/results'
DQ4AI_DIR = f'{BASE}/dq4ai'

# dsc_framework import 경로 추가
if BASE not in sys.path:
    sys.path.insert(0, BASE)

for d in [RAW_DIR, POLLUTED_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

print(f'BASE: {BASE}')
print('디렉토리 준비 완료')"""))

CELLS.append(code("""# ============================================================
# 0-2. 의존성 설치 + DQ4AI 패키지 클론 (polluter용)
# ============================================================
%pip install -q xgboost

if not os.path.exists(DQ4AI_DIR):
    !git clone https://github.com/HPI-Information-Systems/DQ4AI.git {DQ4AI_DIR}
    print('DQ4AI 클론 완료')
else:
    print('DQ4AI 이미 존재 — 스킵')

os.makedirs(f'{DQ4AI_DIR}/data/clean', exist_ok=True)"""))

CELLS.append(md("""## 1. 회귀 데이터셋 로드

3개 데이터셋 (v5 마스터플랜 sect 3-4):
- **California Housing** (sklearn): 20,640 × 9, target=MedHouseVal
- **Bike Sharing hour** (UCI 275, CC BY 4.0): 17,379 × 17, target=cnt — `casual`+`registered`=`cnt`이므로 leakage 제거 필수
- **Wine Quality** (UCI 186, CC BY 4.0): 6,497 × 12, target=quality (3~9 정수)"""))

CELLS.append(code("""# ============================================================
# 1-1. 데이터셋 메타 정의 (사전등록)
# ============================================================
import pandas as pd

DATASETS = {
    'CaliforniaHousing': {
        'path': f'{RAW_DIR}/california_housing.csv',
        'target': 'MedHouseVal',
        'numerical_cols': ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                           'Population', 'AveOccup', 'Latitude', 'Longitude'],
        'categorical_cols': [],
        'drop_cols': [],
        'placeholder_numerical': -1,
        'placeholder_categorical': None,
    },
    'BikeSharing': {
        'path': f'{RAW_DIR}/bike_sharing_hour.csv',
        'target': 'cnt',
        # casual + registered = cnt → leakage. instant(행 인덱스), dteday(날짜 문자열) 제거.
        'drop_cols': ['instant', 'dteday', 'casual', 'registered'],
        'numerical_cols': ['yr', 'mnth', 'hr', 'temp', 'atemp', 'hum', 'windspeed'],
        'categorical_cols': ['season', 'holiday', 'weekday', 'workingday', 'weathersit'],
        'placeholder_numerical': -1,
        'placeholder_categorical': 'empty',
    },
    'WineQuality': {
        'path': f'{RAW_DIR}/wine_quality.csv',
        'target': 'quality',
        'numerical_cols': ['fixed acidity', 'volatile acidity', 'citric acid',
                           'residual sugar', 'chlorides', 'free sulfur dioxide',
                           'total sulfur dioxide', 'density', 'pH',
                           'sulphates', 'alcohol'],
        'categorical_cols': [],
        'drop_cols': [],
        'placeholder_numerical': -1,
        'placeholder_categorical': None,
    },
}

print(f'사전등록 데이터셋: {list(DATASETS.keys())}')"""))

CELLS.append(code("""# ============================================================
# 1-2. 데이터 다운로드 안내 (이미 받아놓은 raw 파일이 없으면 _dev 스크립트 실행)
# ============================================================
import os

missing = [name for name, meta in DATASETS.items() if not os.path.exists(meta['path'])]
if missing:
    print(f'누락 데이터: {missing}')
    print('아래 셀을 수동 실행하여 다운로드:')
    print('  !python notebooks/_dev/download_regression_datasets.py')
else:
    print('모든 회귀 데이터셋 raw 파일 존재')
    for name, meta in DATASETS.items():
        size_mb = os.path.getsize(meta['path']) / 1024 / 1024
        print(f'  {name:<20s} {size_mb:.2f} MB')"""))

CELLS.append(code("""# ============================================================
# 1-3. DQ4AI data/clean/에 원본 복사 (polluter 호환용)
# ============================================================
import shutil

for name, meta in DATASETS.items():
    src = meta['path']
    dst = f'{DQ4AI_DIR}/data/clean/{name}.csv'
    shutil.copy(src, dst)
    print(f'  {name}.csv → DQ4AI/data/clean/')

print('\\n원본 데이터 준비 완료')"""))

CELLS.append(code("""# ============================================================
# 1-4. 데이터 요약 + leakage 컬럼 사전 제거 검증
# ============================================================
for name, meta in DATASETS.items():
    df = pd.read_csv(meta['path'])
    print(f'\\n=== {name} ===')
    print(f'  Shape (raw): {df.shape}')
    if meta['drop_cols']:
        df = df.drop(columns=[c for c in meta['drop_cols'] if c in df.columns])
        print(f'  Drop: {meta["drop_cols"]} → shape {df.shape}')
    target = meta['target']
    print(f'  Target: {target} (dtype={df[target].dtype}, range=[{df[target].min()}, {df[target].max()}], nunique={df[target].nunique()})')
    print(f'  수치형: {len(meta["numerical_cols"])}개, 범주형: {len(meta["categorical_cols"])}개')
    print(f'  Null 합계: {df.isnull().sum().sum()}')"""))

CELLS.append(md("""## 2. DSC v5 회귀 cell 엔진

`dsc_framework.regression_cell.compute_dsc_regression`을 사용한다 (v5 마스터플랜 사전등록).
9개 지표:
- 공통 6: completeness, uniqueness, validity, consistency, outlier_ratio, feature_correlation
- 회귀 전용 3: target_distribution_quality, target_smoothness, feature_informativeness_reg

분류 cell의 `class_balance` → `target_distribution_quality` (10-bin entropy)
분류 cell의 `label_consistency` → `target_smoothness` (k-NN target deviation)
분류 cell의 `feature_informativeness` → `feature_informativeness_reg` (mutual_info_regression)"""))

CELLS.append(code("""# ============================================================
# 2-1. DSC framework import + 검증
# ============================================================
from dsc_framework import (
    compute_dsc_regression,
    DEFAULT_WEIGHTS_REGRESSION,
    compute_dsc_degradation,
)

print('회귀 cell DSC 엔진 import 완료')
print(f'사전등록 가중치 (sum={sum(DEFAULT_WEIGHTS_REGRESSION.values()):.2f}):')
for k, v in DEFAULT_WEIGHTS_REGRESSION.items():
    print(f'  {k:<35s} {v:.2f}')"""))

CELLS.append(md("""## 3. 베이스라인 DSC 점수"""))

CELLS.append(code("""# ============================================================
# 3-1. 원본(clean) 데이터 DSC 점수 — 회귀 cell
# ============================================================
baseline_dsc_rows = []

for ds_name, meta in DATASETS.items():
    df = pd.read_csv(meta['path'])
    if meta['drop_cols']:
        df = df.drop(columns=[c for c in meta['drop_cols'] if c in df.columns])

    result = compute_dsc_regression(
        df,
        target_col=meta['target'],
        numerical_cols=meta['numerical_cols'],
        categorical_cols=meta['categorical_cols'],
        placeholder_numerical=meta.get('placeholder_numerical', -1),
        placeholder_categorical=meta.get('placeholder_categorical', 'empty'),
        reference_df=df,
    )
    row = {
        'dataset': ds_name,
        'polluter': 'none',
        'level': 0.0,
        **result,
    }
    baseline_dsc_rows.append(row)
    print(f"\\n{ds_name}: DSC={result['score']} ({result['grade']})")
    for k, v in result.items():
        if k not in ('score', 'grade'):
            print(f'  {k}: {v}')

df_baseline_dsc = pd.DataFrame(baseline_dsc_rows)
df_baseline_dsc"""))

CELLS.append(md("""## 4. 베이스라인 회귀 모델 학습 & 평가

3 데이터셋 × 5 모델 = 15회

회귀 모델 (사전등록):
- LinearRegression
- RandomForestRegressor (n_estimators=100)
- XGBRegressor (n_estimators=100)
- SVR (kernel='rbf')  ← 분류 SVC=linear와 달리 회귀 표준 RBF
- MLPRegressor (hidden_layer_sizes=(100,100,100,100,100))

평가: R² (음수는 0으로 clip — 모델이 평균보다 나쁠 때)"""))

CELLS.append(code("""# ============================================================
# 4-1. 모델 + 전처리 정의 (회귀)
# ============================================================
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')


def get_models():
    return {
        'LinearRegression': LinearRegression(n_jobs=-1),
        'RandomForestReg': RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        'XGBoostReg': XGBRegressor(
            n_estimators=100, random_state=42, n_jobs=-1,
            verbosity=0,
        ),
        'SVR': SVR(kernel='rbf'),
        'MLPReg': MLPRegressor(
            hidden_layer_sizes=(100, 100, 100, 100, 100),
            random_state=42, max_iter=1000
        ),
    }


def prepare_data(df, meta):
    \"\"\"전처리 파이프라인: train/test 분할 + 인코딩 + 스케일링 (회귀).\"\"\"
    target = meta['target']
    if meta['drop_cols']:
        df = df.drop(columns=[c for c in meta['drop_cols'] if c in df.columns])
    num_cols = [c for c in meta['numerical_cols'] if c in df.columns]
    cat_cols = [c for c in meta['categorical_cols'] if c in df.columns]
    feature_cols = num_cols + cat_cols

    X = df[feature_cols].copy()
    y = pd.to_numeric(df[target], errors='coerce').astype(float)

    # 수치형 컬럼 강제 변환
    for col in num_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    # 결측치 처리 (베이스라인용 간단 처리)
    for col in num_cols:
        X[col] = X[col].fillna(X[col].median())
    for col in cat_cols:
        X[col] = X[col].fillna('MISSING').astype(str)

    # train/test 분할 (회귀 — stratify 없음)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    transformers = []
    if num_cols:
        transformers.append(('num', StandardScaler(), num_cols))
    if cat_cols:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols))

    preprocessor = ColumnTransformer(transformers, remainder='drop')
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    return X_train_t, X_test_t, y_train.values, y_test.values


def evaluate_model(model, X_test, y_test):
    \"\"\"R² 반환 (음수 clip to 0). 회귀 cell 사전등록 기준 (마스터플랜 sect 3-6).\"\"\"
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    r2_clipped = max(0.0, r2)
    return {'r2': round(r2, 4), 'r2_clipped': round(r2_clipped, 4)}


print('모델 + 전처리 함수 정의 완료')"""))

CELLS.append(code("""# ============================================================
# 4-2. 베이스라인 모델 학습 실행 (3 × 5 = 15회)
# ============================================================
from time import time

baseline_perf_rows = []

for ds_name, meta in DATASETS.items():
    print(f'\\n=== {ds_name} ===')
    df = pd.read_csv(meta['path'])
    X_train, X_test, y_train, y_test = prepare_data(df, meta)

    for model_name, model in get_models().items():
        t0 = time()
        model.fit(X_train, y_train)
        scores = evaluate_model(model, X_test, y_test)
        elapsed = time() - t0

        row = {
            'dataset': ds_name,
            'polluter': 'none',
            'level': 0.0,
            'model': model_name,
            **scores,
        }
        baseline_perf_rows.append(row)
        print(f'  {model_name:20s} → R²={scores["r2"]:+.4f}  R²_clip={scores["r2_clipped"]:.4f}  ({elapsed:.1f}s)')

df_baseline_perf = pd.DataFrame(baseline_perf_rows)
print('\\n베이스라인 학습 완료')"""))

CELLS.append(md("""## 5. 결과 저장

회귀 cell 결과는 분류 cell과 분리된 파일에 저장:
- `dsc_scores_regression.csv`
- `model_performance_regression.csv`"""))

CELLS.append(code("""# ============================================================
# 5-1. 베이스라인 결과 저장 (회귀 cell — 별도 파일, baseline upsert 안전)
# ============================================================
dsc_scores_path = f'{RESULTS_DIR}/dsc_scores_regression.csv'
model_perf_path = f'{RESULTS_DIR}/model_performance_regression.csv'

def upsert_baseline(path, new_df):
    '''기존 csv 있으면 baseline (none, level=0.0) 행만 갱신, 폴루션 결과 보존.'''
    if os.path.isfile(path):
        try:
            existing = pd.read_csv(path)
            baseline_mask = (existing.polluter == 'none') & (existing.level == 0.0)
            kept = existing[~baseline_mask].copy()
            for col in new_df.columns:
                if col not in kept.columns:
                    kept[col] = pd.NA
            extra_cols = [c for c in kept.columns if c not in new_df.columns]
            kept = kept[list(new_df.columns) + extra_cols]
            combined = pd.concat([kept, new_df], ignore_index=True)
        except Exception:
            combined = new_df
    else:
        combined = new_df
    combined.to_csv(path, index=False)
    return len(combined)

n_dsc = upsert_baseline(dsc_scores_path, df_baseline_dsc)
n_perf = upsert_baseline(model_perf_path, df_baseline_perf)

print(f'DSC 점수 저장: {dsc_scores_path} (총 {n_dsc}건)')
print(f'모델 성능 저장: {model_perf_path} (총 {n_perf}건)')
print('--- 노트북 01 회귀 버전 완료 ---')
print('다음: 02_pollution_and_dsc_regression.ipynb 실행')"""))

CELLS.append(code("""# ============================================================
# 5-2. 실행 로그 저장 (회귀 cell)
# ============================================================
from datetime import datetime

log_lines = []
log_lines.append('# 노트북 01 회귀 버전 실행 로그')
log_lines.append('')
log_lines.append(f'- **실행 시각**: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}')
log_lines.append(f'- **BASE 경로**: {BASE}')
log_lines.append(f'- **DSC framework**: dsc_framework.regression_cell (v5 사전등록)')
log_lines.append('')

log_lines.append('## 1. 데이터셋')
log_lines.append('')
log_lines.append('| 데이터셋 | 행 | 열(after drop) | 타겟 | nunique | dtype |')
log_lines.append('|---|---|---|---|---|---|')
for ds_name, meta in DATASETS.items():
    df_tmp = pd.read_csv(meta['path'])
    if meta['drop_cols']:
        df_tmp = df_tmp.drop(columns=[c for c in meta['drop_cols'] if c in df_tmp.columns])
    log_lines.append(f'| {ds_name} | {df_tmp.shape[0]:,} | {df_tmp.shape[1]} | {meta[\"target\"]} | {df_tmp[meta[\"target\"]].nunique()} | {df_tmp[meta[\"target\"]].dtype} |')
log_lines.append('')

log_lines.append('## 2. 베이스라인 DSC 점수 (회귀 cell)')
log_lines.append('')
metric_keys = list(DEFAULT_WEIGHTS_REGRESSION.keys())
log_lines.append('| 데이터셋 | DSC Score | 등급 | ' + ' | '.join(metric_keys) + ' |')
log_lines.append('|---|---|---|' + '---|' * len(metric_keys))
for row in baseline_dsc_rows:
    cells = [row['dataset'], row['score'], row['grade']] + [row[k] for k in metric_keys]
    log_lines.append('| ' + ' | '.join(str(c) for c in cells) + ' |')
log_lines.append('')

log_lines.append('## 3. 베이스라인 모델 R²')
log_lines.append('')
log_lines.append('| 데이터셋 | 모델 | R² | R²_clipped |')
log_lines.append('|---|---|---|---|')
for row in baseline_perf_rows:
    log_lines.append(f'| {row[\"dataset\"]} | {row[\"model\"]} | {row[\"r2\"]} | {row[\"r2_clipped\"]} |')
log_lines.append('')

log_lines.append('## 4. 산출물')
log_lines.append('')
log_lines.append(f'- `{dsc_scores_path}` — 베이스라인 DSC 점수 {len(baseline_dsc_rows)}건')
log_lines.append(f'- `{model_perf_path}` — 베이스라인 모델 R² {len(baseline_perf_rows)}건')
log_lines.append('')
log_lines.append('---')
log_lines.append('*이 로그는 노트북 01 회귀 버전 실행 시 자동 생성됨*')

log_path = f'{RESULTS_DIR}/01_regression_execution_log.md'
with open(log_path, 'w', encoding='utf-8') as f:
    f.write('\\n'.join(log_lines))
print(f'실행 로그 저장: {log_path}')"""))


nb = {
    'cells': CELLS,
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3',
        },
        'language_info': {
            'name': 'python',
            'version': '3.11',
        },
    },
    'nbformat': 4,
    'nbformat_minor': 4,
}


os.makedirs(NB_OUT.parent, exist_ok=True)
with open(NB_OUT, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'생성: {NB_OUT}')
n_md = sum(1 for c in CELLS if c['cell_type'] == 'markdown')
n_code = sum(1 for c in CELLS if c['cell_type'] == 'code')
print(f'총 {len(CELLS)} 셀 (markdown {n_md}, code {n_code})')
