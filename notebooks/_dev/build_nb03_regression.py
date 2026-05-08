"""노트북 03 회귀 버전(03_training_regression.ipynb) 생성.

분류 노트북 03와의 변경:
- 모델: classifier → regressor
- preprocess: target LabelEncoder 제거 (회귀)
- evaluate_model: accuracy/f1/auc → R² + R²_clipped
- 디렉토리: train_polluted_regression / test_clean_regression / split_meta_regression
- 결과 파일: model_performance_regression.csv

체크포인트(이미 학습된 결과 skip) 메커니즘은 분류 cell과 동일하게 유지.
SVR(rbf)이 큰 데이터셋(16K~17K rows)에서 느릴 수 있으니 체크포인트 중요.
"""
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NB_OUT = REPO / 'notebooks' / '03_training_regression.ipynb'


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)}


def code(text):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {},
            'outputs': [], 'source': text.splitlines(keepends=True)}


CELLS = []

CELLS.append(md("""# 03. Training & Evaluation (Regression Cell)

**Phase 2**: 모든 오염 데이터셋 × 회귀 모델 5개 학습 → R² 측정

split-first 원칙 + leakage 검증 (1차: split 인덱스 disjoint, 2차: row hash)

---"""))

CELLS.append(md("""## 0. 환경 설정"""))

CELLS.append(code("""# ============================================================
# 0-1. Drive 마운트 + 경로
# ============================================================
from google.colab import drive
drive.mount('/content/drive')

import os, sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from time import time

BASE = '/content/drive/MyDrive/capstone/dsc'
RAW_DIR = f'{BASE}/data/raw'
RESULTS_DIR = f'{BASE}/results'
TRAIN_DIR = f'{BASE}/data/train_polluted_regression'
TEST_DIR = f'{BASE}/data/test_clean_regression'
SPLIT_META_DIR = f'{BASE}/data/split_meta_regression'

if BASE not in sys.path:
    sys.path.insert(0, BASE)

print('환경 설정 완료')"""))

CELLS.append(code("""# ============================================================
# 0-2. 데이터셋 메타 + 회귀 모델 정의
# ============================================================
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score


DATASETS = {
    'CaliforniaHousing': {
        'target': 'MedHouseVal',
        'numerical_cols': ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                           'Population', 'AveOccup', 'Latitude', 'Longitude'],
        'categorical_cols': [],
        'drop_cols': [],
    },
    'BikeSharing': {
        'target': 'cnt',
        'drop_cols': ['instant', 'dteday', 'casual', 'registered'],
        'numerical_cols': ['yr', 'mnth', 'hr', 'temp', 'atemp', 'hum', 'windspeed'],
        'categorical_cols': ['season', 'holiday', 'weekday', 'workingday', 'weathersit'],
    },
    'WineQuality': {
        'target': 'quality',
        'numerical_cols': ['fixed acidity', 'volatile acidity', 'citric acid',
                           'residual sugar', 'chlorides', 'free sulfur dioxide',
                           'total sulfur dioxide', 'density', 'pH',
                           'sulphates', 'alcohol'],
        'categorical_cols': [],
        'drop_cols': [],
    },
}


def get_models():
    return {
        'LinearRegression': LinearRegression(n_jobs=-1),
        'RandomForestReg': RandomForestRegressor(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        'XGBoostReg': XGBRegressor(
            n_estimators=100, random_state=42, n_jobs=-1, verbosity=0,
        ),
        'SVR': SVR(kernel='rbf'),
        'MLPReg': MLPRegressor(
            hidden_layer_sizes=(100, 100, 100, 100, 100),
            random_state=42, max_iter=1000
        ),
    }


def preprocess(df_train, df_test, meta):
    \"\"\"이미 분리된 train/test에 전처리 적용 (회귀 — target 인코딩 없음).\"\"\"
    target = meta['target']
    drop_cols = meta.get('drop_cols', [])

    # drop_cols가 train에 있으면 미리 제거 (test는 02 노트북에서 이미 정리)
    if drop_cols:
        df_train = df_train.drop(columns=[c for c in drop_cols if c in df_train.columns])
        df_test = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns])

    num_cols = [c for c in meta['numerical_cols'] if c in df_train.columns]
    cat_cols = [c for c in meta['categorical_cols'] if c in df_train.columns]
    feature_cols = num_cols + cat_cols

    X_train = df_train[feature_cols].copy()
    X_test = df_test[feature_cols].copy()

    y_train = pd.to_numeric(df_train[target], errors='coerce').astype(float)
    y_test = pd.to_numeric(df_test[target], errors='coerce').astype(float)

    for col in num_cols:
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

    for col in num_cols:
        med = X_train[col].median() if X_train[col].notna().any() else 0
        X_train[col] = X_train[col].fillna(med)
        X_test[col] = X_test[col].fillna(med)
    for col in cat_cols:
        X_train[col] = X_train[col].fillna('MISSING').astype(str)
        X_test[col] = X_test[col].fillna('MISSING').astype(str)

    transformers = []
    if num_cols:
        transformers.append(('num', StandardScaler(), num_cols))
    if cat_cols:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols))

    preprocessor = ColumnTransformer(transformers, remainder='drop')
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # target dropna 정렬 (혹시 결측이 들어왔을 경우)
    train_mask = ~np.isnan(y_train.values)
    test_mask = ~np.isnan(y_test.values)
    return (X_train_t[train_mask], X_test_t[test_mask],
            y_train.values[train_mask], y_test.values[test_mask])


def evaluate_model(model, X_test, y_test):
    \"\"\"R² 반환 (음수 clip to 0). 사전등록 (마스터플랜 sect 3-6).\"\"\"
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    r2_clipped = max(0.0, r2)
    return {'r2': round(r2, 4), 'r2_clipped': round(r2_clipped, 4)}


print('회귀 모델 + 유틸리티 정의 완료')"""))

CELLS.append(md("""## 1. 오염 데이터 목록 스캔"""))

CELLS.append(code("""# ============================================================
# 1-1. 실험 목록 구성 (train_polluted_regression 디렉토리 스캔)
# ============================================================
experiments = []

for ds_name in DATASETS:
    ds_dir = f'{TRAIN_DIR}/{ds_name}'
    if not os.path.isdir(ds_dir):
        continue
    for folder in sorted(os.listdir(ds_dir)):
        train_csv = f'{ds_dir}/{folder}/train_data.csv'
        if not os.path.isfile(train_csv):
            continue
        parts = folder.rsplit('_', 1)
        polluter_name = parts[0]
        level = int(parts[1]) / 100 if len(parts) == 2 else 0.0
        experiments.append({
            'dataset': ds_name,
            'polluter': polluter_name,
            'level': level,
            'train_path': train_csv,
        })

print(f'실험 목록: {len(experiments)}건')
print(f'모델 5개 x {len(experiments)}건 = 총 {5 * len(experiments)}회 학습 예정')"""))

CELLS.append(md("""## 1-2. Leakage 자동 검증 (안전장치)

학습 시작 전에 모든 train 데이터에 test 행이 섞이지 않았는지 검증.

1. **1차**: 노트북 02가 저장한 split 인덱스 disjoint 검증
2. **2차**: row hash 비교 (자연 중복은 baseline 화이트리스트로 차감)

피드백 메모리 (feedback_ml_pipeline_process): split-first 원칙, 사후 필터링 금지, 안전장치 필수."""))

CELLS.append(code("""# ============================================================
# 1-2. Leakage 검증 (학습 전 안전장치)
# ============================================================
import hashlib

def _row_hash(row):
    return hashlib.md5('|'.join(map(str, row.values)).encode()).hexdigest()

print('=== Leakage 검증 시작 ===')
leakage_found = False

for ds_name in DATASETS:
    test_path = f'{TEST_DIR}/{ds_name}_test.csv'
    if not os.path.isfile(test_path):
        print(f'  {ds_name}: test 파일 없음 -> SKIP')
        continue

    # --- 1차: split 인덱스 disjoint ---
    train_idx_p = f'{SPLIT_META_DIR}/{ds_name}_train_idx.npy'
    test_idx_p = f'{SPLIT_META_DIR}/{ds_name}_test_idx.npy'
    if os.path.isfile(train_idx_p) and os.path.isfile(test_idx_p):
        train_idx = set(np.load(train_idx_p).tolist())
        test_idx = set(np.load(test_idx_p).tolist())
        idx_overlap = train_idx & test_idx
        if idx_overlap:
            print(f'  {ds_name}: ❌ split 인덱스 겹침 {len(idx_overlap)}건 — 02 노트북 split 검증 실패!')
            leakage_found = True
            continue
        print(f'  {ds_name}: split disjoint ✓ (train {len(train_idx)}, test {len(test_idx)})')
    else:
        print(f'  {ds_name}: split_meta 없음 → 2차 검증으로만 확인')

    # --- 2차: row hash 기반 ---
    df_test = pd.read_csv(test_path)
    test_hashes = set(df_test.apply(_row_hash, axis=1))

    baseline_path = f'{TRAIN_DIR}/{ds_name}/none_0/train_data.csv'
    df_baseline = pd.read_csv(baseline_path)
    baseline_hashes = set(df_baseline.apply(_row_hash, axis=1))
    natural_overlap = baseline_hashes & test_hashes
    if natural_overlap:
        print(f'  {ds_name}: 자연 중복 {len(natural_overlap)}건 (baseline에 이미 존재 — 회귀 데이터셋의 샘플 중복)')

    ds_experiments = [e for e in experiments if e['dataset'] == ds_name]
    new_leak_count = 0
    for exp in ds_experiments:
        df_train = pd.read_csv(exp['train_path'])
        train_hashes = set(df_train.apply(_row_hash, axis=1))
        new_overlap = (train_hashes & test_hashes) - natural_overlap
        if new_overlap:
            label = f\"{ds_name}/{exp['polluter']}_{int(exp['level']*100)}%\"
            print(f'    LEAKAGE: {label} → 새 겹침 {len(new_overlap)}건')
            leakage_found = True
            new_leak_count += 1

    print(f'  {ds_name}: {len(ds_experiments)}건 검증 완료 (신규 leakage {new_leak_count}건)')

if leakage_found:
    raise RuntimeError('LEAKAGE 발견! 02 노트북의 split·폴루션 단계를 확인하세요.')
print()
print('모든 실험에서 leakage 없음 확인. 학습 진행 가능.')"""))

CELLS.append(md("""## 2. 전체 회귀 모델 학습 & 평가

체크포인트 지원: `model_performance_regression.csv`에 이미 저장된 (dataset, polluter, level, model) 조합은 skip.
SVR(rbf)이 큰 데이터셋에서 느리므로 중간 저장 (10건마다)."""))

CELLS.append(code("""# ============================================================
# 2-1. 학습 루프 (체크포인트 지원, 회귀)
# ============================================================
perf_path = f'{RESULTS_DIR}/model_performance_regression.csv'

if os.path.isfile(perf_path):
    df_perf = pd.read_csv(perf_path)
    existing_keys = set(
        df_perf.apply(lambda r: f\"{r['dataset']}|{r['polluter']}|{r['level']}|{r['model']}\", axis=1)
    )
    perf_rows = df_perf.to_dict('records')
    print(f'기존 결과 {len(perf_rows)}건 로드')
else:
    existing_keys = set()
    perf_rows = []

total_start = time()
error_log = []
completed = 0
skipped = 0

for i, exp in enumerate(experiments):
    ds_name = exp['dataset']
    meta = DATASETS[ds_name]

    try:
        df_train = pd.read_csv(exp['train_path'])
        df_test = pd.read_csv(f'{TEST_DIR}/{ds_name}_test.csv')
        X_train, X_test, y_train, y_test = preprocess(df_train, df_test, meta)
    except Exception as e:
        label = f\"{ds_name}/{exp['polluter']}_{int(exp['level']*100)}%\"
        error_log.append({'label': label, 'model': 'ALL', 'error': str(e)})
        print(f'  [{i+1}/{len(experiments)}] {label} -> 전처리 실패: {e}')
        continue

    for model_name, model in get_models().items():
        key = f\"{ds_name}|{exp['polluter']}|{exp['level']}|{model_name}\"
        if key in existing_keys:
            skipped += 1
            continue
        try:
            t0 = time()
            model.fit(X_train, y_train)
            scores = evaluate_model(model, X_test, y_test)
            elapsed = time() - t0
            row = {
                'dataset': ds_name,
                'polluter': exp['polluter'],
                'level': exp['level'],
                'model': model_name,
                **scores,
            }
            perf_rows.append(row)
            existing_keys.add(key)
            completed += 1
        except Exception as e:
            label = f\"{ds_name}/{exp['polluter']}_{int(exp['level']*100)}%/{model_name}\"
            error_log.append({'label': label, 'error': str(e)})

    if (i + 1) % 5 == 0 or i == len(experiments) - 1:
        elapsed_total = time() - total_start
        print(f'  [{i+1}/{len(experiments)}] 완료={completed}, 스킵={skipped}, 에러={len(error_log)}  ({elapsed_total:.0f}s)')

    if (i + 1) % 10 == 0:
        pd.DataFrame(perf_rows).to_csv(perf_path, index=False)

# 최종 저장
pd.DataFrame(perf_rows).to_csv(perf_path, index=False)
print(f'\\n학습 완료: 완료={completed}, 스킵={skipped}, 에러={len(error_log)}')
print(f'결과 저장: {perf_path}')"""))

CELLS.append(code("""# ============================================================
# 2-2. 에러 로그 저장 (있을 경우)
# ============================================================
if error_log:
    err_path = f'{RESULTS_DIR}/03_regression_errors.csv'
    pd.DataFrame(error_log).to_csv(err_path, index=False)
    print(f'에러 {len(error_log)}건 저장: {err_path}')
else:
    print('에러 없음')"""))

CELLS.append(code("""# ============================================================
# 2-3. 결과 요약
# ============================================================
df_perf = pd.read_csv(perf_path)
print(f'총 학습: {len(df_perf)}건')
print()
print('데이터셋 × 모델 성능 요약 (R² 평균):')
pivot = df_perf.pivot_table(values='r2_clipped', index='model', columns='dataset', aggfunc='mean')
print(pivot.round(4))
print()
print('--- 노트북 03 회귀 버전 완료 ---')
print('다음: 04_scoreboard_regression.ipynb 실행')"""))

CELLS.append(code("""# ============================================================
# 3. 실행 로그 저장
# ============================================================
from datetime import datetime

log_lines = []
log_lines.append('# 노트북 03 회귀 버전 실행 로그')
log_lines.append('')
log_lines.append(f'- **실행 시각**: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}')
log_lines.append(f'- **총 학습**: {len(df_perf)}건')
log_lines.append(f'- **에러**: {len(error_log)}건')
log_lines.append('')

log_lines.append('## 1. 모델 성능 요약 (R² mean by dataset × model)')
log_lines.append('')
log_lines.append(pivot.round(4).to_markdown())
log_lines.append('')

log_lines.append('## 2. 폴루터별 평균 R² 하락')
log_lines.append('')
baseline_perf = df_perf[df_perf.polluter == 'none'].groupby('dataset')['r2_clipped'].mean()
log_lines.append('| 데이터셋 | baseline R² | completeness Δ | uniqueness Δ | feature_accuracy Δ | consistent_repr Δ | target_distribution_skew Δ |')
log_lines.append('|---|---|---|---|---|---|---|')
for ds in DATASETS:
    base = baseline_perf.get(ds, np.nan)
    cells = [ds, f'{base:.4f}']
    for p in ['completeness', 'uniqueness', 'feature_accuracy', 'consistent_repr', 'target_distribution_skew']:
        sub = df_perf[(df_perf.dataset == ds) & (df_perf.polluter == p)]
        if sub.empty:
            cells.append('-')
        else:
            delta = sub['r2_clipped'].mean() - base
            cells.append(f'{delta:+.4f}')
    log_lines.append('| ' + ' | '.join(cells) + ' |')

log_path = f'{RESULTS_DIR}/03_regression_execution_log.md'
with open(log_path, 'w', encoding='utf-8') as f:
    f.write('\\n'.join(log_lines))
print(f'실행 로그 저장: {log_path}')"""))


nb = {
    'cells': CELLS,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.11'},
    },
    'nbformat': 4,
    'nbformat_minor': 4,
}

os.makedirs(NB_OUT.parent, exist_ok=True)
with open(NB_OUT, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

n_md = sum(1 for c in CELLS if c['cell_type'] == 'markdown')
n_code = sum(1 for c in CELLS if c['cell_type'] == 'code')
print(f'생성: {NB_OUT}')
print(f'총 {len(CELLS)} 셀 (markdown {n_md}, code {n_code})')
