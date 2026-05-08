"""노트북 02 회귀 버전(02_pollution_and_dsc_regression.ipynb) 생성.

분류 노트북 02와의 변경:
- DSC 엔진: NEW_DSC_ENGINE 임베딩 → from dsc_framework import compute_dsc_regression
- 데이터셋: SGC/Telco/letter → CalH/Bike/Wine
- Polluter 5종: classbalance → target_distribution_skew (회귀 신설)
- split: stratify=y 제거 (회귀)
- target encoding 제거
- 결과 파일: dsc_scores_regression.csv

사전등록 (ADR-011, v5 마스터플랜 sect 3-3):
- POLLUTION_LEVELS = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
- 5 polluter × 6 level × 3 데이터셋 = 90 폴루션 + 3 baseline = 93건 DSC
- ConsistentRepresentation은 categorical_cols 있는 데이터만 (Bike만 해당)
"""
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NB_OUT = REPO / 'notebooks' / '02_pollution_and_dsc_regression.ipynb'


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)}


def code(text):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {},
            'outputs': [], 'source': text.splitlines(keepends=True)}


CELLS = []

CELLS.append(md("""# 02. Pollution & DSC Scoring (Regression Cell)

**Phase 1**: 회귀 cell polluter 적용 → split-first → train 폴루션 → DSC 점수 계산

DSC v5 framework — regression cell. 사전등록 polluter 라인업:
- CompletenessPolluter
- UniquenessPolluter
- FeatureAccuracyPolluter
- ConsistentRepresentationPolluter (categorical_cols 있는 데이터만)
- **TargetDistributionSkewPolluter** (회귀 신설, ClassBalance 대체)

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

BASE = '/content/drive/MyDrive/capstone/dsc'
RAW_DIR = f'{BASE}/data/raw'
POLLUTED_DIR = f'{BASE}/data/polluted'
RESULTS_DIR = f'{BASE}/results'
DQ4AI_DIR = f'{BASE}/dq4ai'

# DQ4AI를 import 경로에 추가 (polluter용)
sys.path.insert(0, DQ4AI_DIR)
# dsc_framework import 경로
if BASE not in sys.path:
    sys.path.insert(0, BASE)

print('환경 설정 완료')"""))

CELLS.append(code("""# ============================================================
# 0-2. 데이터셋 메타 + 오염 강도 사전등록
# ============================================================
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

POLLUTION_LEVELS = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
RANDOM_SEED = 42

print(f'데이터셋: {list(DATASETS.keys())}')
print(f'오염 강도: {POLLUTION_LEVELS}')"""))

CELLS.append(md("""## 1. DQ4AI Polluter 래핑 (회귀 cell)"""))

CELLS.append(code("""# ============================================================
# 1-1. polluter import + pandas 2.x 호환 패치
# ============================================================
if not hasattr(pd.DataFrame, 'append'):
    def _df_append(self, other, ignore_index=False, **kwargs):
        return pd.concat([self, other], ignore_index=ignore_index)
    pd.DataFrame.append = _df_append
    print('pandas 2.x 호환 패치 적용')

from polluters.completeness_polluter import CompletenessPolluter
from polluters.uniqueness_polluter import UniquenessPolluter
from polluters.feature_accuracy_polluter import FeatureAccuracyPolluter
from polluters.consistent_representation_polluter import ConsistentRepresentationPolluter
from polluters.target_distribution_skew_polluter import TargetDistributionSkewPolluter

print('DQ4AI polluter import 완료 (회귀 cell 5종)')"""))

CELLS.append(code("""# ============================================================
# 1-2. Polluter 인스턴스 생성 함수 (회귀 cell)
# ============================================================
def create_polluters(ds_name, meta, df):
    \"\"\"데이터셋별로 (polluter_name, level, polluter_instance) 리스트 생성 (회귀 cell).\"\"\"
    results = []
    target = meta['target']
    num_cols = meta['numerical_cols']
    cat_cols = meta['categorical_cols']
    ph_num = meta['placeholder_numerical']
    ph_cat = meta['placeholder_categorical']

    for level in POLLUTION_LEVELS:
        # --- Completeness ---
        results.append(('completeness', level, CompletenessPolluter(
            pollution_percentages=level,
            target_feature=target,
            placeholder_numerical=ph_num,
            placeholder_categorical=ph_cat if ph_cat is not None else 'empty',
            numerical_cols=num_cols,
            categorical_cols=cat_cols,
            random_seed=RANDOM_SEED,
        )))

        # --- Uniqueness ---
        factor_map = {0.1: 1.5, 0.25: 2.0, 0.5: 3.0, 0.75: 4.0, 0.9: 5.0, 0.95: 6.0}
        results.append(('uniqueness', level, UniquenessPolluter(
            duplicate_factor=factor_map[level],
            distribution_function_name='same',
            distribution_function_parameters={},
            target_feature=target,
            random_seed=RANDOM_SEED,
        )))

        # --- FeatureAccuracy ---
        results.append(('feature_accuracy', level, FeatureAccuracyPolluter(
            pollution_levels=level,
            categorical_cols=cat_cols,
            numerical_cols=num_cols,
            random_seed=RANDOM_SEED,
        )))

        # --- ConsistentRepresentation (categorical 있는 데이터셋만) ---
        if cat_cols:
            num_of_repr = {
                col: {val: 2 for val in df[col].dropna().unique()}
                for col in cat_cols if col in df.columns
            }
            results.append(('consistent_repr', level, ConsistentRepresentationPolluter(
                random_seed=RANDOM_SEED,
                percentage_polluted_rows=level,
                num_pollutable_columns=len(cat_cols),
                number_of_representations=num_of_repr,
            )))

        # --- TargetDistributionSkew (회귀 신설, classbalance 대체) ---
        results.append(('target_distribution_skew', level, TargetDistributionSkewPolluter(
            skew_level=level,
            target_column=target,
            random_seed=RANDOM_SEED,
        )))

    return results

print('Polluter 팩토리 정의 완료 (회귀 cell)')"""))

CELLS.append(md("""## 2. 오염 실행 & DSC 점수 계산 (회귀 cell)

**split-first 원칙** (피드백 메모리 + ML 파이프라인 프로세스):
1. clean 데이터 → train/test split (stratify 없음, 회귀)
2. split 인덱스 저장 (leakage 방지 1차 검증용)
3. train에만 polluter 적용
4. DSC와 ML이 동일한 train_polluted를 사용

reference_df = df_train_clean → outlier·target_distribution_quality의 분포 거리 비교 기준."""))

CELLS.append(code("""# ============================================================
# 2-1. DSC v5 회귀 엔진 import
# ============================================================
from dsc_framework import compute_dsc_regression
print('회귀 cell DSC 엔진 import 완료')"""))

CELLS.append(code("""# ============================================================
# 2-2. Split → train 폴루션 → DSC 계산
# ============================================================
from sklearn.model_selection import train_test_split
from time import time

TRAIN_DIR = f'{BASE}/data/train_polluted_regression'
TEST_DIR = f'{BASE}/data/test_clean_regression'
SPLIT_META_DIR = f'{BASE}/data/split_meta_regression'
ML_SPLIT_SEED = 1
ML_TEST_SIZE = 0.2

dsc_rows = []
train_total = 0
error_log = []

total_start = time()

for ds_name, meta in DATASETS.items():
    print()
    print('=' * 60)
    print(f'데이터셋: {ds_name}')
    print('=' * 60)

    df_clean = pd.read_csv(meta['path'])
    if meta['drop_cols']:
        df_clean = df_clean.drop(columns=[c for c in meta['drop_cols'] if c in df_clean.columns])

    target = meta['target']

    # --- Split (stratify 없음 - 회귀) ---
    train_idx, test_idx = train_test_split(
        np.arange(len(df_clean)), test_size=ML_TEST_SIZE,
        random_state=ML_SPLIT_SEED,
    )
    df_train_clean = df_clean.iloc[train_idx].reset_index(drop=True)
    df_test_clean = df_clean.iloc[test_idx].reset_index(drop=True)

    # --- Test + split 인덱스 저장 ---
    os.makedirs(TEST_DIR, exist_ok=True)
    df_test_clean.to_csv(f'{TEST_DIR}/{ds_name}_test.csv', index=False)
    os.makedirs(SPLIT_META_DIR, exist_ok=True)
    np.save(f'{SPLIT_META_DIR}/{ds_name}_train_idx.npy', train_idx)
    np.save(f'{SPLIT_META_DIR}/{ds_name}_test_idx.npy', test_idx)
    print(f'  Test 저장: {len(df_test_clean)}행 (split 인덱스도 함께 저장)')

    # --- Baseline (train clean) 저장 + DSC ---
    baseline_dir = f'{TRAIN_DIR}/{ds_name}/none_0'
    os.makedirs(baseline_dir, exist_ok=True)
    df_train_clean.to_csv(f'{baseline_dir}/train_data.csv', index=False)
    train_total += 1

    baseline_dsc = compute_dsc_regression(
        df_train_clean,
        target_col=meta['target'],
        numerical_cols=meta['numerical_cols'],
        categorical_cols=meta['categorical_cols'],
        placeholder_numerical=meta.get('placeholder_numerical', -1),
        placeholder_categorical=meta.get('placeholder_categorical', 'empty'),
        reference_df=df_train_clean,
    )
    dsc_rows.append({'dataset': ds_name, 'polluter': 'none', 'level': 0.0, **baseline_dsc})
    print(f'  baseline (train clean, {len(df_train_clean)}행) → DSC={baseline_dsc[\"score\"]:6.2f} ({baseline_dsc[\"grade\"]})')

    # --- 각 polluter를 train에 적용 + DSC 계산 ---
    polluter_list = create_polluters(ds_name, meta, df_train_clean)
    for polluter_name, level, polluter in polluter_list:
        label = f'{ds_name}/{polluter_name}_{int(level*100)}%'
        try:
            t0 = time()
            df_polluted = polluter.pollute(df_train_clean.copy())
            elapsed = time() - t0

            # 폴루션된 train 저장
            polluted_dir = f'{TRAIN_DIR}/{ds_name}/{polluter_name}_{int(level*100)}'
            os.makedirs(polluted_dir, exist_ok=True)
            df_polluted.to_csv(f'{polluted_dir}/train_data.csv', index=False)
            train_total += 1

            dsc_result = compute_dsc_regression(
                df_polluted,
                target_col=meta['target'],
                numerical_cols=meta['numerical_cols'],
                categorical_cols=meta['categorical_cols'],
                placeholder_numerical=meta.get('placeholder_numerical', -1),
                placeholder_categorical=meta.get('placeholder_categorical', 'empty'),
                reference_df=df_train_clean,
            )
            dsc_rows.append({'dataset': ds_name, 'polluter': polluter_name, 'level': level, **dsc_result})
            print(f'  {label:50s} → DSC={dsc_result[\"score\"]:6.2f} ({dsc_result[\"grade\"]})  [{elapsed:.1f}s]')
        except Exception as e:
            error_log.append({'label': label, 'error': str(e)})
            print(f'  {label:50s} → ERROR: {e}')

total_elapsed = time() - total_start
print()
print(f'총 train 데이터 {train_total}건, DSC {len(dsc_rows)}건 ({total_elapsed:.0f}초)')
if error_log:
    print(f'에러 {len(error_log)}건:')
    for e in error_log:
        print(f'  {e[\"label\"]}: {e[\"error\"]}')"""))

CELLS.append(code("""# ============================================================
# 2-3. DSC 점수 저장 (회귀 cell — 별도 파일)
# ============================================================
df_dsc = pd.DataFrame(dsc_rows)
dsc_path = f'{RESULTS_DIR}/dsc_scores_regression.csv'
df_dsc.to_csv(dsc_path, index=False)

print(f'DSC 점수 저장: {dsc_path}')
print(f'총 {len(df_dsc)}건')
print(f'\\n--- 노트북 02 회귀 버전 완료 ---')
print(f'다음: 03_training_regression.ipynb 실행')

df_dsc.head(15)"""))

CELLS.append(code("""# ============================================================
# 3. 실행 로그 저장 (회귀 cell)
# ============================================================
from datetime import datetime

log_lines = []
log_lines.append('# 노트북 02 회귀 버전 실행 로그')
log_lines.append('')
log_lines.append(f'- **실행 시각**: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}')
log_lines.append(f'- **총 실험**: {len(dsc_rows)}건 (베이스라인 포함)')
log_lines.append(f'- **소요 시간**: {total_elapsed:.0f}초')
log_lines.append(f'- **에러**: {len(error_log)}건')
log_lines.append('')

log_lines.append('## 1. 오염 설정')
log_lines.append('')
log_lines.append(f'- **데이터셋**: {list(DATASETS.keys())}')
log_lines.append(f'- **오염 강도**: {POLLUTION_LEVELS}')
log_lines.append(f'- **Polluter**: Completeness, Uniqueness, FeatureAccuracy, ConsistentRepresentation(범주형 데이터셋만), TargetDistributionSkew')
log_lines.append('')

log_lines.append('## 2. DSC 점수 결과 (회귀 cell)')
log_lines.append('')
metric_keys = ['completeness', 'uniqueness', 'validity', 'consistency', 'outlier_ratio',
               'target_distribution_quality', 'feature_correlation', 'target_smoothness',
               'feature_informativeness_reg']
header = '| 데이터셋 | 오염 유형 | 강도 | DSC | 등급 | ' + ' | '.join(metric_keys) + ' |'
log_lines.append(header)
log_lines.append('|' + '---|' * (len(metric_keys) + 5))
for row in dsc_rows:
    cells = [row['dataset'], row['polluter'], row['level'], row['score'], row['grade']]
    cells += [row.get(k, '') for k in metric_keys]
    log_lines.append('| ' + ' | '.join(str(c) for c in cells) + ' |')
log_lines.append('')

log_path = f'{RESULTS_DIR}/02_regression_execution_log.md'
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
