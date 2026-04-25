"""P6 — 노트북 02 셀 9·12 통합.

기존 구조:
- cell 9: full clean → polluter.pollute → DSC 계산 + polluted/ 저장 (DSC 트랙)
- cell 12: split → train만 polluter.pollute → train_polluted/ 저장 (ML 트랙)
→ 두 트랙이 다른 데이터에 polluter 따로 적용. DSC 측정값과 ML 학습 데이터 불일치.

새 구조 (split-first 단일 트랙):
- cell 9: split → train만 polluter.pollute → train_polluted/ 저장 + 같은 결과로 DSC 계산
  reference는 train_clean. test_clean도 같은 셀에서 저장.
- cell 11(markdown), cell 12(code) 삭제. dsc_scores.csv 저장은 cell 10이 그대로.
"""
import json

NEW_CELL_9 = '''# ============================================================
# 2-2. Split → Train 폴루션 → DSC 계산 (단일 트랙)
# ============================================================
# split-first 원칙: clean → train/test split → train에만 polluter 적용
# DSC와 ML이 동일한 train_polluted를 사용 (P6 해결).
# reference_df = df_train_clean → outlier·value_accuracy의 분포 거리 비교 기준.
# ============================================================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder as _LE
from time import time

TRAIN_DIR = f'{BASE}/data/train_polluted'
TEST_DIR = f'{BASE}/data/test_clean'
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
    if 'TotalCharges' in df_clean.columns:
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce').fillna(0)

    # --- Split (고정 seed, stratified) ---
    target = meta['target']
    _le = _LE()
    y_encoded = _le.fit_transform(df_clean[target].astype(str))
    train_idx, test_idx = train_test_split(
        np.arange(len(df_clean)), test_size=ML_TEST_SIZE,
        random_state=ML_SPLIT_SEED, stratify=y_encoded
    )
    df_train_clean = df_clean.iloc[train_idx].reset_index(drop=True)
    df_test_clean = df_clean.iloc[test_idx].reset_index(drop=True)

    # --- Test 저장 (데이터셋당 1회) ---
    os.makedirs(TEST_DIR, exist_ok=True)
    df_test_clean.to_csv(f'{TEST_DIR}/{ds_name}_test.csv', index=False)
    print(f'  Test 저장: {len(df_test_clean)}행')

    # --- Baseline (train clean) 저장 + DSC ---
    baseline_dir = f'{TRAIN_DIR}/{ds_name}/none_0'
    os.makedirs(baseline_dir, exist_ok=True)
    df_train_clean.to_csv(f'{baseline_dir}/train_data.csv', index=False)
    train_total += 1

    baseline_dsc = compute_dsc(
        df_train_clean,
        target_col=meta['target'],
        numerical_cols=meta['numerical_cols'],
        categorical_cols=meta['categorical_cols'],
        placeholder_numerical=meta.get('placeholder_numerical', -1),
        placeholder_categorical=meta.get('placeholder_categorical', 'empty'),
        reference_df=df_train_clean,
    )
    dsc_rows.append({'dataset': ds_name, 'polluter': 'none', 'level': 0.0, **baseline_dsc})
    print(f'  baseline (train clean, {len(df_train_clean)}행) → DSC={baseline_dsc["score"]:6.2f} ({baseline_dsc["grade"]})')

    # --- 각 polluter를 train에 적용 + DSC 계산 ---
    polluter_list = create_polluters(ds_name, meta, df_train_clean)
    for polluter_name, level, polluter in polluter_list:
        label = f'{ds_name}/{polluter_name}_{int(level*100)}%'
        try:
            t0 = time()
            polluted_train = polluter.pollute(df_train_clean.copy())
            elapsed = time() - t0

            save_dir = f'{TRAIN_DIR}/{ds_name}/{polluter_name}_{int(level*100)}'
            os.makedirs(save_dir, exist_ok=True)
            polluted_train.to_csv(f'{save_dir}/train_data.csv', index=False)
            train_total += 1

            dsc_result = compute_dsc(
                polluted_train,
                target_col=meta['target'],
                numerical_cols=meta['numerical_cols'],
                categorical_cols=meta['categorical_cols'],
                placeholder_numerical=meta.get('placeholder_numerical', -1),
                placeholder_categorical=meta.get('placeholder_categorical', 'empty'),
                reference_df=df_train_clean,
            )
            dsc_rows.append({'dataset': ds_name, 'polluter': polluter_name, 'level': level, **dsc_result})
            print(f'  {label:45s} → DSC={dsc_result["score"]:6.2f} ({dsc_result["grade"]})  [{elapsed:.1f}s]')
        except Exception as e:
            error_log.append({'label': label, 'error': str(e)})
            print(f'  {label:45s} → ERROR: {e}')

total_elapsed = time() - total_start
print()
print(f'총 train 데이터 {train_total}건, DSC {len(dsc_rows)}건 ({total_elapsed:.0f}초)')
if error_log:
    print(f'에러 {len(error_log)}건:')
    for e in error_log:
        print(f'  {e["label"]}: {e["error"]}')'''

path = 'G:/내 드라이브/capstone/dsc/notebooks/02_pollution_and_dsc.ipynb'
nb = json.load(open(path, encoding='utf-8'))

# cell 9 교체
nb['cells'][9]['source'] = [NEW_CELL_9]
print('cell 9 통합 코드로 교체')

# cell 11 (markdown) + cell 12 (code) 삭제
# 인덱스 주의: 12를 먼저 삭제해야 11 인덱스 안 바뀜
del nb['cells'][12]
del nb['cells'][11]
print('cell 11(markdown) + cell 12(code) 삭제')

# 저장
json.dump(nb, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('노트북 02 저장 완료')

# 구조 재확인
nb = json.load(open(path, encoding='utf-8'))
print()
print('새 셀 구조:')
for i, cell in enumerate(nb['cells']):
    head = ''.join(cell['source']).split('\n')[0][:80]
    print(f'  cell {i:2d} [{cell["cell_type"]:8s}] {head}')

# syntax 검증
import ast
print()
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code': continue
    src = ''.join(cell['source'])
    clean = '\n'.join(l for l in src.split('\n') if not l.strip().startswith(('!', '%')))
    try:
        ast.parse(clean)
    except SyntaxError as e:
        print(f'cell {i} SYNTAX ERROR: {e}')
print('syntax 검증 통과')
