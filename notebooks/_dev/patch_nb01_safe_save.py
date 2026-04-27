"""01 cell 18 패치 — model_performance.csv가 폴루션 결과까지 포함된 상태에서 01만 다시 돌려도
baseline 행만 upsert하고 폴루션 결과는 보존하도록.
"""
import json
import ast

NB = 'G:/내 드라이브/capstone/dsc/notebooks/01_setup_and_baseline.ipynb'

NEW_CELL_18 = """# ============================================================
# 5-1. 베이스라인 결과를 CSV로 저장 (체크포인트 안전 — baseline만 upsert)
# ============================================================
dsc_scores_path = f'{RESULTS_DIR}/dsc_scores.csv'
model_perf_path = f'{RESULTS_DIR}/model_performance.csv'

def upsert_baseline(path, new_df):
    '''기존 csv 있으면 baseline (none, level=0.0) 행만 갱신, 폴루션 결과 보존.'''
    if os.path.isfile(path):
        try:
            existing = pd.read_csv(path)
            baseline_mask = (existing.polluter == 'none') & (existing.level == 0.0)
            kept = existing[~baseline_mask].copy()
            # 새 컬럼 추가 (구 baseline에 없던 신지표 등)
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
print('--- 노트북 01 완료 ---')
print('다음: 02_pollution_and_dsc.ipynb 실행')
"""

nb = json.load(open(NB, encoding='utf-8'))
nb['cells'][18]['source'] = [NEW_CELL_18]
json.dump(nb, open(NB, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

# syntax 검증
src = ''.join(nb['cells'][18]['source'])
clean = '\n'.join(l for l in src.split('\n') if not l.lstrip().startswith(('!', '%')))
ast.parse(clean)
print('cell 18 패치 완료, syntax OK')
