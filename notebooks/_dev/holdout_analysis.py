"""F1 — Polluter Hold-out 검증.

목적: v4 가중치가 ML 결과를 보고 fitting 한 결과인지(circular reasoning),
     아니면 polluter에 무관하게 일관되게 동작하는 진짜 신호인지 검증.

방법:
  - 5개 polluter (completeness, uniqueness, consistent_repr, class_balance, feature_accuracy)
    각각을 hold-out으로 지정.
  - hold-out 슬라이스만 추출하여 Pearson r 측정.
  - 비교군: hold-out 슬라이스에서 v3.2와 v4.0이 각각 얻는 r.

검증 기준:
  - 5개 hold-out 모두에서 v4 r > 0.3 + p < 0.05면 "fitting 아닌 일반화 신호"
  - 4/5 통과면 부분 통과 — 약한 1개는 한계로 인정
  - v4 r이 v3.2 r보다 모든 또는 대부분 hold-out에서 향상되어야 함
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


BASE = Path('G:/내 드라이브/capstone/dsc')
DEV = BASE / 'notebooks' / '_dev'
RESULTS = BASE / 'results'

POLLUTERS = ['completeness', 'uniqueness', 'consistent_repr', 'class_balance', 'feature_accuracy']


def hold_out_r(merged: pd.DataFrame, polluter_holdout: str) -> dict:
    """hold-out polluter 슬라이스의 r 통계 + 전체에서 hold-out 제외한 r.
    none(baseline) 행은 두 슬라이스 모두에서 제외 (level=0이라 변동성 없음)."""
    sub = merged[(merged.polluter == polluter_holdout)].copy()
    if len(sub) < 10:
        return {'holdout': polluter_holdout, 'n': len(sub), 'r': np.nan, 'p': np.nan}
    r, p = sp_stats.pearsonr(sub.score, sub.f1_macro)
    return {'holdout': polluter_holdout, 'n': len(sub), 'r': r, 'p': p}


def main():
    print('=' * 80)
    print('F1 검증 — Polluter Hold-out 분석')
    print('=' * 80)

    # v3.2 (현재 정식 결과)
    print('\n--- v3.2 (현재 정식 결과) ---')
    merged_v32 = pd.read_csv(RESULTS / 'merged_results.csv')
    print(f'  표본 수: {len(merged_v32)}, polluter: {sorted(merged_v32.polluter.unique())}')
    rows_v32 = []
    for pol in POLLUTERS:
        if pol not in merged_v32.polluter.unique():
            continue
        out = hold_out_r(merged_v32, pol)
        rows_v32.append(out)
        ok = (out['r'] > 0.3) and (out['p'] < 0.05)
        mark = 'PASS' if ok else 'FAIL'
        print(f"  hold-out {pol:20s}  r={out['r']:+.3f}  p={out['p']:.2e}  n={out['n']}  [{mark}]")

    # v4.0 시뮬
    print('\n--- v4.0 (시뮬레이션) ---')
    merged_v4_path = DEV / 'simulated_v4_merged.csv'
    if not merged_v4_path.exists():
        print(f'  {merged_v4_path} 없음 → simulate_v4_scores.py 먼저 실행')
        return
    merged_v4 = pd.read_csv(merged_v4_path)
    print(f'  표본 수: {len(merged_v4)}, polluter: {sorted(merged_v4.polluter.unique())}')
    rows_v4 = []
    for pol in POLLUTERS:
        if pol not in merged_v4.polluter.unique():
            continue
        out = hold_out_r(merged_v4, pol)
        rows_v4.append(out)
        ok = (out['r'] > 0.3) and (out['p'] < 0.05)
        mark = 'PASS' if ok else 'FAIL'
        print(f"  hold-out {pol:20s}  r={out['r']:+.3f}  p={out['p']:.2e}  n={out['n']}  [{mark}]")

    # 비교
    print('\n--- v3.2 vs v4.0 비교 ---')
    df32 = pd.DataFrame(rows_v32).set_index('holdout')
    df4  = pd.DataFrame(rows_v4).set_index('holdout')
    cmp = pd.DataFrame({
        'v3.2_r': df32['r'], 'v3.2_p': df32['p'],
        'v4_r':   df4['r'],   'v4_p':   df4['p'],
    })
    cmp['Δr'] = cmp['v4_r'] - cmp['v3.2_r']
    print(cmp.to_string(float_format=lambda x: f'{x:+.4f}' if abs(x) < 100 else str(x)))

    # 통과 검증
    pass_v4 = (cmp['v4_r'] > 0.3) & (cmp['v4_p'] < 0.05)
    print(f'\nv4 hold-out 통과: {pass_v4.sum()}/{len(pass_v4)}')
    print(f"통과 polluter: {pass_v4[pass_v4].index.tolist()}")
    print(f"실패 polluter: {pass_v4[~pass_v4].index.tolist()}")

    # 출력 저장
    out_path = DEV / 'holdout_analysis_results.csv'
    cmp.to_csv(out_path)
    print(f'\n결과 저장: {out_path}')


if __name__ == '__main__':
    main()
