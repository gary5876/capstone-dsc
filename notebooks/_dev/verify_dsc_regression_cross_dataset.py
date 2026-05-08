"""
DSC 회귀 엔진 v5 — Cross-dataset 안정성 검증.

3개 회귀 데이터셋에서 동일한 검증 시나리오를 반복하여 메트릭이 데이터 종류와
크기에 무관하게 합리적으로 동작하는지 확인.

데이터셋 (사전 다운로드 완료, data/raw/):
- California Housing: 20,640 × 8 numerical, target=MedHouseVal (continuous, 0.15~5.0)
- Bike Sharing hour:  17,379 × 12 mixed,    target=cnt          (count, 1~977)
- Wine Quality:        6,497 × 11 numerical, target=quality      (ordinal int, 3~9)

검증 시나리오 (각 데이터셋별 동일):
  S1. baseline DSC (clean)
  S2. target_distribution_skew(0.75) → TDQ Δ ≤ -0.10
  S3. feature_accuracy noise(0.75)   → TS Δ ≤ -0.10
  S4. uniqueness duplication(×4)     → TS/FI |Δ| ≤ 0.05 (false positive 없음)
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

from dq4ai.dsc_engine_regression_v5 import compute_dsc_regression
from dq4ai.polluters.target_distribution_skew_polluter import TargetDistributionSkewPolluter

RAW = os.path.join(PROJECT_ROOT, 'data', 'raw')


def load_california():
    df = pd.read_csv(os.path.join(RAW, 'california_housing.csv'))
    target_col = 'MedHouseVal'
    feature_cols = [c for c in df.columns if c != target_col]
    return df, target_col, feature_cols, []


def load_bike_sharing():
    df = pd.read_csv(os.path.join(RAW, 'bike_sharing_hour.csv'))
    # Drop utility/leakage columns
    drop_cols = ['instant', 'dteday', 'casual', 'registered']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    target_col = 'cnt'
    # Treat truly continuous + ordinal as numerical; binary/categorical-encoded as cat.
    numerical = ['temp', 'atemp', 'hum', 'windspeed', 'mnth', 'hr', 'weekday']
    categorical = ['season', 'yr', 'holiday', 'workingday', 'weathersit']
    # Cast categorical-encoded ints to str for consistent handling
    for col in categorical:
        df[col] = df[col].astype(str)
    return df, target_col, numerical, categorical


def load_wine_quality():
    df = pd.read_csv(os.path.join(RAW, 'wine_quality.csv'))
    target_col = 'quality'
    feature_cols = [c for c in df.columns if c != target_col]
    return df, target_col, feature_cols, []


DATASETS = {
    'California Housing': load_california,
    'Bike Sharing':       load_bike_sharing,
    'Wine Quality':       load_wine_quality,
}


def run_scenarios(name, df, target_col, num_cols, cat_cols):
    print(f"\n{'─' * 64}")
    print(f"[{name}] {df.shape[0]} rows × {df.shape[1]} cols "
          f"(num={len(num_cols)}, cat={len(cat_cols)})")
    print(f"  target='{target_col}', range=[{pd.to_numeric(df[target_col]).min():.3f}, "
          f"{pd.to_numeric(df[target_col]).max():.3f}]")

    # S1. Baseline
    baseline = compute_dsc_regression(df, target_col, num_cols, cat_cols, reference_df=df)
    print(f"\n  [S1] Baseline DSC: score={baseline['score']:.2f} (grade {baseline['grade']})")
    print(f"       완전성·유일성·신지표:")
    print(f"         completeness={baseline['completeness']:.4f}  "
          f"uniqueness={baseline['uniqueness']:.4f}")
    print(f"         target_smoothness={baseline['target_smoothness']:.4f}  "
          f"target_distribution_quality={baseline['target_distribution_quality']:.4f}")
    print(f"         feature_informativeness_reg={baseline['feature_informativeness_reg']:.4f}")

    results = {'name': name, 'baseline': baseline, 'pass': []}

    # S2. target_distribution_skew(0.75) → TDQ Δ ≤ -0.10
    polluter = TargetDistributionSkewPolluter(
        skew_level=0.75, target_column=target_col, random_seed=1)
    polluted = polluter.pollute(df)
    res = compute_dsc_regression(polluted, target_col, num_cols, cat_cols, reference_df=df)
    delta_tdq = res['target_distribution_quality'] - baseline['target_distribution_quality']
    s2_pass = delta_tdq <= -0.10
    print(f"\n  [S2] skew=0.75: TDQ {baseline['target_distribution_quality']:.4f} → "
          f"{res['target_distribution_quality']:.4f} (Δ={delta_tdq:+.4f}) "
          f"{'✅ PASS' if s2_pass else '❌ FAIL (≤-0.10 요구)'}")
    results['pass'].append(('S2', s2_pass, delta_tdq))

    # S3. feature_accuracy noise(0.75) → TS Δ ≤ -0.10
    rng = np.random.RandomState(1)
    polluted = df.copy()
    for col in num_cols:
        s = pd.to_numeric(polluted[col], errors='coerce')
        if s.std() == 0 or pd.isna(s.std()):
            continue
        mask = rng.rand(len(polluted)) < 0.75
        noise = rng.randn(mask.sum()) * s.std() * 3
        polluted.loc[mask, col] = s.loc[mask] + noise
    res = compute_dsc_regression(polluted, target_col, num_cols, cat_cols, reference_df=df)
    delta_ts = res['target_smoothness'] - baseline['target_smoothness']
    s3_pass = delta_ts <= -0.10
    print(f"  [S3] feat noise 0.75: TS {baseline['target_smoothness']:.4f} → "
          f"{res['target_smoothness']:.4f} (Δ={delta_ts:+.4f}) "
          f"{'✅ PASS' if s3_pass else '❌ FAIL (≤-0.10 요구)'}")
    results['pass'].append(('S3', s3_pass, delta_ts))

    # S4. uniqueness duplication(×4) → TS, FI |Δ| ≤ 0.05
    n_dup = int(len(df) * 3.0)
    dup_idx = np.random.RandomState(1).choice(df.index, size=n_dup, replace=True)
    duplicated = pd.concat([df, df.loc[dup_idx]], ignore_index=True)
    res = compute_dsc_regression(duplicated, target_col, num_cols, cat_cols, reference_df=df)
    abs_dts = abs(res['target_smoothness'] - baseline['target_smoothness'])
    abs_dfi = abs(res['feature_informativeness_reg'] - baseline['feature_informativeness_reg'])
    s4_pass = abs_dts <= 0.05 and abs_dfi <= 0.05
    print(f"  [S4] dup ×4: |Δ TS|={abs_dts:.4f}  |Δ FI|={abs_dfi:.4f} "
          f"{'✅ PASS (false positive 없음)' if s4_pass else '❌ FAIL (|Δ| > 0.05)'}")
    results['pass'].append(('S4', s4_pass, max(abs_dts, abs_dfi)))

    return results


def main():
    print("=" * 64)
    print("DSC 회귀 엔진 v5 — Cross-dataset 안정성 검증")
    print("=" * 64)

    all_results = []
    for name, loader in DATASETS.items():
        try:
            df, target_col, num_cols, cat_cols = loader()
            res = run_scenarios(name, df, target_col, num_cols, cat_cols)
            all_results.append(res)
        except Exception as e:
            import traceback
            print(f"\n❌ [{name}] 검증 중 예외: {e}")
            traceback.print_exc()
            all_results.append({'name': name, 'pass': [('ERR', False, None)]})

    # 통합 요약 표
    print(f"\n{'=' * 64}")
    print("통합 요약")
    print(f"{'=' * 64}\n")

    print(f"{'Dataset':<20} {'baseline':>9} {'TDQ':>7} {'TS':>7} {'FI':>7}  {'S2':>3} {'S3':>3} {'S4':>3}")
    print(f"{'-' * 20} {'-' * 9} {'-' * 7} {'-' * 7} {'-' * 7}  {'-' * 3} {'-' * 3} {'-' * 3}")
    total_checks = 0
    passed_checks = 0
    for r in all_results:
        if 'baseline' not in r:
            print(f"{r['name']:<20} {'(error)':>9}")
            continue
        b = r['baseline']
        marks = {sid: '✅' if p else '❌' for sid, p, _ in r['pass']}
        print(f"{r['name']:<20} {b['score']:>9.2f} {b['target_distribution_quality']:>7.4f} "
              f"{b['target_smoothness']:>7.4f} {b['feature_informativeness_reg']:>7.4f}  "
              f"{marks.get('S2', '-'):>3} {marks.get('S3', '-'):>3} {marks.get('S4', '-'):>3}")
        for _, p, _ in r['pass']:
            total_checks += 1
            if p:
                passed_checks += 1

    print(f"\n전체 검증: {passed_checks}/{total_checks} PASS")
    if passed_checks == total_checks:
        print("\n✅ ALL CROSS-DATASET CHECKS PASSED")
    else:
        print(f"\n❌ {total_checks - passed_checks} 항목 실패 — 검토 필요")
        sys.exit(1)


if __name__ == '__main__':
    main()
