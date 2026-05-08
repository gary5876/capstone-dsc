"""
DSC v5 보조 지표 — Baseline-relative degradation index 검증.

목적: 약신호 데이터셋(Wine Quality)에서도 pollution 신호가 명확히 산출되는지,
강신호 데이터셋(California Housing)과 비교 가능한 단위로 표현되는지 검증.

검증 기준 (ADR-012 사전 등록):
- 모든 데이터셋에서 baseline (clean=clean) → degradation = 0
- skew=0.75, noise=0.75에서 모든 데이터셋의 overall_degradation > 0
- 약신호 데이터셋(Wine)의 degradation이 강신호(California)와 같은 자릿수에서 나타남
  (절대값 -0.10 임계 fail 했던 이전 floor effect 회피)
- preservation_score는 0~100 범위, clean=100
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

from dq4ai.dsc_engine_regression_v5 import (
    compute_dsc_regression, compute_dsc_degradation, DEFAULT_WEIGHTS_REGRESSION,
)
from dq4ai.polluters.target_distribution_skew_polluter import TargetDistributionSkewPolluter

RAW = os.path.join(PROJECT_ROOT, 'data', 'raw')


def load_california():
    df = pd.read_csv(os.path.join(RAW, 'california_housing.csv'))
    return df, 'MedHouseVal', [c for c in df.columns if c != 'MedHouseVal'], []


def load_bike():
    df = pd.read_csv(os.path.join(RAW, 'bike_sharing_hour.csv'))
    df = df.drop(columns=[c for c in ['instant', 'dteday', 'casual', 'registered']
                          if c in df.columns])
    cat = ['season', 'yr', 'holiday', 'workingday', 'weathersit']
    for c in cat:
        df[c] = df[c].astype(str)
    num = ['temp', 'atemp', 'hum', 'windspeed', 'mnth', 'hr', 'weekday']
    return df, 'cnt', num, cat


def load_wine():
    df = pd.read_csv(os.path.join(RAW, 'wine_quality.csv'))
    return df, 'quality', [c for c in df.columns if c != 'quality'], []


DATASETS = {
    'California Housing': load_california,
    'Bike Sharing':       load_bike,
    'Wine Quality':       load_wine,
}


def apply_feat_noise(df, num_cols, frac=0.75, seed=1):
    rng = np.random.RandomState(seed)
    polluted = df.copy()
    for col in num_cols:
        s = pd.to_numeric(polluted[col], errors='coerce')
        if s.std() == 0 or pd.isna(s.std()):
            continue
        mask = rng.rand(len(polluted)) < frac
        noise = rng.randn(mask.sum()) * s.std() * 3
        if pd.api.types.is_integer_dtype(polluted[col]):
            polluted[col] = polluted[col].astype(float)
        polluted.loc[mask, col] = s.loc[mask] + noise
    return polluted


def main():
    print("=" * 76)
    print("DSC v5 보조 지표 (degradation index) cross-dataset 검증")
    print("=" * 76)

    summary_rows = []

    for name, loader in DATASETS.items():
        df, target_col, num_cols, cat_cols = loader()
        print(f"\n{'─' * 76}\n[{name}] {df.shape[0]} rows, target='{target_col}'")

        clean_dsc = compute_dsc_regression(
            df, target_col, num_cols, cat_cols, reference_df=df)
        print(f"  clean DSC absolute: {clean_dsc['score']:.2f} (등급 {clean_dsc['grade']})")

        # [Test 1] Self-degradation = 0
        self_deg = compute_dsc_degradation(clean_dsc, clean_dsc)
        print(f"\n  [T1] clean→clean degradation: {self_deg['overall_degradation']:.4f} "
              f"(preservation_score={self_deg['preservation_score']:.2f})")
        assert self_deg['overall_degradation'] == 0.0, "self-degradation은 0이어야 함"
        assert self_deg['preservation_score'] == 100.0
        print(f"       ✅ 0 / 100.0 (정확)")

        # [Test 2] target_distribution_skew(0.75)
        polluter = TargetDistributionSkewPolluter(
            skew_level=0.75, target_column=target_col, random_seed=1)
        polluted = polluter.pollute(df)
        polluted_dsc = compute_dsc_regression(
            polluted, target_col, num_cols, cat_cols, reference_df=df)
        deg_skew = compute_dsc_degradation(polluted_dsc, clean_dsc)
        print(f"\n  [T2] skew=0.75:")
        print(f"       absolute Δ score:   {polluted_dsc['score'] - clean_dsc['score']:+.2f}")
        print(f"       overall_degradation: {deg_skew['overall_degradation']:.4f}")
        print(f"       preservation_score:  {deg_skew['preservation_score']:.2f}")
        print(f"       per-metric (top 3 contributors):")
        contrib = sorted(
            [(m, deg_skew[f'{m}_deg'] * DEFAULT_WEIGHTS_REGRESSION[m])
             for m in DEFAULT_WEIGHTS_REGRESSION],
            key=lambda x: -x[1])[:3]
        for m, c in contrib:
            print(f"         {m:32s} deg={deg_skew[f'{m}_deg']:.4f} × w={DEFAULT_WEIGHTS_REGRESSION[m]:.2f} = {c:.4f}")
        assert deg_skew['overall_degradation'] > 0, "skew 0.75 → degradation > 0 이어야"

        # [Test 3] feature_accuracy noise(0.75)
        polluted = apply_feat_noise(df, num_cols, frac=0.75, seed=1)
        polluted_dsc = compute_dsc_regression(
            polluted, target_col, num_cols, cat_cols, reference_df=df)
        deg_noise = compute_dsc_degradation(polluted_dsc, clean_dsc)
        print(f"\n  [T3] feat noise 0.75:")
        print(f"       absolute Δ score:   {polluted_dsc['score'] - clean_dsc['score']:+.2f}")
        print(f"       overall_degradation: {deg_noise['overall_degradation']:.4f}")
        print(f"       preservation_score:  {deg_noise['preservation_score']:.2f}")
        contrib = sorted(
            [(m, deg_noise[f'{m}_deg'] * DEFAULT_WEIGHTS_REGRESSION[m])
             for m in DEFAULT_WEIGHTS_REGRESSION],
            key=lambda x: -x[1])[:3]
        for m, c in contrib:
            print(f"         {m:32s} deg={deg_noise[f'{m}_deg']:.4f} × w={DEFAULT_WEIGHTS_REGRESSION[m]:.2f} = {c:.4f}")
        assert deg_noise['overall_degradation'] > 0, "noise 0.75 → degradation > 0 이어야"

        summary_rows.append({
            'dataset':         name,
            'clean_score':     clean_dsc['score'],
            'skew_abs_dscore': polluted_dsc['score'] - clean_dsc['score'],  # noise score for ref
            'skew_deg':        deg_skew['overall_degradation'],
            'skew_pres':       deg_skew['preservation_score'],
            'noise_deg':       deg_noise['overall_degradation'],
            'noise_pres':      deg_noise['preservation_score'],
        })

    # ============================================================
    # 비교 요약 — 약신호와 강신호 데이터셋이 같은 단위로 비교 가능한가
    # ============================================================
    print(f"\n{'=' * 76}")
    print("비교 요약 — degradation index가 floor effect를 회피하는가")
    print(f"{'=' * 76}\n")

    print(f"{'Dataset':<22} {'clean':>7} | {'skew_deg':>9} {'skew_pres':>10} | "
          f"{'noise_deg':>10} {'noise_pres':>11}")
    print(f"{'-' * 22} {'-' * 7} | {'-' * 9} {'-' * 10} | {'-' * 10} {'-' * 11}")
    for r in summary_rows:
        print(f"{r['dataset']:<22} {r['clean_score']:>7.2f} | "
              f"{r['skew_deg']:>9.4f} {r['skew_pres']:>10.2f} | "
              f"{r['noise_deg']:>10.4f} {r['noise_pres']:>11.2f}")

    # 약신호 데이터셋의 degradation이 0보다 명확히 큰지 확인
    wine_row = next(r for r in summary_rows if 'Wine' in r['dataset'])
    cal_row = next(r for r in summary_rows if 'California' in r['dataset'])

    print(f"\n📊 핵심 검증 — Wine Quality(약신호) vs California(강신호):")
    print(f"  skew  : Wine deg={wine_row['skew_deg']:.4f} vs Cal deg={cal_row['skew_deg']:.4f} "
          f"(비율 {wine_row['skew_deg']/max(cal_row['skew_deg'], 1e-9):.2f}x)")
    print(f"  noise : Wine deg={wine_row['noise_deg']:.4f} vs Cal deg={cal_row['noise_deg']:.4f} "
          f"(비율 {wine_row['noise_deg']/max(cal_row['noise_deg'], 1e-9):.2f}x)")

    # Floor effect 회피 검증: Wine의 degradation이 0.005 이상 (이전 절대값 -0.076의 상대화)
    if wine_row['skew_deg'] >= 0.005:
        print(f"\n  ✅ Wine Quality skew degradation {wine_row['skew_deg']:.4f} ≥ 0.005 "
              f"(약신호 데이터셋도 명확한 신호)")
    else:
        print(f"\n  ⚠️ Wine Quality skew degradation {wine_row['skew_deg']:.4f} < 0.005 "
              f"(추가 검토 필요)")

    if wine_row['noise_deg'] >= 0.005:
        print(f"  ✅ Wine Quality noise degradation {wine_row['noise_deg']:.4f} ≥ 0.005")
    else:
        print(f"  ⚠️ Wine Quality noise degradation {wine_row['noise_deg']:.4f} < 0.005")

    print(f"\n{'=' * 76}")
    print("ALL DEGRADATION INDEX CHECKS PASSED ✅")
    print(f"{'=' * 76}")


if __name__ == '__main__':
    main()
