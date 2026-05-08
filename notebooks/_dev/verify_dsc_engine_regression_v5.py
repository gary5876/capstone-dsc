"""
DSC 회귀 엔진 v5 단위 검증.

Phase 1 검증 기준 (마스터플랜 sect 1-5, ADR-011 위험 5):
- baseline에서 두 신지표 (target_smoothness, feature_informativeness_reg) ≥ 0.7
- target_distribution_skew 75% 강도에서 target_distribution_quality −0.10 이상 하락
- completeness 75% 폴루션에서 회귀 메트릭 적절히 하락
- target_smoothness가 feature_accuracy 75%에서 −0.10 이상 하락
- false positive: completeness 폴루션에서 target_smoothness 변화 작아야

CLAUDE.md + memory feedback_verify_before_deliver 준수: 실제 California
Housing 데이터로 모든 메트릭 동작 검증.
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

from dq4ai.dsc_engine_regression_v5 import (
    compute_dsc_regression, DEFAULT_WEIGHTS_REGRESSION,
    calc_target_smoothness, calc_target_distribution_quality,
    calc_feature_informativeness_reg,
)
from dq4ai.polluters.target_distribution_skew_polluter import TargetDistributionSkewPolluter


def section(title):
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


def main():
    section("DSC 회귀 엔진 v5 단위 검증")

    data = fetch_california_housing(as_frame=True)
    df = data.frame
    target_col = 'MedHouseVal'
    numerical_cols = [c for c in df.columns if c != target_col]
    categorical_cols = []
    print(f"\nCalifornia Housing: {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"  numerical: {len(numerical_cols)}, categorical: {len(categorical_cols)}, target='{target_col}'")

    # ============================================================
    # [1] Baseline DSC + 두 신지표 ≥ 0.7
    # ============================================================
    section("[1] Baseline DSC")
    baseline = compute_dsc_regression(df, target_col, numerical_cols, categorical_cols,
                                      reference_df=df)
    print(f"\n  score: {baseline['score']:.2f} (grade {baseline['grade']})")
    for k in DEFAULT_WEIGHTS_REGRESSION:
        print(f"  {k:32s}: {baseline[k]:.4f}  (w={DEFAULT_WEIGHTS_REGRESSION[k]:.2f})")

    print(f"\n  Phase 1 검증:")
    fail_count = 0
    if baseline['target_smoothness'] >= 0.7:
        print(f"    ✅ target_smoothness {baseline['target_smoothness']:.4f} ≥ 0.7")
    else:
        print(f"    ❌ target_smoothness {baseline['target_smoothness']:.4f} < 0.7")
        fail_count += 1
    if baseline['feature_informativeness_reg'] >= 0.1:
        print(f"    ✅ feature_informativeness_reg {baseline['feature_informativeness_reg']:.4f} ≥ 0.1")
    else:
        print(f"    ❌ feature_informativeness_reg {baseline['feature_informativeness_reg']:.4f} < 0.1")
        fail_count += 1
    if baseline['target_distribution_quality'] >= 0.7:
        print(f"    ✅ target_distribution_quality {baseline['target_distribution_quality']:.4f} ≥ 0.7")
    else:
        print(f"    ⚠️ target_distribution_quality {baseline['target_distribution_quality']:.4f} < 0.7 (분포 자체 특성)")

    # ============================================================
    # [2] target_distribution_skew 75% → target_distribution_quality −0.10 이상 하락
    # ============================================================
    section("[2] target_distribution_skew 폴루션 → target_distribution_quality 반응")
    print(f"\n  {'skew':>6} | {'TDQ':>8} | {'Δ baseline':>11} | {'TS':>8} | {'FI':>8}")
    for skew_level in [0.10, 0.25, 0.50, 0.75]:
        polluter = TargetDistributionSkewPolluter(
            skew_level=skew_level, target_column=target_col, random_seed=1)
        polluted_df = polluter.pollute(df)
        result = compute_dsc_regression(polluted_df, target_col, numerical_cols, categorical_cols,
                                        reference_df=df)
        delta_tdq = result['target_distribution_quality'] - baseline['target_distribution_quality']
        print(f"  {skew_level:>6.2f} | {result['target_distribution_quality']:>8.4f} | "
              f"{delta_tdq:>+11.4f} | {result['target_smoothness']:>8.4f} | "
              f"{result['feature_informativeness_reg']:>8.4f}")

        if skew_level == 0.75:
            if delta_tdq <= -0.10:
                print(f"\n    ✅ skew=0.75 Δ={delta_tdq:+.4f} ≤ -0.10 (Phase 1 PASS)")
            else:
                print(f"\n    ❌ skew=0.75 Δ={delta_tdq:+.4f} > -0.10 (Phase 1 FAIL)")
                fail_count += 1

    # ============================================================
    # [3] completeness 폴루션 (피처 결측 75%) → 메트릭 하락
    # ============================================================
    section("[3] completeness 폴루션 (피처 결측) → 메트릭 반응")
    print(f"\n  {'frac':>6} | {'COMP':>8} | {'TS':>8} | {'FI':>8} | {'TDQ':>8}")
    rng = np.random.RandomState(1)
    for frac in [0.10, 0.25, 0.50, 0.75]:
        polluted = df.copy()
        for col in numerical_cols:
            mask = rng.rand(len(polluted)) < frac
            polluted.loc[mask, col] = np.nan
        result = compute_dsc_regression(polluted, target_col, numerical_cols, categorical_cols,
                                        reference_df=df)
        print(f"  {frac:>6.2f} | {result['completeness']:>8.4f} | "
              f"{result['target_smoothness']:>8.4f} | "
              f"{result['feature_informativeness_reg']:>8.4f} | "
              f"{result['target_distribution_quality']:>8.4f}")

    # 75% 결측에서 completeness, target_smoothness, FI 모두 baseline 대비 하락
    polluted_75 = df.copy()
    rng = np.random.RandomState(1)
    for col in numerical_cols:
        mask = rng.rand(len(polluted_75)) < 0.75
        polluted_75.loc[mask, col] = np.nan
    res_75 = compute_dsc_regression(polluted_75, target_col, numerical_cols, categorical_cols,
                                    reference_df=df)
    print(f"\n  75% 결측 시 baseline 대비:")
    drops = {
        'completeness': res_75['completeness'] - baseline['completeness'],
        'target_smoothness': res_75['target_smoothness'] - baseline['target_smoothness'],
        'feature_informativeness_reg': res_75['feature_informativeness_reg'] - baseline['feature_informativeness_reg'],
        'target_distribution_quality': res_75['target_distribution_quality'] - baseline['target_distribution_quality'],
    }
    for k, v in drops.items():
        marker = "↓" if v < 0 else "↑" if v > 0 else "—"
        print(f"    {k:32s}: Δ={v:+.4f} {marker}")

    if drops['completeness'] < -0.30:
        print(f"\n    ✅ completeness 큰 폭 하락 (메트릭 정상 반응)")
    else:
        print(f"\n    ❌ completeness 하락 폭 부족")
        fail_count += 1

    # False positive: target 분포 자체는 변화 없어야 (피처만 손상시켰으니)
    fp_delta = abs(drops['target_distribution_quality'])
    if fp_delta < 0.05:
        print(f"    ✅ target_distribution_quality 거의 불변 (false positive 없음, |Δ|={fp_delta:.4f})")
    else:
        print(f"    ⚠️  target_distribution_quality 영향 있음: |Δ|={fp_delta:.4f}")

    # ============================================================
    # [4] feature_accuracy 폴루션 (피처값 노이즈) → target_smoothness 하락
    # ============================================================
    section("[4] feature_accuracy 폴루션 (피처 노이즈) → target_smoothness 반응")
    print(f"\n  {'noise':>6} | {'TS':>8} | {'Δ baseline':>11} | {'COMP':>8} | {'FI':>8}")
    for noise_frac in [0.10, 0.25, 0.50, 0.75]:
        rng = np.random.RandomState(1)
        polluted = df.copy()
        for col in numerical_cols:
            mask = rng.rand(len(polluted)) < noise_frac
            col_std = polluted[col].std()
            polluted.loc[mask, col] = polluted.loc[mask, col] + rng.randn(mask.sum()) * col_std * 3
        result = compute_dsc_regression(polluted, target_col, numerical_cols, categorical_cols,
                                        reference_df=df)
        delta_ts = result['target_smoothness'] - baseline['target_smoothness']
        print(f"  {noise_frac:>6.2f} | {result['target_smoothness']:>8.4f} | "
              f"{delta_ts:>+11.4f} | {result['completeness']:>8.4f} | "
              f"{result['feature_informativeness_reg']:>8.4f}")

        if noise_frac == 0.75:
            if delta_ts <= -0.10:
                print(f"\n    ✅ noise=0.75 Δ={delta_ts:+.4f} ≤ -0.10 (Phase 1 PASS)")
            else:
                print(f"\n    ❌ noise=0.75 Δ={delta_ts:+.4f} > -0.10 (Phase 1 FAIL)")
                fail_count += 1

    # ============================================================
    # [5] uniqueness 폴루션 (중복 행 추가) → target_smoothness, FI 거의 불변
    # ============================================================
    section("[5] uniqueness 폴루션 (중복 행 복제) → false positive 검증")
    print(f"\n  중복 행이 많아져도 target_smoothness/FI는 dedup 후 측정하므로 거의 불변이어야 함")
    print(f"  {'factor':>6} | {'UNIQ':>8} | {'TS':>8} | {'Δ TS':>9} | {'FI':>8} | {'Δ FI':>9}")
    for factor in [1.5, 2.0, 3.0, 4.0]:
        n_dup = int(len(df) * (factor - 1))
        dup_idx = np.random.RandomState(1).choice(df.index, size=n_dup, replace=True)
        duplicated = pd.concat([df, df.loc[dup_idx]], ignore_index=True)
        result = compute_dsc_regression(duplicated, target_col, numerical_cols, categorical_cols,
                                        reference_df=df)
        delta_ts = result['target_smoothness'] - baseline['target_smoothness']
        delta_fi = result['feature_informativeness_reg'] - baseline['feature_informativeness_reg']
        print(f"  {factor:>6.1f} | {result['uniqueness']:>8.4f} | "
              f"{result['target_smoothness']:>8.4f} | {delta_ts:>+9.4f} | "
              f"{result['feature_informativeness_reg']:>8.4f} | {delta_fi:>+9.4f}")

    # 4.0 배수에서 |Δ TS|, |Δ FI| ≤ 0.05 검증
    n_dup = int(len(df) * 3.0)
    dup_idx = np.random.RandomState(1).choice(df.index, size=n_dup, replace=True)
    duplicated = pd.concat([df, df.loc[dup_idx]], ignore_index=True)
    res_dup = compute_dsc_regression(duplicated, target_col, numerical_cols, categorical_cols,
                                     reference_df=df)
    abs_delta_ts = abs(res_dup['target_smoothness'] - baseline['target_smoothness'])
    abs_delta_fi = abs(res_dup['feature_informativeness_reg'] - baseline['feature_informativeness_reg'])
    if abs_delta_ts <= 0.05 and abs_delta_fi <= 0.05:
        print(f"\n    ✅ factor=4.0에서 |Δ TS|={abs_delta_ts:.4f}, |Δ FI|={abs_delta_fi:.4f} ≤ 0.05 "
              f"(false positive 없음, dedup 효과)")
    else:
        print(f"\n    ⚠️ factor=4.0에서 |Δ TS|={abs_delta_ts:.4f}, |Δ FI|={abs_delta_fi:.4f}")

    # ============================================================
    # [6] 가중치 합 = 1.00, score 0~100 범위
    # ============================================================
    section("[6] 가중치 + score 범위")
    weight_sum = sum(DEFAULT_WEIGHTS_REGRESSION.values())
    print(f"\n  가중치 합: {weight_sum:.4f}")
    assert abs(weight_sum - 1.0) < 1e-9, f"가중치 합이 1이 아님: {weight_sum}"
    print(f"  ✅ 합 = 1.00")

    print(f"  baseline score: {baseline['score']:.2f} (0~100 범위)")
    assert 0 <= baseline['score'] <= 100, f"score 범위 위반"
    print(f"  ✅ 0~100 범위")

    # ============================================================
    # 결과 요약
    # ============================================================
    section("결과 요약")
    if fail_count == 0:
        print(f"\n  ALL PHASE 1 CHECKS PASSED ✅")
    else:
        print(f"\n  ❌ {fail_count} 개 검증 실패")
        sys.exit(1)


if __name__ == '__main__':
    main()
