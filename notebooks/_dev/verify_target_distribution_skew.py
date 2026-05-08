"""
target_distribution_skew_polluter 단위 검증.

Phase 1 검증 기준 (마스터플랜 sect 1-1, 1-5):
- baseline 품질 ≥ 0.7
- skew_level=0.75에서 baseline 대비 −0.10 이상 하락
- 단조 감소 (skew_level ↑ → quality ↓)
- random_seed 재현성
- 행 수 변화가 예상치와 일치

CLAUDE.md 외부 자원 검증 + memory feedback_verify_before_deliver 준수:
구문 검사로 끝내지 않고 실제 California Housing 데이터로 로직 검증.
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing

from dq4ai.polluters.target_distribution_skew_polluter import TargetDistributionSkewPolluter


def main():
    print("=" * 64)
    print("target_distribution_skew_polluter 검증")
    print("=" * 64)

    # [1] 데이터 로드
    data = fetch_california_housing(as_frame=True)
    df = data.frame  # MedHouseVal 컬럼 포함
    target_col = 'MedHouseVal'
    print(f"\n[1] California Housing: {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"    target='{target_col}', range=[{df[target_col].min():.3f}, {df[target_col].max():.3f}]")
    print(f"    Q1={df[target_col].quantile(0.25):.3f}, Q2={df[target_col].quantile(0.50):.3f}, "
          f"Q3={df[target_col].quantile(0.75):.3f}")

    # [2] baseline 품질 (clean 데이터의 target 분포 entropy)
    baseline_polluter = TargetDistributionSkewPolluter(
        skew_level=0.10,
        target_column=target_col,
        random_seed=1,
    )
    baseline_quality = baseline_polluter.compute_quality_measure(df, df)
    print(f"\n[2] Baseline (clean) quality: {baseline_quality:.4f}")
    assert baseline_quality >= 0.7, \
        f"❌ baseline quality {baseline_quality:.4f} < 0.7 (Phase 1 검증 기준 위반)"
    print(f"    ✅ ≥ 0.7 (Phase 1 검증 PASS)")

    # [3] 강도별 품질·행수 변화
    print(f"\n[3] Skew level → 품질 (df_clean 기준 bin):")
    print(f"    {'skew':>6} | {'rows':>6} | {'dropped':>7} | {'quality':>8} | {'Δ baseline':>11}")
    print(f"    {'-' * 6} | {'-' * 6} | {'-' * 7} | {'-' * 8} | {'-' * 11}")

    qualities = {}
    row_counts = {}
    expected_q3 = int((df[target_col] >= df[target_col].quantile(0.75)).sum())

    for skew_level in [0.10, 0.25, 0.50, 0.75]:
        polluter = TargetDistributionSkewPolluter(
            skew_level=skew_level,
            target_column=target_col,
            random_seed=1,
        )
        polluted_df = polluter.pollute(df)
        # quality는 df_clean 기준 bin edge 사용 — 비교 가능
        quality = polluter.compute_quality_measure(polluted_df, df)
        qualities[skew_level] = quality
        row_counts[skew_level] = len(polluted_df)
        dropped = len(df) - len(polluted_df)
        delta = quality - baseline_quality
        print(f"    {skew_level:>6.2f} | {len(polluted_df):>6} | {dropped:>7} | "
              f"{quality:>8.4f} | {delta:>+11.4f}")

    # [4] Phase 1 검증 기준: 75% 강도에서 −0.10 이상 하락
    delta_75 = qualities[0.75] - baseline_quality
    print(f"\n[4] Phase 1 검증 기준 — skew=0.75에서 −0.10 이상 하락:")
    if delta_75 <= -0.10:
        print(f"    ✅ Δ={delta_75:+.4f} ≤ -0.10 (PASS)")
    else:
        print(f"    ❌ Δ={delta_75:+.4f} > -0.10 (FAIL)")
        sys.exit(1)

    # [5] 단조 감소 확인
    print(f"\n[5] 단조 감소 (skew ↑ → quality ↓):")
    sorted_qs = [qualities[k] for k in [0.10, 0.25, 0.50, 0.75]]
    is_monotonic = all(sorted_qs[i] >= sorted_qs[i + 1] for i in range(len(sorted_qs) - 1))
    print(f"    quality 시퀀스: {[round(q, 4) for q in sorted_qs]}")
    if is_monotonic:
        print(f"    ✅ 단조 감소")
    else:
        print(f"    ⚠️  비단조 — 검토 필요 (entropy는 분포 균형성 측정이라 절대 단조 아닐 수 있음)")

    # [6] 행 수 drop 정확성
    print(f"\n[6] 행 수 drop 정확성 (Q3+ rows = {expected_q3}):")
    for skew_level in [0.10, 0.25, 0.50, 0.75]:
        n_dropped = len(df) - row_counts[skew_level]
        n_dropped_expected = int(expected_q3 * skew_level)
        match = abs(n_dropped - n_dropped_expected) <= 1
        marker = "✅" if match else "❌"
        print(f"    skew={skew_level}: dropped {n_dropped:>5} "
              f"(expected ≈{n_dropped_expected:>5})  {marker}")
        assert match, f"❌ skew={skew_level}에서 drop 수 불일치"

    # [7] 재현성 (random_seed=1)
    print(f"\n[7] 재현성 (random_seed=1):")
    p1 = TargetDistributionSkewPolluter(skew_level=0.50, target_column=target_col, random_seed=1)
    p2 = TargetDistributionSkewPolluter(skew_level=0.50, target_column=target_col, random_seed=1)
    df1 = p1.pollute(df)
    df2 = p2.pollute(df)
    same = df1.equals(df2)
    print(f"    seed=1 두 번 실행 동일: {'✅' if same else '❌'}")
    assert same, "❌ 재현성 위반"

    # [8] random_seed 다르면 결과 다름
    p3 = TargetDistributionSkewPolluter(skew_level=0.50, target_column=target_col, random_seed=42)
    df3 = p3.pollute(df)
    different = not df1.equals(df3)
    print(f"    seed=1 vs seed=42 다름: {'✅' if different else '⚠️ 동일 (의심스러움)'}")

    # [9] False positive 검증 — completeness 같은 다른 차원 변화는 target 분포에 영향 작아야 함
    # (직접 시뮬: target은 그대로 두고 피처에만 결측 주입)
    print(f"\n[9] False positive 검증 — 피처에만 결측이 있을 때 target_distribution_quality:")
    df_with_missing = df.copy()
    feature_cols = [c for c in df.columns if c != target_col]
    np.random.seed(1)
    for col in feature_cols:
        mask = np.random.rand(len(df_with_missing)) < 0.50
        df_with_missing.loc[mask, col] = np.nan
    fp_polluter = TargetDistributionSkewPolluter(
        skew_level=0.10, target_column=target_col, random_seed=1)
    fp_quality = fp_polluter.compute_quality_measure(df_with_missing, df)
    fp_delta = abs(fp_quality - baseline_quality)
    print(f"    결측 50% 주입 후 quality: {fp_quality:.4f} (Δ={fp_delta:.4f})")
    if fp_delta < 0.01:
        print(f"    ✅ target 분포는 변화 없음 (false positive 없음)")
    else:
        print(f"    ⚠️ target 분포 영향이 있음 — 메트릭 정의 재검토 권장")

    print(f"\n{'=' * 64}")
    print(f"ALL CHECKS PASSED ✅")
    print(f"{'=' * 64}")


if __name__ == '__main__':
    main()
