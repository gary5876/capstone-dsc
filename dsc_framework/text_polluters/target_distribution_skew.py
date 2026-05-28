"""TargetDistributionSkewTextPolluter — dq4ai.TargetDistributionSkewPolluter wrapper.

ADR-017 §3-3 — 회귀 target 분포 편향 (Q3 이상 일부 제거).
texts와 targets를 같은 인덱스로 묶어 dq4ai polluter에 넘김.

level ∈ [0, 1] = upper-quartile에서 제거할 비율.
"""
from __future__ import annotations

import pandas as pd

from dq4ai.polluters import TargetDistributionSkewPolluter


class TargetDistributionSkewTextPolluter:
    """회귀 target 편향 (dq4ai wrapper).

    Args:
        level: ∈ [0, 1] — skew_level
        random_seed: 재현성
        n_bins: target binning 수 (default 10, ADR-017과 동일)
    """

    def __init__(self, level, random_seed=42, n_bins=10):
        if not 0.0 <= float(level) <= 1.0:
            raise ValueError(f"level must be in [0, 1], got {level}")
        self.level = float(level)
        self.random_seed = int(random_seed)
        self.n_bins = int(n_bins)

    def pollute(self, texts, targets):
        n = len(texts)
        targets_float = pd.Series(list(targets), dtype=float)
        df = pd.DataFrame({
            'text_idx': range(n),
            'target': targets_float,
        })
        polluter = TargetDistributionSkewPolluter(
            skew_level=self.level,
            target_column='target',
            n_bins=self.n_bins,
            random_seed=self.random_seed,
        )
        polluted_df = polluter.pollute(df)
        out_indices = polluted_df['text_idx'].astype(int).tolist()
        texts_out = [texts[i] for i in out_indices]
        targets_out = polluted_df['target'].astype(float).tolist()
        return texts_out, targets_out
