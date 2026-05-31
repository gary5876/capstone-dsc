"""TargetDistributionSkewImagePolluter — dq4ai.TargetDistributionSkewPolluter wrapper.

ADR-018 §3-3 — 회귀 target 분포 편향 (Q3 이상 일부 제거).
images와 targets를 같은 인덱스로 묶어 dq4ai polluter에 넘김. 이미지 본문은 불변,
선택된 인덱스만 남김 (text cell의 target_distribution_skew와 동일 패턴).

level ∈ [0, 1] = upper-quartile에서 제거할 비율.
"""
from __future__ import annotations

import pandas as pd

from dq4ai.polluters import TargetDistributionSkewPolluter


class TargetDistributionSkewImagePolluter:
    """회귀 target 편향 (dq4ai wrapper).

    Args:
        level: ∈ [0, 1] — skew_level
        random_seed: 재현성
        n_bins: target binning 수 (default 10, ADR-018과 동일)
    """

    def __init__(self, level, random_seed=42, n_bins=10):
        if not 0.0 <= float(level) <= 1.0:
            raise ValueError(f"level must be in [0, 1], got {level}")
        self.level = float(level)
        self.random_seed = int(random_seed)
        self.n_bins = int(n_bins)

    def pollute(self, images, targets):
        n = len(images)
        targets_float = pd.Series(list(targets), dtype=float)
        df = pd.DataFrame({
            'image_idx': range(n),
            'target': targets_float,
        })
        polluter = TargetDistributionSkewPolluter(
            skew_level=self.level,
            target_column='target',
            n_bins=self.n_bins,
            random_seed=self.random_seed,
        )
        polluted_df = polluter.pollute(df)
        out_indices = polluted_df['image_idx'].astype(int).tolist()
        images_out = [images[i] for i in out_indices]
        targets_out = polluted_df['target'].astype(float).tolist()
        return images_out, targets_out
