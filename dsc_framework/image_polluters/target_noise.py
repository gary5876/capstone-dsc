"""TargetNoiseImagePolluter — dq4ai.TargetAccuracyPolluter (회귀 분기) wrapper.

ADR-018 §3-3 — 회귀 target에 Gaussian noise 추가.
images는 그대로, targets만 변경 (text cell의 target_noise와 동일 패턴).

level ∈ [0, 1] = pollution_level (= noise std factor, dq4ai 정의).
"""
from __future__ import annotations

import pandas as pd

from dq4ai.polluters import TargetAccuracyPolluter


class TargetNoiseImagePolluter:
    """회귀 target에 noise 주입 (dq4ai wrapper).

    Args:
        level: ∈ [0, 1] — pollution_level (= noise std factor)
        random_seed: 재현성
    """

    def __init__(self, level, random_seed=42):
        if not 0.0 <= float(level) <= 1.0:
            raise ValueError(f"level must be in [0, 1], got {level}")
        self.level = float(level)
        self.random_seed = int(random_seed)

    def pollute(self, images, targets):
        targets_float = pd.Series(list(targets), dtype=float)
        df = pd.DataFrame({'target': targets_float})
        polluter = TargetAccuracyPolluter(
            pollution_level=self.level,
            target_col='target',
            is_categorical=False,
            random_seed=self.random_seed,
        )
        polluted_df = polluter.pollute(df)
        targets_out = polluted_df['target'].astype(float).tolist()
        return list(images), targets_out
