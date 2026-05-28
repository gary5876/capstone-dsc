"""LabelSwapTextPolluter — dq4ai.TargetAccuracyPolluter (분류 분기) wrapper.

ADR-016 §3-3 — 라벨 swap은 데이터 유형 무관이라 dq4ai 로직 재사용.
texts는 그대로, labels만 무작위 다른 클래스로 변경.

level ∈ [0, 1] = swap 비율.
"""
from __future__ import annotations

import pandas as pd

from dq4ai.polluters import TargetAccuracyPolluter


class LabelSwapTextPolluter:
    """label noise 주입 (dq4ai wrapper, 분류 전용).

    Args:
        level: ∈ [0, 1] — swap할 비율
        random_seed: 재현성
    """

    def __init__(self, level, random_seed=42):
        if not 0.0 <= float(level) <= 1.0:
            raise ValueError(f"level must be in [0, 1], got {level}")
        self.level = float(level)
        self.random_seed = int(random_seed)

    def pollute(self, texts, labels):
        labels_series = pd.Series(list(labels))
        # int dtype 보장 — TargetAccuracyPolluter는 float이 아니면 categorical 처리
        if labels_series.dtype.kind == 'f':
            labels_series = labels_series.astype(int)
        df = pd.DataFrame({'label': labels_series})
        polluter = TargetAccuracyPolluter(
            pollution_level=self.level,
            target_col='label',
            is_categorical=True,
            random_seed=self.random_seed,
        )
        polluted_df = polluter.pollute(df)
        labels_out = polluted_df['label'].astype(labels_series.dtype).tolist()
        return list(texts), labels_out
