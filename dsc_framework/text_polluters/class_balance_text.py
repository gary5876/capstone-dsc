"""ClassBalanceTextPolluter — dq4ai.ClassBalancePolluter wrapper.

ADR-016 §3-3 — 라벨 조작은 데이터 유형 무관이라 dq4ai 로직 재사용.
(texts, labels) → DataFrame({'text_idx', 'label'}) → ClassBalancePolluter → (texts_out, labels_out).

level ∈ [0, 1] = imbalance_level (0=완전 균형, 1=극단 불균형).
"""
from __future__ import annotations

import pandas as pd

# 프로젝트 루트(dsc/)가 PYTHONPATH에 있다고 가정
from dq4ai.polluters import ClassBalancePolluter


class ClassBalanceTextPolluter:
    """클래스 불균형 주입 (dq4ai wrapper).

    Args:
        level: ∈ [0, 1] — imbalance_level
        random_seed: 재현성
    """

    def __init__(self, level, random_seed=42):
        if not 0.0 <= float(level) <= 1.0:
            raise ValueError(f"level must be in [0, 1], got {level}")
        self.level = float(level)
        self.random_seed = int(random_seed)

    def pollute(self, texts, labels):
        n = len(texts)
        df = pd.DataFrame({
            'text_idx': range(n),
            'label': list(labels),
        })
        polluter = ClassBalancePolluter(
            imbalance_level=self.level,
            target_column='label',
            n_samples=n,
            random_seed=self.random_seed,
        )
        polluted_df = polluter.pollute(df)
        out_indices = polluted_df['text_idx'].astype(int).tolist()
        texts_out = [texts[i] for i in out_indices]
        labels_out = polluted_df['label'].tolist()
        return texts_out, labels_out
