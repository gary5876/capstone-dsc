"""LabelSwapPolluter — label noise 주입.

level ∈ [0, 1] → label을 무작위 다른 클래스로 변경할 비율.
"""
import numpy as np


class LabelSwapPolluter:
    """Label noise.

    Args:
        level: ∈ [0, 1] — swap할 비율
        random_seed: 재현성
    """

    def __init__(self, level, random_seed=42):
        self.level = float(level)
        self.random_seed = random_seed

    def pollute(self, images, labels):
        rng = np.random.RandomState(self.random_seed)
        labels_arr = np.asarray(labels).copy()
        classes = np.unique(labels_arr)
        n = len(labels_arr)

        if len(classes) <= 1:
            return list(images), list(labels)

        n_swap = int(self.level * n)
        swap_idx = rng.choice(n, size=n_swap, replace=False)

        for i in swap_idx:
            old = labels_arr[i]
            choices = classes[classes != old]
            labels_arr[i] = rng.choice(choices)

        return list(images), labels_arr.tolist()
