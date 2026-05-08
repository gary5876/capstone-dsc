"""ClassBalanceImagePolluter — 클래스별 undersampling으로 imbalance 유발.

level ∈ [0, 1] → minority class 비율 = (1 - level) * (1/n_classes)
0=균형 유지, 1에 가까울수록 minority class 거의 사라짐.

tabular의 ClassBalancePolluter와 동일 정의식 (이미지 적용).
"""
from collections import defaultdict

import numpy as np


class ClassBalanceImagePolluter:
    """클래스 imbalance 유발.

    Args:
        level: ∈ [0, 1] — minority 클래스 축소 강도
        random_seed: 재현성
    """

    def __init__(self, level, random_seed=42):
        self.level = float(level)
        self.random_seed = random_seed

    def pollute(self, images, labels):
        rng = np.random.RandomState(self.random_seed)
        labels_arr = np.asarray(labels)
        classes, counts = np.unique(labels_arr, return_counts=True)
        n_classes = len(classes)
        if n_classes <= 1:
            return list(images), list(labels)

        # majority class 그대로, 나머지는 (1-level) 비율로 undersample
        max_count = counts.max()
        majority = classes[counts.argmax()]

        keep_idx = []
        for cls, count in zip(classes, counts):
            cls_idx = np.where(labels_arr == cls)[0]
            if cls == majority:
                keep_idx.extend(cls_idx.tolist())
            else:
                keep_n = max(1, int((1.0 - self.level) * count))
                sampled = rng.choice(cls_idx, size=keep_n, replace=False)
                keep_idx.extend(sampled.tolist())

        keep_idx = sorted(keep_idx)
        out_images = [images[i] for i in keep_idx]
        out_labels = [labels[i] for i in keep_idx]
        return out_images, out_labels
