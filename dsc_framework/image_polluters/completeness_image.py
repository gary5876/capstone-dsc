"""CompletenessImagePolluter — 이미지 픽셀 일부를 배경색(black)으로 마스킹.

level = 마스킹할 이미지 비율. 선택된 이미지의 마스킹 영역은 random rectangle (이미지 면적의 30% 고정).
"""
import numpy as np

from ..image_cell import _to_np_uint8


class CompletenessImagePolluter:
    """픽셀 마스킹 polluter.

    Args:
        level: ∈ [0, 1] — 마스킹 적용할 이미지 비율
        mask_value: 마스킹 픽셀 값 (default 0 = black)
        mask_area_ratio: 각 이미지에서 마스킹 면적 비율 (default 0.3)
        random_seed: 재현성
    """

    def __init__(self, level, mask_value=0, mask_area_ratio=0.3, random_seed=42):
        self.level = float(level)
        self.mask_value = mask_value
        self.mask_area_ratio = mask_area_ratio
        self.random_seed = random_seed

    def pollute(self, images, labels):
        rng = np.random.RandomState(self.random_seed)
        n = len(images)
        n_target = int(self.level * n)
        target_idx = set(rng.choice(n, size=n_target, replace=False).tolist()) if n_target > 0 else set()

        out_images = []
        for i, img in enumerate(images):
            arr = _to_np_uint8(img).copy()
            if i in target_idx:
                h, w = arr.shape[:2]
                # 면적 mask_area_ratio 비율의 random rectangle
                mask_h = max(1, int(h * np.sqrt(self.mask_area_ratio)))
                mask_w = max(1, int(w * np.sqrt(self.mask_area_ratio)))
                y0 = rng.randint(0, h - mask_h + 1) if h > mask_h else 0
                x0 = rng.randint(0, w - mask_w + 1) if w > mask_w else 0
                arr[y0:y0 + mask_h, x0:x0 + mask_w] = self.mask_value
            out_images.append(arr)

        return out_images, list(labels)
