"""NoiseInjectionPolluter — Gaussian noise를 픽셀에 추가.

level = noise std / image_std (선택 이미지에 적용할 noise 강도).
선택 비율은 항상 1.0 (모든 이미지에 적용, level이 강도 자체).
"""
import numpy as np

from ..image_cell import _to_np_uint8


class NoiseInjectionPolluter:
    """Gaussian noise injection.

    Args:
        level: ∈ [0, 1] — 정규화된 noise std. 0=노이즈 없음, 1=image_std 만큼 노이즈
        random_seed: 재현성
    """

    def __init__(self, level, random_seed=42):
        self.level = float(level)
        self.random_seed = random_seed

    def pollute(self, images, labels):
        rng = np.random.RandomState(self.random_seed)
        out_images = []
        for img in images:
            arr = _to_np_uint8(img).astype(np.float32)
            std = arr.std()
            noise_std = self.level * std
            noise = rng.normal(0, noise_std, arr.shape)
            polluted = np.clip(arr + noise, 0, 255).astype(np.uint8)
            out_images.append(polluted)

        return out_images, list(labels)
