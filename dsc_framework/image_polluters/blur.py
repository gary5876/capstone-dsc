"""BlurPolluter — Gaussian blur 적용.

level ∈ [0, 1] → sigma = level * max_sigma (default max_sigma=5.0).
선택 비율은 모든 이미지(level이 강도 자체).
"""
import numpy as np

from ..image_cell import _to_np_uint8


class BlurPolluter:
    """Gaussian blur (각 이미지에 동일 sigma 적용).

    Args:
        level: ∈ [0, 1]
        max_sigma: level=1.0일 때의 sigma (default 5.0 — 32×32 이미지엔 매우 강한 blur)
        random_seed: 재현성 (현재 사용 안 함)
    """

    def __init__(self, level, max_sigma=5.0, random_seed=42):
        self.level = float(level)
        self.max_sigma = float(max_sigma)
        self.random_seed = random_seed

    def pollute(self, images, labels):
        sigma = self.level * self.max_sigma
        if sigma == 0:
            return [_to_np_uint8(img).copy() for img in images], list(labels)

        # cv2 우선 (빠름), 없으면 scipy
        try:
            import cv2
            ksize = int(2 * round(3 * sigma) + 1)  # ~3 sigma 커버
            ksize = max(3, ksize)
            out_images = []
            for img in images:
                arr = _to_np_uint8(img)
                if arr.ndim == 3 and arr.shape[2] == 1:
                    arr = arr.squeeze(-1)
                blurred = cv2.GaussianBlur(arr, (ksize, ksize), sigma)
                if blurred.ndim == 2:
                    blurred = blurred[:, :, None]
                out_images.append(blurred)
            return out_images, list(labels)
        except ImportError:
            from scipy.ndimage import gaussian_filter
            out_images = []
            for img in images:
                arr = _to_np_uint8(img).astype(np.float32)
                if arr.ndim == 3:
                    blurred = np.stack([gaussian_filter(arr[..., c], sigma) for c in range(arr.shape[-1])], axis=-1)
                else:
                    blurred = gaussian_filter(arr, sigma)
                out_images.append(np.clip(blurred, 0, 255).astype(np.uint8))
            return out_images, list(labels)
