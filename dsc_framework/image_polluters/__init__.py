"""Image polluters for v5 image cell (ADR-014 사전등록).

5종 polluter:
- CompletenessImagePolluter — 픽셀 마스킹
- NoiseInjectionPolluter    — Gaussian noise
- BlurPolluter              — Gaussian blur
- ClassBalanceImagePolluter — class undersampling
- LabelSwapPolluter         — label noise

공통 인터페이스:
    polluter.pollute(images, labels) -> (images_polluted, labels_polluted)

`level` ∈ [0, 1] 의미는 polluter별로 다름 (각 클래스 docstring 참조).
"""
from .blur import BlurPolluter
from .class_balance_image import ClassBalanceImagePolluter
from .completeness_image import CompletenessImagePolluter
from .label_swap import LabelSwapPolluter
from .noise_injection import NoiseInjectionPolluter

__all__ = [
    'CompletenessImagePolluter',
    'NoiseInjectionPolluter',
    'BlurPolluter',
    'ClassBalanceImagePolluter',
    'LabelSwapPolluter',
]
