"""Image polluters for v5 image cell.

분류(ADR-014) 5종:
- CompletenessImagePolluter — 픽셀 마스킹
- NoiseInjectionPolluter    — Gaussian noise
- BlurPolluter              — Gaussian blur
- ClassBalanceImagePolluter — class undersampling
- LabelSwapPolluter         — label noise

회귀(ADR-018) 전용 2종 (분류의 class_balance·label_swap 대체):
- TargetDistributionSkewImagePolluter — target 분포 편향
- TargetNoiseImagePolluter            — target Gaussian noise

공통 인터페이스:
    분류: polluter.pollute(images, labels)  -> (images_polluted, labels_polluted)
    회귀: polluter.pollute(images, targets) -> (images_polluted, targets_polluted)

`level` ∈ [0, 1] 의미는 polluter별로 다름 (각 클래스 docstring 참조).
"""
from .blur import BlurPolluter
from .class_balance_image import ClassBalanceImagePolluter
from .completeness_image import CompletenessImagePolluter
from .label_swap import LabelSwapPolluter
from .noise_injection import NoiseInjectionPolluter
from .target_distribution_skew import TargetDistributionSkewImagePolluter
from .target_noise import TargetNoiseImagePolluter

__all__ = [
    'CompletenessImagePolluter',
    'NoiseInjectionPolluter',
    'BlurPolluter',
    'ClassBalanceImagePolluter',
    'LabelSwapPolluter',
    'TargetDistributionSkewImagePolluter',
    'TargetNoiseImagePolluter',
]
