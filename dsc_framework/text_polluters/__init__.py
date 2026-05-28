"""Text polluters for v5 text cell (ADR-016 + ADR-017 사전등록).

7종 polluter (분류 5 + 회귀 5, 본문 변형 3개 공유):

분류 트랙 (ADR-016):
- CompletenessTextPolluter   — 단어 → [MASK]              (신규)
- NoiseInjectionTextPolluter — character-level typo        (신규)
- WordShufflePolluter        — 어순 셔플                   (신규)
- ClassBalanceTextPolluter   — dq4ai.ClassBalancePolluter wrapper
- LabelSwapTextPolluter      — dq4ai.TargetAccuracyPolluter (분류 분기) wrapper

회귀 트랙 (ADR-017) 추가:
- TargetDistributionSkewTextPolluter — dq4ai.TargetDistributionSkewPolluter wrapper
- TargetNoiseTextPolluter            — dq4ai.TargetAccuracyPolluter (회귀 분기) wrapper

공통 인터페이스:
    polluter = Polluter(level, random_seed)
    texts_polluted, labels_polluted = polluter.pollute(texts, labels)

`level` ∈ [0, 1] 의미는 polluter별 docstring 참조.
"""
from .class_balance_text import ClassBalanceTextPolluter
from .completeness_text import CompletenessTextPolluter
from .label_swap_text import LabelSwapTextPolluter
from .noise_injection_text import NoiseInjectionTextPolluter
from .target_distribution_skew import TargetDistributionSkewTextPolluter
from .target_noise import TargetNoiseTextPolluter
from .word_shuffle import WordShufflePolluter

__all__ = [
    'CompletenessTextPolluter',
    'NoiseInjectionTextPolluter',
    'WordShufflePolluter',
    'ClassBalanceTextPolluter',
    'LabelSwapTextPolluter',
    'TargetDistributionSkewTextPolluter',
    'TargetNoiseTextPolluter',
]
