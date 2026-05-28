"""Text polluters for v5 text cell (ADR-016 + ADR-017 사전등록).

**Pandas 2.x 호환 shim**: dq4ai/polluters/classbalance.py의 일부 코드 경로가
DataFrame.append (pandas 2.0에서 제거됨)를 사용. dq4ai/는 본 repo에서 gitignored
+ 별도 maintainer이므로 본 모듈 import 시 DataFrame.append를 폴리필. shim은
hasattr 가드로 idempotent + pandas <2.0에서는 no-op.

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
# --- Pandas 2.x compat shim (dq4ai 의존) ---
import pandas as _pd

if not hasattr(_pd.DataFrame, 'append'):
    def _df_append_shim(self, other, ignore_index=False, **_kwargs):
        """dq4ai legacy 호환용 append. pd.concat으로 위임."""
        if not isinstance(other, (list, tuple)):
            other = [other]
        return _pd.concat([self, *other], ignore_index=ignore_index)
    _pd.DataFrame.append = _df_append_shim  # type: ignore[attr-defined]
# --- /shim ---

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
