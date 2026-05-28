"""NoiseInjectionTextPolluter — character-level typo 주입.

ADR-016 §3-3 사전등록.
level ∈ [0, 1] = 문자별 노이즈 적용 확률.

각 문자마다 독립적으로 (확률 level)으로 노이즈 트리거 → 4가지 연산 균등 샘플:
1. delete (문자 삭제)
2. insert (alpha 문자 1개 삽입)
3. swap (인접 문자와 위치 교환)
4. substitute (다른 alpha 문자로 치환)

이미지 cell의 noise_injection (Gaussian noise on pixel)에 대응.
공백·`[MASK]`·`[PAD]` 토큰은 보호.
"""
import string

import numpy as np


PROTECTED_TOKENS = ('[MASK]', '[PAD]', '[CLS]', '[SEP]', '[UNK]')


class NoiseInjectionTextPolluter:
    """문자 단위 typo 주입.

    Args:
        level: ∈ [0, 1] — 문자별 노이즈 적용 확률
        random_seed: 재현성
    """

    def __init__(self, level, random_seed=42):
        if not 0.0 <= float(level) <= 1.0:
            raise ValueError(f"level must be in [0, 1], got {level}")
        self.level = float(level)
        self.random_seed = int(random_seed)

    def _noise_token(self, token, rng):
        """단일 토큰에 char-level 노이즈 적용. 보호 토큰은 그대로."""
        if token in PROTECTED_TOKENS or not token:
            return token
        chars = list(token)
        out = []
        i = 0
        while i < len(chars):
            c = chars[i]
            if rng.random() < self.level:
                op = rng.randint(0, 4)
                if op == 0:
                    # delete
                    i += 1
                    continue
                elif op == 1:
                    # insert random alpha before
                    insert_c = string.ascii_lowercase[rng.randint(0, 26)]
                    out.append(insert_c)
                    out.append(c)
                elif op == 2 and i + 1 < len(chars):
                    # swap with next
                    out.append(chars[i + 1])
                    out.append(c)
                    i += 2
                    continue
                else:
                    # substitute
                    sub_c = string.ascii_lowercase[rng.randint(0, 26)]
                    out.append(sub_c)
            else:
                out.append(c)
            i += 1
        return ''.join(out)

    def pollute(self, texts, labels):
        rng = np.random.RandomState(self.random_seed)
        out_texts = []
        for t in texts:
            tokens = str(t).split()
            tokens_out = [self._noise_token(tok, rng) for tok in tokens]
            tokens_out = [tok for tok in tokens_out if tok]
            out_texts.append(' '.join(tokens_out) if tokens_out else '')
        return out_texts, list(labels)
