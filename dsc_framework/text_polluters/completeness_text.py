"""CompletenessTextPolluter — 단어 일부를 [MASK] 토큰으로 치환.

ADR-016 §3-3 사전등록.
level ∈ [0, 1] = 단어 단위 마스킹 비율 (whitespace tokenize 기준).

이미지 cell의 completeness_image (픽셀 → mask_value)에 대응.
"""
import numpy as np


MASK_TOKEN = '[MASK]'


class CompletenessTextPolluter:
    """단어 단위 마스킹.

    Args:
        level: ∈ [0, 1] — 마스킹할 단어 비율
        random_seed: 재현성
        mask_token: 치환 토큰 (default '[MASK]', DistilBERT tokenizer 호환)
    """

    def __init__(self, level, random_seed=42, mask_token=MASK_TOKEN):
        if not 0.0 <= float(level) <= 1.0:
            raise ValueError(f"level must be in [0, 1], got {level}")
        self.level = float(level)
        self.random_seed = int(random_seed)
        self.mask_token = mask_token

    def _mask_one(self, text, rng):
        tokens = text.split()
        if not tokens:
            return text
        n_mask = int(round(self.level * len(tokens)))
        if n_mask <= 0:
            return text
        idx = rng.choice(len(tokens), size=n_mask, replace=False)
        for i in idx:
            tokens[i] = self.mask_token
        return ' '.join(tokens)

    def pollute(self, texts, labels):
        rng = np.random.RandomState(self.random_seed)
        out_texts = [self._mask_one(str(t), rng) for t in texts]
        return out_texts, list(labels)
