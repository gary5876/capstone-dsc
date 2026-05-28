"""WordShufflePolluter — 어순 무작위 셔플.

ADR-016 §3-3 사전등록.
level ∈ [0, 1] = shuffle 강도:
- 0.0 = 원본 어순 보존
- 1.0 = 완전 셔플 (np.random.permutation)
- 중간값 = 부분 셔플 (level 비율의 단어 위치 쌍을 swap)

이미지 cell의 blur (Gaussian blur, spatial 정보 손상)에 대응 —
텍스트는 sequential 정보 손상으로 BoW 모델은 영향 적고 transformer는 큰 영향.
"""
import numpy as np


class WordShufflePolluter:
    """단어 어순 셔플.

    Args:
        level: ∈ [0, 1] — shuffle 강도
        random_seed: 재현성
    """

    def __init__(self, level, random_seed=42):
        if not 0.0 <= float(level) <= 1.0:
            raise ValueError(f"level must be in [0, 1], got {level}")
        self.level = float(level)
        self.random_seed = int(random_seed)

    def _shuffle_one(self, text, rng):
        tokens = text.split()
        n = len(tokens)
        if n <= 1:
            return text
        if self.level >= 0.999:
            perm = rng.permutation(n)
            return ' '.join(tokens[i] for i in perm)
        n_swaps = int(round(self.level * n))
        if n_swaps <= 0:
            return text
        for _ in range(n_swaps):
            i, j = rng.randint(0, n), rng.randint(0, n)
            tokens[i], tokens[j] = tokens[j], tokens[i]
        return ' '.join(tokens)

    def pollute(self, texts, labels):
        rng = np.random.RandomState(self.random_seed)
        out_texts = [self._shuffle_one(str(t), rng) for t in texts]
        return out_texts, list(labels)
