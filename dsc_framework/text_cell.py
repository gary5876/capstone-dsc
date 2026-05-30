"""DSC v5 텍스트 cell — text × classification.

ADR-016 사전등록 (10개 메트릭, fallback 가중치 합 1.00):
- completeness_text   ([MASK]/[PAD]/빈 토큰 비율 보수)         : 0.15
- uniqueness          (정규화 hash 중복 비율 보수)              : 0.10
- validity            (UTF-8 + 비empty + token≥1 비율)         : 0.05
- consistency         (token count 5-bucket entropy 보수)      : 0.05
- outlier_ratio       (token count IQR outlier 보수)           : 0.05
- class_balance       (tabular와 동일 정의식)                    : 0.10
- feature_correlation (DistilBERT embedding 768-d cosine 상관)  : 0.05
- label_consistency   (k-NN embedding 라벨 일관성, chance 보정) : 0.20
- feature_informativeness (embedding → label MI / H(Y))         : 0.10
- sample_quality_text (TTR + length_adequacy 결합, 신설)        : 0.15

ADR-011 강한 버전 원칙: 차원 이름이 같아도 정의식이 cell마다 다름.
예) feature_correlation: tabular는 컬럼 Pearson, image는 ResNet embedding cosine,
    text는 DistilBERT mean-pool embedding cosine.

ADR-015: 본 모듈의 DEFAULT_WEIGHTS_TEXT는 LLM weight generator의 fallback 용도.
운영·검증 시 LLM 출력 가중치를 우선 사용.

입력 형식:
    texts: list[str] (또는 pd.Series of str)
    labels: list/np.ndarray of int (분류 라벨)

DistilBERT는 lazy import — torch/transformers 미설치 환경에선
use_embeddings=False로 폴백 (3개 embedding 메트릭 1.0 처리).
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from math import log

import numpy as np

from .shared_metrics import to_grade


# =================================================================
# 입력 정규화
# =================================================================

_WS_RE = re.compile(r'\s+')
_PROTECTED = {'[MASK]', '[PAD]', '[CLS]', '[SEP]', '[UNK]'}


def _ensure_text_list(texts):
    """list[str]로 정규화. pandas Series나 numpy array도 허용."""
    if hasattr(texts, 'tolist'):
        texts = texts.tolist()
    return [str(t) if t is not None else '' for t in texts]


def _ensure_label_array(labels):
    """1-D numpy array로 정규화. int dtype 유지 시도."""
    arr = np.asarray(labels)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    return arr


def _tokenize_whitespace(text):
    """공백 기준 토큰화. DistilBERT BPE와 다르지만 cell 메트릭은 일관 정의."""
    return text.split()


# =================================================================
# 1. completeness_text — [MASK]/[PAD]/빈 토큰 비율 보수
# =================================================================

def calc_completeness_text(texts, sample_cap=2000, random_state=1):
    """텍스트별 비완전 토큰 비율의 평균 보수.

    비완전 = [MASK], [PAD], [UNK], [SEP], [CLS], 빈 문자열, 또는 토큰 없음.
    """
    texts = _ensure_text_list(texts)
    if not texts:
        return 1.0
    rng = np.random.RandomState(random_state)
    if sample_cap and len(texts) > sample_cap:
        idx = rng.choice(len(texts), sample_cap, replace=False)
        sample = [texts[i] for i in idx]
    else:
        sample = texts

    ratios = []
    for t in sample:
        toks = _tokenize_whitespace(t)
        if not toks:
            ratios.append(1.0)
            continue
        bad = sum(1 for tok in toks if tok in _PROTECTED or not tok.strip())
        ratios.append(bad / len(toks))
    return float(1.0 - np.mean(ratios)) if ratios else 1.0


# =================================================================
# 2. uniqueness — 정규화 hash 중복 비율 보수
# =================================================================

def _normalize(text):
    return _WS_RE.sub(' ', text.lower()).strip()


def calc_uniqueness(texts, sample_cap=2000, random_state=1):
    """1 - (중복 hash 비율). lower + whitespace 정규화 후 SHA-256."""
    texts = _ensure_text_list(texts)
    if len(texts) <= 1:
        return 1.0
    rng = np.random.RandomState(random_state)
    if sample_cap and len(texts) > sample_cap:
        idx = rng.choice(len(texts), sample_cap, replace=False)
        sample = [texts[i] for i in idx]
    else:
        sample = texts

    hashes = [hashlib.sha256(_normalize(t).encode('utf-8', errors='replace')).hexdigest()
              for t in sample]
    counts = Counter(hashes)
    n = len(hashes)
    duplicates = sum(c - 1 for c in counts.values() if c > 1)
    return float(1.0 - duplicates / n)


# =================================================================
# 3. validity — 비empty + UTF-8 + 최소 토큰
# =================================================================

def calc_validity(texts, sample_cap=2000, random_state=1):
    texts = _ensure_text_list(texts)
    if not texts:
        return 1.0
    rng = np.random.RandomState(random_state)
    if sample_cap and len(texts) > sample_cap:
        idx = rng.choice(len(texts), sample_cap, replace=False)
        sample = [texts[i] for i in idx]
    else:
        sample = texts

    valid = 0
    for t in sample:
        try:
            if not t:
                continue
            t.encode('utf-8')
            if len(_tokenize_whitespace(t)) >= 1:
                valid += 1
        except UnicodeEncodeError:
            pass
    return float(valid / len(sample))


# =================================================================
# 4. consistency — token count 5-bucket entropy 보수
# =================================================================

_LEN_BUCKETS = [0, 10, 50, 150, 500, float('inf')]


def calc_consistency(texts, sample_cap=2000, random_state=1):
    texts = _ensure_text_list(texts)
    if len(texts) <= 1:
        return 1.0
    rng = np.random.RandomState(random_state)
    if sample_cap and len(texts) > sample_cap:
        idx = rng.choice(len(texts), sample_cap, replace=False)
        sample = [texts[i] for i in idx]
    else:
        sample = texts

    counts = np.zeros(len(_LEN_BUCKETS) - 1, dtype=int)
    for t in sample:
        n = len(_tokenize_whitespace(t))
        for i in range(len(_LEN_BUCKETS) - 1):
            if _LEN_BUCKETS[i] <= n < _LEN_BUCKETS[i + 1]:
                counts[i] += 1
                break
    nonzero = counts[counts > 0]
    if len(nonzero) <= 1:
        return 1.0
    probs = nonzero / nonzero.sum()
    ent = float(-(probs * np.log(probs)).sum())
    max_ent = log(len(nonzero))
    return float(1.0 - ent / max_ent) if max_ent > 0 else 1.0


# =================================================================
# 5. outlier_ratio — token count IQR outlier 보수
# =================================================================

def calc_outlier_ratio(texts, sample_cap=2000, random_state=1):
    texts = _ensure_text_list(texts)
    if len(texts) < 4:
        return 1.0
    rng = np.random.RandomState(random_state)
    if sample_cap and len(texts) > sample_cap:
        idx = rng.choice(len(texts), sample_cap, replace=False)
        sample = [texts[i] for i in idx]
    else:
        sample = texts

    lengths = np.array([len(_tokenize_whitespace(t)) for t in sample])
    q1, q3 = np.percentile(lengths, 25), np.percentile(lengths, 75)
    iqr = q3 - q1
    if iqr == 0:
        return 1.0
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = ((lengths < lower) | (lengths > upper)).sum()
    return float(1.0 - outliers / len(lengths))


# =================================================================
# 6. class_balance — tabular와 동일
# =================================================================

def calc_class_balance(labels):
    arr = _ensure_label_array(labels)
    counts = np.bincount(arr) if arr.dtype.kind in 'iu' else Counter(arr.tolist())
    if isinstance(counts, np.ndarray):
        counts = counts[counts > 0]
        n_classes = len(counts)
    else:
        counts = list(counts.values())
        n_classes = len(counts)
    if n_classes <= 1:
        return 1.0
    counts_arr = np.asarray(counts, dtype=float)
    min_ratio = counts_arr.min() / counts_arr.sum()
    ideal_ratio = 1.0 / n_classes
    return float(min(min_ratio / ideal_ratio, 1.0))


# =================================================================
# DistilBERT embedding helper — torch/transformers lazy import
# =================================================================

_EMBED_CACHE = {'model': None, 'tokenizer': None}
_MODEL_ID = 'distilbert-base-uncased'
_MAX_LEN = 256


def _get_embedder():
    """DistilBERT-base-uncased lazy load. ADR-016 §3-2 사전등록."""
    if _EMBED_CACHE['model'] is not None:
        return _EMBED_CACHE['model'], _EMBED_CACHE['tokenizer']
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
    model = AutoModel.from_pretrained(_MODEL_ID)
    model.eval()
    _EMBED_CACHE['model'] = model
    _EMBED_CACHE['tokenizer'] = tokenizer
    return model, tokenizer


def _mean_pool(last_hidden, attention_mask):
    """attention_mask 적용 mean pooling (PAD 토큰 제외)."""
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _extract_features(texts, sample_cap=1000, random_state=1, batch_size=16):
    """DistilBERT mean-pool으로 N × 768 feature 추출.

    sample_cap=1000 (이미지 cell의 feature_correlation과 동일 사양).
    """
    import torch

    texts = _ensure_text_list(texts)
    rng = np.random.RandomState(random_state)
    if sample_cap and len(texts) > sample_cap:
        idx = rng.choice(len(texts), sample_cap, replace=False)
        sample = [texts[i] for i in idx]
        sample_idx = idx
    else:
        sample = texts
        sample_idx = np.arange(len(texts))

    # 빈 문자열은 placeholder로 (DistilBERT가 빈 input 못 받음)
    sample = [t if t else '[PAD]' for t in sample]

    model, tokenizer = _get_embedder()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    feats = []
    with torch.no_grad():
        for i in range(0, len(sample), batch_size):
            batch = sample[i:i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=_MAX_LEN, return_tensors='pt')
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            pooled = _mean_pool(out.last_hidden_state, enc['attention_mask'])
            feats.append(pooled.cpu().numpy())
    return np.concatenate(feats, axis=0), sample_idx


# =================================================================
# 7. feature_correlation — DistilBERT embedding cosine 상관 보수
# =================================================================

def _calc_feature_correlation_from_feats(feats, threshold=0.95):
    if feats.shape[1] < 2:
        return 1.0
    corr = np.corrcoef(feats.T)
    upper = np.triu(np.abs(corr), k=1)
    total_pairs = (corr.shape[0] * (corr.shape[0] - 1)) // 2
    high = (upper > threshold).sum()
    return float(1.0 - high / total_pairs) if total_pairs > 0 else 1.0


def calc_feature_correlation(texts, sample_cap=1000, random_state=1, threshold=0.95):
    """DistilBERT 768-d 임베딩 차원 간 |corr|>threshold 비율의 보수."""
    if len(texts) < 10:
        return 1.0
    feats, _ = _extract_features(texts, sample_cap=sample_cap, random_state=random_state)
    return _calc_feature_correlation_from_feats(feats, threshold=threshold)


# =================================================================
# 8. label_consistency — k-NN embedding 라벨 일관성 (chance 보정)
# =================================================================

def _calc_label_consistency_from_feats(feats, y, k=5):
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    if len(feats) < k + 1:
        return 1.0
    feats_std = StandardScaler().fit_transform(feats)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(feats_std)
    _, idx = nn.kneighbors(feats_std)
    raw = (y[idx[:, 1:]] == y[:, None]).mean()
    if y.dtype.kind in 'iu':
        counts = np.bincount(y)
        class_props = counts[counts > 0] / counts.sum()
    else:
        cnt = Counter(y.tolist())
        class_props = np.array(list(cnt.values())) / sum(cnt.values())
    chance = float((class_props ** 2).sum())
    if chance >= 1.0:
        return 1.0
    return float(np.clip((raw - chance) / (1.0 - chance), 0.0, 1.0))


def calc_label_consistency(texts, labels, k=5, sample_cap=1000, random_state=1):
    if len(texts) < k + 1:
        return 1.0
    feats, sample_idx = _extract_features(texts, sample_cap=sample_cap, random_state=random_state)
    y = _ensure_label_array(labels)[sample_idx]
    return _calc_label_consistency_from_feats(feats, y, k=k)


# =================================================================
# 9. feature_informativeness — embedding → label MI / H(Y)
# =================================================================

def _calc_feature_informativeness_from_feats(feats, y, random_state=1):
    from sklearn.feature_selection import mutual_info_classif

    try:
        mi = mutual_info_classif(feats, y, discrete_features=False, random_state=random_state)
    except Exception:
        return 1.0

    if y.dtype.kind in 'iu':
        counts = np.bincount(y)
        class_props = counts[counts > 0] / counts.sum()
    else:
        cnt = Counter(y.tolist())
        class_props = np.array(list(cnt.values())) / sum(cnt.values())
    h_y = float(-(class_props * np.log(class_props)).sum())
    if h_y <= 0:
        return 1.0
    return float(np.clip(mi.sum() / h_y, 0.0, 1.0))


def calc_feature_informativeness(texts, labels, sample_cap=1000, random_state=1):
    if len(texts) < 10:
        return 1.0
    feats, sample_idx = _extract_features(texts, sample_cap=sample_cap, random_state=random_state)
    y = _ensure_label_array(labels)[sample_idx]
    return _calc_feature_informativeness_from_feats(feats, y, random_state=random_state)


# =================================================================
# 10. sample_quality_text — TTR + length_adequacy 결합 (NEW)
# =================================================================

def calc_sample_quality_text(texts, target_len=20, ttr_scale=2.0,
                             sample_cap=2000, random_state=1):
    """sample-level 텍스트 품질 — type-token ratio + length adequacy 결합.

    각 텍스트에 대해:
        ttr      = unique_tokens / max(1, total_tokens)
        ttr_score = min(1, ttr * ttr_scale)
        length_score = min(1, token_count / target_len)
    text 점수 = (ttr_score + length_score) / 2
    cell 점수 = 평균.

    target_len, ttr_scale은 ADR-016에 사전등록 (Phase 1 verify 후 조정 가능).
    """
    texts = _ensure_text_list(texts)
    if not texts:
        return 1.0
    rng = np.random.RandomState(random_state)
    if sample_cap and len(texts) > sample_cap:
        idx = rng.choice(len(texts), sample_cap, replace=False)
        sample = [texts[i] for i in idx]
    else:
        sample = texts

    scores = []
    for t in sample:
        toks = _tokenize_whitespace(t)
        if not toks:
            scores.append(0.0)
            continue
        ttr = len(set(toks)) / len(toks)
        ttr_score = min(1.0, ttr * ttr_scale)
        length_score = min(1.0, len(toks) / target_len)
        scores.append((ttr_score + length_score) / 2)
    return float(np.mean(scores))


# =================================================================
# 가중치 + 통합 진입점
# =================================================================

DEFAULT_WEIGHTS_TEXT = {
    'completeness_text':        0.15,
    'uniqueness':               0.10,
    'validity':                 0.05,
    'consistency':              0.05,
    'outlier_ratio':            0.05,
    'class_balance':            0.10,
    'feature_correlation':      0.05,
    'label_consistency':        0.20,
    'feature_informativeness':  0.10,
    'sample_quality_text':      0.15,
}


def compute_dsc_text(texts, labels, weights=None,
                     use_embeddings=True,
                     sample_cap=1000, random_state=1):
    """DSC text cell 점수 (0~100) + 등급 + 지표별.

    Args:
        texts: list[str] / pd.Series / np.ndarray
        labels: list/np.ndarray of int
        weights: 가중치 dict (None → DEFAULT_WEIGHTS_TEXT, ADR-015 fallback)
        use_embeddings: False면 DistilBERT 의존 메트릭 3개를 1.0 처리 (debug)
        sample_cap: embedding 추출 sample cap (cost 통제용)

    Pre-registered: ADR-016. 운영 시 weights는 LLM weight generator 출력 사용.
    """
    w = weights or DEFAULT_WEIGHTS_TEXT
    texts = _ensure_text_list(texts)
    y_arr = _ensure_label_array(labels)

    metrics = {
        'completeness_text':   calc_completeness_text(texts, sample_cap=sample_cap, random_state=random_state),
        'uniqueness':          calc_uniqueness(texts, sample_cap=sample_cap, random_state=random_state),
        'validity':            calc_validity(texts, sample_cap=sample_cap, random_state=random_state),
        'consistency':         calc_consistency(texts, sample_cap=sample_cap, random_state=random_state),
        'outlier_ratio':       calc_outlier_ratio(texts, sample_cap=sample_cap, random_state=random_state),
        'class_balance':       calc_class_balance(y_arr),
        'sample_quality_text': calc_sample_quality_text(texts, sample_cap=sample_cap, random_state=random_state),
    }
    if use_embeddings and len(texts) >= 10:
        feats, sample_idx = _extract_features(texts, sample_cap=sample_cap, random_state=random_state)
        y_sample = y_arr[sample_idx]
        metrics['feature_correlation'] = _calc_feature_correlation_from_feats(feats)
        metrics['label_consistency'] = (
            _calc_label_consistency_from_feats(feats, y_sample, k=5)
            if len(feats) >= 6 else 1.0
        )
        metrics['feature_informativeness'] = _calc_feature_informativeness_from_feats(
            feats, y_sample, random_state=random_state)
    else:
        metrics['feature_correlation'] = 1.0
        metrics['label_consistency'] = 1.0
        metrics['feature_informativeness'] = 1.0

    score = sum(metrics[k] * w[k] for k in w) * 100
    rounded = {k: round(v, 4) for k, v in metrics.items()}
    return {'score': round(score, 2), 'grade': to_grade(score),
            **rounded, 'metrics': rounded}
