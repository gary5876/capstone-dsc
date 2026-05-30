"""DSC v5 이미지 cell — image × classification.

ADR-014 사전등록 (10개 메트릭, 가중치 합 1.00):
- completeness_image (마스킹 비율 보수)         : 0.15
- uniqueness        (perceptual hash 중복 보수)  : 0.10
- validity          (이미지 load 성공 비율)      : 0.05
- consistency       (color mode + size 일관성)   : 0.05
- outlier_ratio     (mean intensity IQR 보수)    : 0.05
- class_balance     (tabular와 동일 정의식)       : 0.10
- feature_correlation (embedding cosine 상관)     : 0.05
- label_consistency  (k-NN embedding 라벨 일관성) : 0.20
- feature_informativeness (embedding → label MI)  : 0.10
- sample_quality_image (blur + contrast 결합)     : 0.15  ← 신설

ADR-011 강한 버전 원칙: 차원 이름이 같아도 정의식이 cell마다 다름.
예) feature_correlation은 tabular는 컬럼 Pearson, image는 ResNet embedding cosine.

입력 형식:
    images: list of np.ndarray (HWC, uint8) or PIL Image — torchvision Dataset의 (img, _) 시퀀스에서 추출 가능
    labels: list/np.ndarray of int (분류 라벨)

PyTorch/torchvision 의존 메트릭(feature_*, label_consistency)은 lazy import.
캡스톤은 GPU 환경(Colab) 가정. CPU만 있으면 sample_cap을 작게.
"""
from __future__ import annotations

from collections import Counter
from math import log

import numpy as np

from .shared_metrics import to_grade


# =================================================================
# 이미지 입력 정규화 helper
# =================================================================

def _to_np_uint8(img):
    """PIL.Image / np.ndarray / torch.Tensor → np.ndarray (HWC, uint8).

    torch.Tensor는 (C,H,W) float[0,1] 또는 (H,W,C) uint8 양쪽 처리.
    """
    # PIL
    try:
        from PIL import Image
        if isinstance(img, Image.Image):
            arr = np.array(img)
            if arr.ndim == 2:
                arr = arr[:, :, None]
            return arr.astype(np.uint8)
    except ImportError:
        pass
    # torch
    try:
        import torch
        if isinstance(img, torch.Tensor):
            arr = img.detach().cpu().numpy()
            if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
                arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype != np.uint8:
                arr = (arr * 255).clip(0, 255).astype(np.uint8) if arr.max() <= 1.5 else arr.astype(np.uint8)
            if arr.ndim == 2:
                arr = arr[:, :, None]
            return arr
    except ImportError:
        pass
    # np
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr


# =================================================================
# 1. completeness_image — 마스킹/배경(black) 픽셀 비율 보수
# =================================================================

def calc_completeness_image(images, mask_value=0, mask_tolerance=2,
                            sample_cap=2000, random_state=1):
    """모든 채널이 mask_value 근처인 픽셀 비율의 평균 보수 (1 - mean ratio).

    mask_value=0이면 검은 픽셀(완전 black)을 마스킹으로 간주.
    1.0 = 마스킹 없음, 0.0 = 전부 마스킹.
    """
    if len(images) == 0:
        return 1.0
    rng = np.random.RandomState(random_state)
    if sample_cap and len(images) > sample_cap:
        idx = rng.choice(len(images), sample_cap, replace=False)
        sample = [images[i] for i in idx]
    else:
        sample = images

    ratios = []
    for img in sample:
        arr = _to_np_uint8(img)
        # 모든 채널이 mask_value 근처
        is_mask = np.all(np.abs(arr.astype(int) - mask_value) <= mask_tolerance, axis=-1)
        ratios.append(float(is_mask.mean()))
    return float(1.0 - np.mean(ratios)) if ratios else 1.0


# =================================================================
# 2. uniqueness — perceptual hash 기반 중복 보수
# =================================================================

def _phash_fallback(arr_uint8, hash_size=8):
    """imagehash 없을 때 폴백 — 8x8 평균 비교 hash."""
    # 2D 그레이스케일로
    if arr_uint8.ndim == 3:
        gray = arr_uint8.mean(axis=-1)
    else:
        gray = arr_uint8
    # resize (단순 양자화)
    h, w = gray.shape
    if h < hash_size or w < hash_size:
        gray = np.pad(gray, ((0, max(0, hash_size - h)), (0, max(0, hash_size - w))))
    bh, bw = gray.shape[0] // hash_size, gray.shape[1] // hash_size
    if bh == 0 or bw == 0:
        return tuple(gray.flatten().astype(int).tolist())
    small = gray[:bh * hash_size, :bw * hash_size].reshape(hash_size, bh, hash_size, bw).mean(axis=(1, 3))
    bits = (small > small.mean()).flatten()
    return tuple(int(b) for b in bits)


def calc_uniqueness(images, sample_cap=2000, random_state=1):
    """1 - (중복 hash 비율). imagehash 있으면 phash, 없으면 폴백."""
    if len(images) <= 1:
        return 1.0
    rng = np.random.RandomState(random_state)
    if sample_cap and len(images) > sample_cap:
        idx = rng.choice(len(images), sample_cap, replace=False)
        sample = [images[i] for i in idx]
    else:
        sample = images

    try:
        import imagehash
        from PIL import Image
        hashes = []
        for img in sample:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(_to_np_uint8(img).squeeze())
            hashes.append(str(imagehash.phash(img)))
    except ImportError:
        hashes = [_phash_fallback(_to_np_uint8(img)) for img in sample]

    counts = Counter(hashes)
    n = len(hashes)
    duplicates = sum(c - 1 for c in counts.values() if c > 1)
    return float(1.0 - duplicates / n)


# =================================================================
# 3. validity — load/decode 성공 비율
# =================================================================

def calc_validity(images, sample_cap=2000, random_state=1):
    """numpy 변환 가능 + 0이 아닌 dim 비율. 1.0 = 모두 정상."""
    if len(images) == 0:
        return 1.0
    rng = np.random.RandomState(random_state)
    if sample_cap and len(images) > sample_cap:
        idx = rng.choice(len(images), sample_cap, replace=False)
        sample = [images[i] for i in idx]
    else:
        sample = images

    valid = 0
    for img in sample:
        try:
            arr = _to_np_uint8(img)
            if arr.size > 0 and arr.ndim >= 2:
                valid += 1
        except Exception:
            pass
    return float(valid / len(sample))


# =================================================================
# 4. consistency — color mode + size 일관성 (분포 entropy 보수)
# =================================================================

def calc_consistency(images, sample_cap=2000, random_state=1):
    """이미지 mode(채널 수) + size(H,W) 분포의 entropy 기반 일관성.
    동일 mode + 동일 size면 1.0, 분산되어 있으면 0에 가까움."""
    if len(images) <= 1:
        return 1.0
    rng = np.random.RandomState(random_state)
    if sample_cap and len(images) > sample_cap:
        idx = rng.choice(len(images), sample_cap, replace=False)
        sample = [images[i] for i in idx]
    else:
        sample = images

    keys = []
    for img in sample:
        arr = _to_np_uint8(img)
        keys.append((arr.shape[0], arr.shape[1], arr.shape[2] if arr.ndim == 3 else 1))

    counts = Counter(keys)
    probs = np.array([c / len(keys) for c in counts.values()])
    if len(probs) <= 1:
        return 1.0
    ent = float(-(probs * np.log(probs)).sum())
    max_ent = log(len(probs))
    # entropy가 작을수록 일관성 높음 → 1 - normalized entropy
    return float(1.0 - ent / max_ent) if max_ent > 0 else 1.0


# =================================================================
# 5. outlier_ratio — mean intensity의 IQR-based outlier 보수
# =================================================================

def calc_outlier_ratio(images, sample_cap=2000, random_state=1):
    """이미지 평균 intensity의 IQR-based outlier 비율의 보수."""
    if len(images) < 4:
        return 1.0
    rng = np.random.RandomState(random_state)
    if sample_cap and len(images) > sample_cap:
        idx = rng.choice(len(images), sample_cap, replace=False)
        sample = [images[i] for i in idx]
    else:
        sample = images

    means = np.array([_to_np_uint8(img).mean() for img in sample])
    q1, q3 = np.percentile(means, 25), np.percentile(means, 75)
    iqr = q3 - q1
    if iqr == 0:
        return 1.0
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = ((means < lower) | (means > upper)).sum()
    return float(1.0 - outliers / len(means))


# =================================================================
# 6. class_balance — tabular와 동일 정의식
# =================================================================

def calc_class_balance(labels):
    """min_ratio / ideal_ratio. tabular cell의 calc_class_balance와 동일."""
    arr = np.asarray(labels)
    counts = np.bincount(arr) if arr.dtype.kind in 'iu' else Counter(arr.tolist())
    if isinstance(counts, np.ndarray):
        counts = counts[counts > 0]
        n_classes = len(counts)
    else:
        counts = list(counts.values())
        n_classes = len(counts)
    if n_classes <= 1:
        return 1.0
    counts = np.asarray(counts, dtype=float)
    min_ratio = counts.min() / counts.sum()
    ideal_ratio = 1.0 / n_classes
    return float(min(min_ratio / ideal_ratio, 1.0))


# =================================================================
# Embedding helper — torchvision 의존 (lazy import)
# =================================================================

_EMBED_CACHE = {'model': None, 'transform': None}


def _get_embedder():
    """ResNet18 pretrained을 feature extractor로. ADR-014 사전등록."""
    if _EMBED_CACHE['model'] is not None:
        return _EMBED_CACHE['model'], _EMBED_CACHE['transform']
    import torch
    import torchvision.models as tvm
    import torchvision.transforms as T

    model = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT)
    # avgpool 직전까지 — 512-d feature
    model.fc = torch.nn.Identity()
    model.eval()

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    _EMBED_CACHE['model'] = model
    _EMBED_CACHE['transform'] = transform
    return model, transform


def _extract_features(images, sample_cap=2000, random_state=1, batch_size=64):
    """ResNet18 pretrained으로 N×512 feature 추출."""
    import torch
    from PIL import Image

    rng = np.random.RandomState(random_state)
    if sample_cap and len(images) > sample_cap:
        idx = rng.choice(len(images), sample_cap, replace=False)
        sample = [images[i] for i in idx]
        sample_idx = idx
    else:
        sample = images
        sample_idx = np.arange(len(images))

    model, transform = _get_embedder()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    feats = []
    with torch.no_grad():
        for i in range(0, len(sample), batch_size):
            batch_imgs = sample[i:i + batch_size]
            tensors = []
            for img in batch_imgs:
                if isinstance(img, np.ndarray):
                    arr = _to_np_uint8(img).squeeze()
                    pil = Image.fromarray(arr) if arr.ndim == 2 else Image.fromarray(arr[..., :3] if arr.shape[-1] >= 3 else arr.repeat(3, axis=-1))
                    if pil.mode != 'RGB':
                        pil = pil.convert('RGB')
                else:
                    pil = img
                    if hasattr(pil, 'mode') and pil.mode != 'RGB':
                        pil = pil.convert('RGB')
                tensors.append(transform(pil))
            batch_t = torch.stack(tensors).to(device)
            f = model(batch_t).cpu().numpy()
            feats.append(f)
    return np.concatenate(feats, axis=0), sample_idx


# =================================================================
# 7. feature_correlation — embedding cosine 상관
# =================================================================

def _calc_feature_correlation_from_feats(feats, threshold=0.95):
    if feats.shape[1] < 2:
        return 1.0
    corr = np.corrcoef(feats.T)
    upper = np.triu(np.abs(corr), k=1)
    total_pairs = (corr.shape[0] * (corr.shape[0] - 1)) // 2
    high = (upper > threshold).sum()
    return float(1.0 - high / total_pairs) if total_pairs > 0 else 1.0


def calc_feature_correlation(images, sample_cap=1000, random_state=1, threshold=0.95):
    """ResNet18 embedding 차원 간 상관 — 고상관(>threshold) 차원 비율의 보수.
    임베딩 차원 페어 수가 많아 sample_cap 작게."""
    if len(images) < 10:
        return 1.0
    feats, _ = _extract_features(images, sample_cap=sample_cap, random_state=random_state)
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
    counts = np.bincount(y) if y.dtype.kind in 'iu' else np.array(list(Counter(y.tolist()).values()))
    total = counts.sum()
    class_props = counts[counts > 0] / total
    chance = float((class_props ** 2).sum())
    if chance >= 1.0:
        return 1.0
    return float(np.clip((raw - chance) / (1.0 - chance), 0.0, 1.0))


def calc_label_consistency(images, labels, k=5, sample_cap=2000, random_state=1):
    """tabular cell의 calc_label_consistency 패턴 — feature 공간만 변경.
    chance level 보정 포함."""
    if len(images) < k + 1:
        return 1.0
    feats, sample_idx = _extract_features(images, sample_cap=sample_cap, random_state=random_state)
    y = np.asarray(labels)[sample_idx]
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

    counts = np.bincount(y) if y.dtype.kind in 'iu' else np.array(list(Counter(y.tolist()).values()))
    total = counts.sum()
    class_props = counts[counts > 0] / total
    h_y = float(-(class_props * np.log(class_props)).sum())
    if h_y <= 0:
        return 1.0
    return float(np.clip(mi.sum() / h_y, 0.0, 1.0))


def calc_feature_informativeness(images, labels, sample_cap=2000, random_state=1):
    if len(images) < 10:
        return 1.0
    feats, sample_idx = _extract_features(images, sample_cap=sample_cap, random_state=random_state)
    y = np.asarray(labels)[sample_idx]
    return _calc_feature_informativeness_from_feats(feats, y, random_state=random_state)


# =================================================================
# 10. sample_quality_image — blur + contrast 결합 (NEW, 이미지 cell 신설)
# =================================================================

def _laplacian_variance(gray_uint8):
    """blur 측정: Laplacian variance. cv2 우선, 없으면 scipy fallback."""
    try:
        import cv2
        lap = cv2.Laplacian(gray_uint8, cv2.CV_64F)
        return float(lap.var())
    except ImportError:
        from scipy import ndimage
        lap = ndimage.laplace(gray_uint8.astype(np.float64))
        return float(lap.var())


def calc_sample_quality_image(images, blur_norm=200.0, contrast_norm=50.0,
                              sample_cap=2000, random_state=1):
    """sample-level 이미지 품질 — blur(Laplacian variance) + contrast(RMS) 결합.

    각 이미지에 대해:
        blur_score    = min(1, lap_var / blur_norm)        # blur ↑ → score ↓
        contrast_score = min(1, RMS / contrast_norm)        # contrast ↑ → score ↑
    이미지 점수 = (blur_score + contrast_score) / 2
    cell 점수 = 평균.

    norm 값은 ADR-014에 사전등록되며, Phase 1 verify에서 분포 보고 후 보충 ADR로 조정 가능.
    """
    if len(images) == 0:
        return 1.0
    rng = np.random.RandomState(random_state)
    if sample_cap and len(images) > sample_cap:
        idx = rng.choice(len(images), sample_cap, replace=False)
        sample = [images[i] for i in idx]
    else:
        sample = images

    scores = []
    for img in sample:
        arr = _to_np_uint8(img)
        gray = arr.mean(axis=-1).astype(np.uint8) if arr.ndim == 3 else arr
        lap_var = _laplacian_variance(gray)
        rms = float(gray.std())
        blur_score = min(1.0, lap_var / blur_norm)
        contrast_score = min(1.0, rms / contrast_norm)
        scores.append((blur_score + contrast_score) / 2)
    return float(np.mean(scores))


# =================================================================
# 가중치 + 통합 진입점
# =================================================================

DEFAULT_WEIGHTS_IMAGE = {
    'completeness_image':       0.15,
    'uniqueness':               0.10,
    'validity':                 0.05,
    'consistency':              0.05,
    'outlier_ratio':            0.05,
    'class_balance':            0.10,
    'feature_correlation':      0.05,
    'label_consistency':        0.20,
    'feature_informativeness':  0.10,
    'sample_quality_image':     0.15,
}


def compute_dsc_image(images, labels, weights=None,
                      use_embeddings=True,
                      sample_cap=2000, random_state=1):
    """DSC image cell 점수 (0~100) + 등급 + 지표별.

    Args:
        images: list of np.ndarray/PIL.Image/torch.Tensor
        labels: list/np.ndarray of int (분류 라벨)
        weights: 가중치 dict (None → DEFAULT_WEIGHTS_IMAGE)
        use_embeddings: False면 ResNet 의존 메트릭 3개를 1.0으로 폴백 (debug용)

    Pre-registered: ADR-014, 이미지 cell 마스터플랜.
    """
    w = weights or DEFAULT_WEIGHTS_IMAGE

    metrics = {
        'completeness_image':  calc_completeness_image(images, sample_cap=sample_cap, random_state=random_state),
        'uniqueness':          calc_uniqueness(images, sample_cap=sample_cap, random_state=random_state),
        'validity':            calc_validity(images, sample_cap=sample_cap, random_state=random_state),
        'consistency':         calc_consistency(images, sample_cap=sample_cap, random_state=random_state),
        'outlier_ratio':       calc_outlier_ratio(images, sample_cap=sample_cap, random_state=random_state),
        'class_balance':       calc_class_balance(labels),
        'sample_quality_image': calc_sample_quality_image(images, sample_cap=sample_cap, random_state=random_state),
    }
    if use_embeddings and len(images) >= 10:
        # ResNet18 feature를 1번만 추출, 3개 embedding 메트릭이 공유 (~3x 가속).
        feats, sample_idx = _extract_features(images, sample_cap=sample_cap, random_state=random_state)
        y_sample = np.asarray(labels)[sample_idx]
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
