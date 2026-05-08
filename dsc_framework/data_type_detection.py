"""입력 형태 감지 — DataFrame / 이미지 컬렉션 / etc.

router.py에서 select_profile((data_type, task)) 분기 위해 사용.
"""


def detect_data_type(input_obj):
    """입력 형태 추정.

    Returns:
        'tabular'     — pandas DataFrame
        'image'       — PyTorch Dataset, list of PIL/np 이미지, 또는 (images, labels) tuple
        'unknown'     — 그 외

    Args:
        input_obj: 분석할 객체
    """
    # pandas DataFrame
    try:
        import pandas as pd
        if isinstance(input_obj, pd.DataFrame):
            return 'tabular'
    except ImportError:
        pass

    # torch Dataset
    try:
        import torch
        if isinstance(input_obj, torch.utils.data.Dataset):
            return 'image'
    except ImportError:
        pass

    # tuple (images, labels)
    if isinstance(input_obj, tuple) and len(input_obj) == 2:
        images, labels = input_obj
        if hasattr(images, '__len__') and hasattr(labels, '__len__'):
            if len(images) > 0:
                # PIL 또는 numpy ndarray 또는 Tensor
                first = images[0]
                if _looks_like_image(first):
                    return 'image'

    # list-like of images
    if hasattr(input_obj, '__len__') and hasattr(input_obj, '__getitem__'):
        try:
            first = input_obj[0]
            if _looks_like_image(first):
                return 'image'
        except (IndexError, TypeError):
            pass

    return 'unknown'


def _looks_like_image(obj):
    """객체가 이미지처럼 생겼는지 — PIL.Image, np.ndarray (≥2D), torch.Tensor (≥2D)."""
    try:
        from PIL import Image
        if isinstance(obj, Image.Image):
            return True
    except ImportError:
        pass

    try:
        import numpy as np
        if isinstance(obj, np.ndarray) and obj.ndim >= 2:
            return True
    except ImportError:
        pass

    try:
        import torch
        if isinstance(obj, torch.Tensor) and obj.ndim >= 2:
            return True
    except ImportError:
        pass

    return False
