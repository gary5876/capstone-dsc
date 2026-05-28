"""입력 형태 감지 — DataFrame / 이미지 컬렉션 / etc.

router.py에서 select_profile((data_type, task)) 분기 위해 사용.
"""


def detect_data_type(input_obj):
    """입력 형태 추정.

    Returns:
        'tabular'     — pandas DataFrame
        'text'        — list[str] / pd.Series[str] / (texts, labels) tuple
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
        # pandas Series of str → text
        if isinstance(input_obj, pd.Series):
            if input_obj.dtype == object and len(input_obj) > 0:
                first = input_obj.iloc[0]
                if isinstance(first, str):
                    return 'text'
    except ImportError:
        pass

    # torch Dataset (이미지 가정 — 텍스트 dataset도 가능하나 캡스톤 범위에서는 이미지로 폴백)
    try:
        import torch
        if isinstance(input_obj, torch.utils.data.Dataset):
            return 'image'
    except ImportError:
        pass

    # tuple (data, labels)
    if isinstance(input_obj, tuple) and len(input_obj) == 2:
        data, labels = input_obj
        if hasattr(data, '__len__') and hasattr(labels, '__len__') and len(data) > 0:
            first = data[0] if hasattr(data, '__getitem__') else None
            if isinstance(first, str):
                return 'text'
            if _looks_like_image(first):
                return 'image'

    # list-like of items
    if hasattr(input_obj, '__len__') and hasattr(input_obj, '__getitem__') \
            and not isinstance(input_obj, str):
        try:
            first = input_obj[0]
            if isinstance(first, str):
                return 'text'
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
