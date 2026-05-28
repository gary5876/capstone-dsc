"""DSC v5 Framework — task-conditional data quality scoring.

Public API:
    compute_dsc(input_obj=..., data_type=None, task=None) — 통합 진입점
        - tabular: compute_dsc(df=df)
        - image:   compute_dsc(images=..., labels=..., data_type='image')
        - text:    compute_dsc(texts=..., labels=..., data_type='text')
                   compute_dsc(texts=..., targets=..., data_type='text', task='regression')
    auto_detect_columns(df) → (target, num_cols, cat_cols, task)
    detect_data_type(input_obj) → 'tabular' | 'image' | 'text' | 'unknown'
    select_profile((data_type, task)) → {compute_fn, default_weights}

Cell-specific (필요 시):
    compute_dsc_classification(...)
    compute_dsc_regression(...)
    compute_dsc_image(images, labels, ...)
    compute_dsc_text(texts, labels, ...)
    compute_dsc_text_regression(texts, targets, ...)
    compute_dsc_degradation(polluted, clean) — ADR-012 보조 지표

Pre-registered weights (ADR-009/011/012/014/016/017, v4·v5 마스터플랜):
    DEFAULT_WEIGHTS_CLASSIFICATION, DEFAULT_WEIGHTS_REGRESSION,
    DEFAULT_WEIGHTS_IMAGE, DEFAULT_WEIGHTS_TEXT, DEFAULT_WEIGHTS_TEXT_REG

**Robust import 정책**: 패키지 일부 파일이 누락 (G드라이브 부분 sync 등)된
환경에서도 가능한 cell만 노출하여 부분 사용을 허용. 누락 항목은 warning.
"""
from __future__ import annotations

import warnings

_FAILED: dict[str, str] = {}

try:
    from .classification_cell import (
        DEFAULT_WEIGHTS_CLASSIFICATION, compute_dsc_classification,
    )
except Exception as _e:  # noqa: BLE001
    _FAILED['classification_cell'] = repr(_e)
    DEFAULT_WEIGHTS_CLASSIFICATION = None
    compute_dsc_classification = None

try:
    from .column_detection import auto_detect_columns, detect_task
except Exception as _e:
    _FAILED['column_detection'] = repr(_e)
    auto_detect_columns = None
    detect_task = None

try:
    from .data_type_detection import detect_data_type
except Exception as _e:
    _FAILED['data_type_detection'] = repr(_e)
    detect_data_type = None

try:
    from .image_cell import DEFAULT_WEIGHTS_IMAGE, compute_dsc_image
except Exception as _e:
    _FAILED['image_cell'] = repr(_e)
    DEFAULT_WEIGHTS_IMAGE = None
    compute_dsc_image = None

try:
    from .regression_cell import (
        DEFAULT_WEIGHTS_REGRESSION, compute_dsc_degradation, compute_dsc_regression,
    )
except Exception as _e:
    _FAILED['regression_cell'] = repr(_e)
    DEFAULT_WEIGHTS_REGRESSION = None
    compute_dsc_degradation = None
    compute_dsc_regression = None

try:
    from .text_cell import DEFAULT_WEIGHTS_TEXT, compute_dsc_text
except Exception as _e:
    _FAILED['text_cell'] = repr(_e)
    DEFAULT_WEIGHTS_TEXT = None
    compute_dsc_text = None

try:
    from .text_cell_regression import (
        DEFAULT_WEIGHTS_TEXT_REG, compute_dsc_text_regression,
    )
except Exception as _e:
    _FAILED['text_cell_regression'] = repr(_e)
    DEFAULT_WEIGHTS_TEXT_REG = None
    compute_dsc_text_regression = None

try:
    from .text_trainers import (
        CLASSIFICATION_MODELS, REGRESSION_MODELS,
        train_logreg_tfidf, train_ridge_tfidf, train_textcnn,
        train_transformer, train_xgb_tfidf,
    )
except Exception as _e:
    _FAILED['text_trainers'] = repr(_e)
    CLASSIFICATION_MODELS = None
    REGRESSION_MODELS = None
    train_logreg_tfidf = None
    train_ridge_tfidf = None
    train_textcnn = None
    train_transformer = None
    train_xgb_tfidf = None

try:
    from .router import compute_dsc, select_profile
except Exception as _e:
    _FAILED['router'] = repr(_e)
    compute_dsc = None
    select_profile = None

if _FAILED:
    warnings.warn(
        'dsc_framework: 일부 submodule import 실패 (부분 동작 모드). '
        f'실패: {sorted(_FAILED.keys())}. '
        f'전체 메시지: {_FAILED}'
    )

__all__ = [
    'compute_dsc',
    'auto_detect_columns',
    'detect_task',
    'detect_data_type',
    'select_profile',
    'compute_dsc_classification',
    'compute_dsc_regression',
    'compute_dsc_image',
    'compute_dsc_text',
    'compute_dsc_text_regression',
    'compute_dsc_degradation',
    'DEFAULT_WEIGHTS_CLASSIFICATION',
    'DEFAULT_WEIGHTS_REGRESSION',
    'DEFAULT_WEIGHTS_IMAGE',
    'DEFAULT_WEIGHTS_TEXT',
    'DEFAULT_WEIGHTS_TEXT_REG',
]
