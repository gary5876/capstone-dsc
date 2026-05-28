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
    DEFAULT_WEIGHTS_CLASSIFICATION
    DEFAULT_WEIGHTS_REGRESSION
    DEFAULT_WEIGHTS_IMAGE
    DEFAULT_WEIGHTS_TEXT
    DEFAULT_WEIGHTS_TEXT_REG
"""
from .classification_cell import (
    DEFAULT_WEIGHTS_CLASSIFICATION, compute_dsc_classification,
)
from .column_detection import auto_detect_columns, detect_task
from .data_type_detection import detect_data_type
from .image_cell import DEFAULT_WEIGHTS_IMAGE, compute_dsc_image
from .regression_cell import (
    DEFAULT_WEIGHTS_REGRESSION, compute_dsc_degradation, compute_dsc_regression,
)
from .router import compute_dsc, select_profile
from .text_cell import DEFAULT_WEIGHTS_TEXT, compute_dsc_text
from .text_cell_regression import (
    DEFAULT_WEIGHTS_TEXT_REG, compute_dsc_text_regression,
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
