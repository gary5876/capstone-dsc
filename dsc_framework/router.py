"""DSC v5 Framework 라우터 — (data_type, task)에 따라 cell 엔진 선택.

select_profile((data_type, task)) → (compute_fn, default_weights)
compute_dsc(...)                  → 통합 진입점. data_type/task auto_detect.

웹 백엔드 측 webplatform 등 외부에서는 이 모듈만 import:
    from dsc_framework import compute_dsc, auto_detect_columns
"""
from .classification_cell import (
    DEFAULT_WEIGHTS_CLASSIFICATION, compute_dsc_classification,
)
from .column_detection import auto_detect_columns
from .data_type_detection import detect_data_type
from .image_cell import DEFAULT_WEIGHTS_IMAGE, compute_dsc_image
from .image_cell_regression import (
    DEFAULT_WEIGHTS_IMAGE_REG, compute_dsc_image_regression,
)
from .regression_cell import (
    DEFAULT_WEIGHTS_REGRESSION, compute_dsc_regression,
)
from .text_cell import DEFAULT_WEIGHTS_TEXT, compute_dsc_text
from .text_cell_regression import (
    DEFAULT_WEIGHTS_TEXT_REG, compute_dsc_text_regression,
)


_PROFILES = {
    ('tabular', 'classification'): {
        'compute_fn': compute_dsc_classification,
        'default_weights': DEFAULT_WEIGHTS_CLASSIFICATION,
    },
    ('tabular', 'regression'): {
        'compute_fn': compute_dsc_regression,
        'default_weights': DEFAULT_WEIGHTS_REGRESSION,
    },
    ('image', 'classification'): {
        'compute_fn': compute_dsc_image,
        'default_weights': DEFAULT_WEIGHTS_IMAGE,
    },
    ('image', 'regression'): {
        'compute_fn': compute_dsc_image_regression,
        'default_weights': DEFAULT_WEIGHTS_IMAGE_REG,
    },
    ('text', 'classification'): {
        'compute_fn': compute_dsc_text,
        'default_weights': DEFAULT_WEIGHTS_TEXT,
    },
    ('text', 'regression'): {
        'compute_fn': compute_dsc_text_regression,
        'default_weights': DEFAULT_WEIGHTS_TEXT_REG,
    },
}

# task 단독 키도 지원 (backward-compat — tabular가 default data_type)
_TASK_ONLY_FALLBACK = {
    'classification': ('tabular', 'classification'),
    'regression': ('tabular', 'regression'),
}


def select_profile(key):
    """(data_type, task) tuple 또는 task 문자열 → {compute_fn, default_weights}.

    task 문자열만 받으면 tabular로 폴백 (회귀 cell 노트북과의 호환).
    """
    if isinstance(key, str):
        if key in _TASK_ONLY_FALLBACK:
            key = _TASK_ONLY_FALLBACK[key]
        else:
            raise ValueError(
                f"Unknown task '{key}'. Supported task strings: {sorted(_TASK_ONLY_FALLBACK.keys())}")
    if key not in _PROFILES:
        raise ValueError(
            f"Unknown profile {key}. Supported: {sorted(_PROFILES.keys())}")
    return _PROFILES[key]


def compute_dsc(input_obj=None, df=None,
                target_col=None, numerical_cols=None, categorical_cols=None,
                images=None, labels=None,
                texts=None, targets=None,
                data_type=None, task=None,
                weights=None, **kwargs):
    """DSC 통합 진입점 — tabular/image/text 모두 처리.

    호출 패턴:
      tabular: compute_dsc(df=df) 또는 compute_dsc(df)
      image:   compute_dsc(images=imgs, labels=lbs, data_type='image')
               compute_dsc((imgs, lbs))
      text:    compute_dsc(texts=txs, labels=lbs, data_type='text')             ← 분류
               compute_dsc(texts=txs, targets=ts, data_type='text', task='regression')
               compute_dsc((txs, lbs))  ← tuple, 첫 원소가 str이면 'text' 감지

    data_type=None이면 detect_data_type(input_obj)로 자동 추정.
    task=None이면:
      tabular: column_detection.auto_detect_columns로 추정
      image:   'classification' 폴백 (ADR-014)
      text:    'classification' 폴백 (ADR-016). 회귀는 task='regression' 명시.

    Returns:
        cell 결과 dict + 'task'·'data_type' 키 추가.
    """
    # 1) input_obj/df 정리
    if input_obj is None and df is not None:
        input_obj = df

    # 2) data_type 결정
    if data_type is None:
        if images is not None:
            data_type = 'image'
        elif texts is not None:
            data_type = 'text'
        elif input_obj is not None:
            data_type = detect_data_type(input_obj)
            if data_type == 'unknown':
                raise ValueError(
                    "data_type 자동 감지 실패. data_type='tabular'/'image'/'text'를 명시하세요.")
        else:
            raise ValueError("input_obj/df, images, 또는 texts를 제공해야 함.")

    # 3) data_type별 분기
    if data_type == 'tabular':
        # input_obj가 (images, labels) tuple로 잘못 들어왔는지 검증
        df_in = input_obj if input_obj is not None else df
        if target_col is None or numerical_cols is None or categorical_cols is None or task is None:
            auto_target, auto_num, auto_cat, auto_task = auto_detect_columns(
                df_in, target_col=target_col, task=task)
            target_col = target_col or auto_target
            numerical_cols = numerical_cols if numerical_cols is not None else auto_num
            categorical_cols = categorical_cols if categorical_cols is not None else auto_cat
            task = task or auto_task

        profile = select_profile(('tabular', task))
        result = profile['compute_fn'](
            df_in, target_col, numerical_cols, categorical_cols,
            weights=weights, **kwargs)
        result['task'] = task
        result['data_type'] = 'tabular'
        return result

    elif data_type == 'image':
        # images/labels(또는 targets) 추출
        if images is None:
            if isinstance(input_obj, tuple) and len(input_obj) == 2:
                images, second = input_obj
                if labels is None and targets is None:
                    if task == 'regression':
                        targets = second
                    else:
                        labels = second
            elif hasattr(input_obj, '__len__'):
                # PyTorch Dataset 가정 — (img, label) 튜플
                images = []
                labels = []
                for item in input_obj:
                    if isinstance(item, tuple) and len(item) == 2:
                        images.append(item[0]); labels.append(item[1])
                    else:
                        images.append(item)
            else:
                raise ValueError("images=... 명시 필요.")

        if task is None:
            # targets만 주어지면 회귀, 아니면 분류 폴백 (ADR-014)
            task = 'regression' if (targets is not None and labels is None) else 'classification'

        profile = select_profile(('image', task))
        if task == 'regression':
            t = targets if targets is not None else labels
            result = profile['compute_fn'](images, t, weights=weights, **kwargs)
        else:
            result = profile['compute_fn'](images, labels, weights=weights, **kwargs)
        result['task'] = task
        result['data_type'] = 'image'
        return result

    elif data_type == 'text':
        # texts/labels(또는 targets) 추출
        if texts is None:
            if isinstance(input_obj, tuple) and len(input_obj) == 2:
                texts, second = input_obj
                if labels is None and targets is None:
                    # task에 따라 두 번째 원소 해석
                    if task == 'regression':
                        targets = second
                    else:
                        labels = second
            elif hasattr(input_obj, '__len__') and len(input_obj) > 0 \
                    and isinstance(input_obj[0], str):
                texts = list(input_obj)
            else:
                raise ValueError("texts=... 또는 (texts, labels/targets) tuple이 필요.")

        if task is None:
            task = 'regression' if (targets is not None and labels is None) else 'classification'

        profile = select_profile(('text', task))
        if task == 'regression':
            t = targets if targets is not None else labels
            result = profile['compute_fn'](texts, t, weights=weights, **kwargs)
        else:
            result = profile['compute_fn'](texts, labels, weights=weights, **kwargs)
        result['task'] = task
        result['data_type'] = 'text'
        return result

    else:
        raise ValueError(f"data_type '{data_type}' 미지원.")
