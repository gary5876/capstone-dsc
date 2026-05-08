"""DataFrame에서 target 컬럼, 수치형/범주형 분류, task 자동 감지.

webplatform v3.2의 auto_detect_columns를 v5 framework용으로 확장:
- task('classification' | 'regression') 추가 반환
- target 휴리스틱 강화 (수치형 vs 범주형, cardinality 기반)
"""
import numpy as np


_TARGET_NAME_CANDIDATES = (
    'target', 'label', 'class', 'y',
    'Churn', 'default',
    'price', 'Price', 'MedHouseVal', 'medv',
    'quality', 'Quality',
)


def detect_task(target_series, num_unique_threshold=20):
    """target Series에서 task 추정.

    규칙:
    - dtype이 object/category → classification
    - 정수형이고 nunique <= threshold → classification (예: 0/1, 1~5점)
    - 그 외 수치형 → regression
    """
    if target_series.dtype == 'object' or str(target_series.dtype) == 'category':
        return 'classification'
    nunique = target_series.nunique(dropna=True)
    if np.issubdtype(target_series.dtype, np.integer) and nunique <= num_unique_threshold:
        return 'classification'
    if nunique <= 2:
        return 'classification'
    return 'regression'


def auto_detect_columns(df, target_col=None, task=None,
                        num_unique_threshold=20):
    """target 컬럼·수치형/범주형·task 자동 판별.

    Args:
        df: pandas DataFrame
        target_col: 강제 지정 (None이면 휴리스틱)
        task: 강제 지정 ('classification' | 'regression' | None=auto)
        num_unique_threshold: 정수형 target을 분류로 보는 nunique 상한

    Returns:
        (target_col, numerical_cols, categorical_cols, task)
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if target_col is None:
        for candidate in _TARGET_NAME_CANDIDATES:
            if candidate in df.columns:
                target_col = candidate
                break
        if target_col is None:
            target_col = df.columns[-1]

    if target_col in numerical_cols:
        numerical_cols.remove(target_col)
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)

    if task is None:
        task = detect_task(df[target_col],
                           num_unique_threshold=num_unique_threshold)

    return target_col, numerical_cols, categorical_cols, task
