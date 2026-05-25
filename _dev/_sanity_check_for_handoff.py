"""Handoff doc 작성 전 sanity check — compute_dsc 호출 모양 검증."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from dsc_framework import (
    compute_dsc, auto_detect_columns, detect_data_type,
    DEFAULT_WEIGHTS_CLASSIFICATION, DEFAULT_WEIGHTS_REGRESSION,
)

rng = np.random.default_rng(0)
n = 200

# 1) classification
df_cls = pd.DataFrame({
    'age': rng.integers(20, 70, n),
    'income': rng.normal(50000, 12000, n),
    'region': rng.choice(['A', 'B', 'C'], n),
    'target': rng.integers(0, 2, n),
})
print("=== classification (auto) ===")
r = compute_dsc(df=df_cls)
print({k: r[k] for k in ['score', 'grade', 'task', 'data_type']})
print("metrics:", sorted(k for k in r if k not in ('score', 'grade', 'task', 'data_type')))

# 2) regression
df_reg = pd.DataFrame({
    'feat1': rng.normal(0, 1, n),
    'feat2': rng.normal(5, 2, n),
    'cat': rng.choice(['x', 'y'], n),
    'price': rng.normal(100, 20, n),
})
print("\n=== regression (auto) ===")
r = compute_dsc(df=df_reg)
print({k: r[k] for k in ['score', 'grade', 'task', 'data_type']})
print("metrics:", sorted(k for k in r if k not in ('score', 'grade', 'task', 'data_type')))

# 3) classification with explicit task + custom weights
print("\n=== classification (task=classification, custom weights) ===")
custom_w = dict(DEFAULT_WEIGHTS_CLASSIFICATION)
custom_w['class_balance'] = 0.30
# normalize
s = sum(custom_w.values())
custom_w = {k: v / s for k, v in custom_w.items()}
r = compute_dsc(df=df_cls, task='classification', weights=custom_w)
print({k: r[k] for k in ['score', 'grade', 'task', 'data_type']})

# 4) auto_detect_columns return shape
print("\n=== auto_detect_columns ===")
print(auto_detect_columns(df_cls))

# 5) detect_data_type
print("\n=== detect_data_type ===")
print("DataFrame:", detect_data_type(df_cls))

print("\nDEFAULT_WEIGHTS_CLASSIFICATION keys:", sorted(DEFAULT_WEIGHTS_CLASSIFICATION.keys()))
print("sum:", sum(DEFAULT_WEIGHTS_CLASSIFICATION.values()))
print("DEFAULT_WEIGHTS_REGRESSION keys:", sorted(DEFAULT_WEIGHTS_REGRESSION.keys()))
print("sum:", sum(DEFAULT_WEIGHTS_REGRESSION.values()))
