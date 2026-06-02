"""성능 측정 — 고정 특징 위 경량 probe 모델 (ADR-019).

모달리티 무관: 특징 행렬(정형 컬럼 / ResNet18·DistilBERT 임베딩)만 받아
task별 표준 경량 모델을 fit → held-out(clean test) 특징으로 평가.

evaluate_probes(X_tr, y_tr, X_te, y_te, task) -> {model_name: score}
  - regression: R² (음수 clip 0)
  - classification: F1(macro)

정형 분류 cell(r=0.598)·DQ4AI와 동일한 "고정 특징 + 표준 모델" 프로토콜을
이미지·텍스트 임베딩으로 확장. finetune이 아니라 probe라 건당 <1초(CPU).
"""
from __future__ import annotations

import numpy as np


REGRESSION_PROBES = ('ridge', 'random_forest', 'mlp', 'knn')
CLASSIFICATION_PROBES = ('logreg', 'random_forest', 'mlp', 'knn')


def _build(task, name, random_state=42):
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.neural_network import MLPRegressor, MLPClassifier
    from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

    if task == 'regression':
        return {
            'ridge': Ridge(alpha=1.0),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1),
            'mlp': MLPRegressor(hidden_layer_sizes=(128,), max_iter=300, random_state=random_state),
            'knn': KNeighborsRegressor(n_neighbors=10),
        }[name]
    return {
        'logreg': LogisticRegression(max_iter=1000),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1),
        'mlp': MLPClassifier(hidden_layer_sizes=(128,), max_iter=300, random_state=random_state),
        'knn': KNeighborsClassifier(n_neighbors=10),
    }[name]


def evaluate_probes(X_tr, y_tr, X_te, y_te, task, models=None, random_state=42):
    """고정 특징 위 probe 모델들을 학습→clean test 평가.

    Args:
        X_tr, X_te: 특징 행렬 (n, d) — 임베딩 또는 컬럼
        y_tr, y_te: target (회귀 float / 분류 int)
        task: 'regression' | 'classification'
        models: 모델 이름 튜플 (None → task 기본 4종)
    Returns:
        {model_name: score}  (회귀 R² clip0 / 분류 F1 macro). 실패 모델은 건너뜀(경고 키 'error').
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, f1_score

    X_tr = np.asarray(X_tr, dtype=float)
    X_te = np.asarray(X_te, dtype=float)
    y_tr = np.asarray(y_tr)
    y_te = np.asarray(y_te)

    # 선형/MLP/kNN 위해 표준화 (train fit → test transform). RF는 무해.
    scaler = StandardScaler().fit(X_tr)
    Xtr_s, Xte_s = scaler.transform(X_tr), scaler.transform(X_te)

    if models is None:
        models = REGRESSION_PROBES if task == 'regression' else CLASSIFICATION_PROBES

    out = {}
    for name in models:
        try:
            model = _build(task, name, random_state=random_state)
            model.fit(Xtr_s, y_tr.astype(float) if task == 'regression' else y_tr)
            pred = model.predict(Xte_s)
            if task == 'regression':
                out[name] = round(float(max(0.0, r2_score(y_te.astype(float), pred))), 4)
            else:
                out[name] = round(float(f1_score(y_te, pred, average='macro')), 4)
        except Exception as exc:  # noqa: BLE001
            out[name] = None
            out.setdefault('_errors', {})[name] = repr(exc)
    return out


__all__ = ['evaluate_probes', 'REGRESSION_PROBES', 'CLASSIFICATION_PROBES']
