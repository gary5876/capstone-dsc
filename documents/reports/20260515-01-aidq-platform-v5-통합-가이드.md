# aidq-platform DSC v5 통합 가이드

DSC 측이 웹 백엔드 측에 v5 framework 사용을 위해 제공하는 인터페이스 명세 및 v3.2 대비 변경점 정리.

---

## 1. Scope

DSC 측 책임: `(dataset, data_type, task, weights?)` → 점수 dict.

웹 백엔드 측 책임: MQ/S3/DB/프론트엔드. DSC 측은 엔진 외부 영역에 관여하지 않음.

## 2. v3.2 → v5 변경 narrative

웹 백엔드 측의 현재 의존: `engine/dsc_engine.py`의 `compute_dsc(df, weights=...)` + `auto_detect_columns(df)`. 지표 8종 고정, CSV 전용.

v5 framework 전환 후 인터페이스 차이:

- **진입점**: `compute_dsc`는 그대로지만 `data_type`·`task` 인자가 추가됨. 둘 다 optional. 미지정 시 자동감지.
- **지표 schema**: 고정 8종 → task-conditional. tabular/classification 9종, tabular/regression 9종, image/classification 10종. `value_accuracy` 제거 (ADR-009), `label_consistency`·`feature_informativeness` 신설.
- **기본 가중치**: DSC 측 사전등록 값 (`DEFAULT_WEIGHTS_*`). cell별 합계 1.00. 웹 측이 `weights=None`으로 호출하면 그대로 적용.
- **반환 dict**: `{score, grade, <metric별 값>}`에 `task`, `data_type` 두 키 추가.
- **`auto_detect_columns` 반환**: 3-tuple `(target, num_cols, cat_cols)` → 4-tuple `(target, num_cols, cat_cols, task)`.
- **이미지 입력**: `(images, labels)` 또는 PyTorch Dataset 직접 수용.

## 3. 인터페이스 명세

### 3-0. 호출 모델

DSC 엔진은 **Python 라이브러리**. 자체 서버(HTTP/MQ) 미제공.

호출 경로: 웹 백엔드 측 worker 프로세스가 `dsc_framework` 패키지를 import하여 `compute_dsc(...)`를 함수로 직접 호출. v3.2의 `from dsc_engine import compute_dsc` 구조와 동일한 모델.

표준 흐름 (v3.2 worker.py 패턴 그대로 이어받음):

```
[Spring Boot] ── MQ 진단요청 ──▶ [worker.py (웹 측)]
                                       │
                                       │ ① s3Key로 CSV/이미지 다운로드
                                       │ ② DataFrame 또는 (images, labels) 구성
                                       │ ③ compute_dsc(dataset, data_type, task, weights) 호출 ← DSC 엔진 import 호출
                                       │ ④ 반환 dict를 결과 JSON으로 직렬화
                                       ▼
[Spring Boot] ◀── MQ 결과메시지 ── [worker.py]
```

worker.py 매핑 예시 (참고용, 실제 MQ 메시지 schema는 웹 백엔드 측 결정):

```python
import io
import pandas as pd
from dsc_framework import compute_dsc

def handle_diagnosis_message(msg, s3_client):
    """MQ 메시지 → compute_dsc 호출 → 결과 dict."""
    content = s3_client.get_object(Key=msg['s3Key'])['Body'].read()
    df = pd.read_csv(io.BytesIO(content))

    result = compute_dsc(
        df=df,
        data_type=msg.get('data_type'),  # 없으면 None → DataFrame이므로 'tabular' 자동감지
        task=msg.get('task'),            # 없으면 None → target 컬럼 dtype 기반 자동감지
        weights=msg.get('weights'),      # 없으면 None → cell별 DEFAULT_WEIGHTS_* 사용
    )
    return result  # {'score', 'grade', 'task', 'data_type', <metric...>}
```

이미지 케이스의 dataset 구성은 웹 측 worker가 zip/폴더 압축 풀고 `(images, labels)` tuple로 만든 뒤 `compute_dsc(images=..., labels=..., data_type='image')` 호출. 입력 format은 §3-1 참조.

DSC 엔진이 MQ 메시지 schema·s3 다운로드·결과 직렬화·DB 저장에 관여하지 않음. 위 코드는 v3.2 worker.py가 이미 수행하는 작업 흐름을 v5에 맞춰 인자 두 개(`data_type`, `task`) 추가한 형태.

### 3-1. 통합 진입점

```python
from dsc_framework import compute_dsc

result = compute_dsc(
    input_obj=...,     # pd.DataFrame 또는 (images, labels) tuple
    df=...,            # input_obj와 동의어 (v3.2 호환)
    data_type=None,    # 'tabular' | 'image'. None=자동감지
    task=None,         # 'classification' | 'regression'. None=자동감지
    target_col=None,   # tabular 한정. None=자동감지
    numerical_cols=None,
    categorical_cols=None,
    images=None,       # image 한정 (input_obj와 양자택일)
    labels=None,
    weights=None,      # dict. None=cell별 DEFAULT_WEIGHTS_*
)
```

### 3-2. 반환 schema

공통 키: `score` (float, 0~100), `grade` ('A'/'B'/'C'/'D'), `task`, `data_type`.

추가로 task별 metric 키가 dict 최상위에 펼쳐져 들어감 (v3.2와 동일한 flat 구조 유지).

등급 기준은 v3.2와 동일: ≥90 A, ≥75 B, ≥60 C, 그 외 D.

### 3-3. 자동감지 동작

- `data_type` 미지정: `pandas.DataFrame` → `'tabular'`, PyTorch Dataset 또는 (images, labels) tuple → `'image'`.
- `task` 미지정 (tabular): target이 object/category → classification. integer이고 nunique ≤ 20 → classification. nunique ≤ 2 → classification. 그 외 → regression.
- `task` 미지정 (image): `'classification'`으로 폴백 (ADR-014 사전등록).
- `target_col` 미지정 (tabular): `target`/`label`/`class`/`y`/`Churn`/`default`/`price`/`Price`/`MedHouseVal`/`medv`/`quality`/`Quality` 순으로 컬럼명 후보 검사, 없으면 마지막 컬럼 사용.

## 4. Tabular 지표 schema

모든 metric 값은 0~1 범위. **1 = 좋음** 방향. 종합점수 = Σ(metric × weight) × 100.

### 4-1. 분류 (`task='classification'`)

| metric | 의미 | DEFAULT 가중치 |
|---|---|---:|
| `completeness` | 결측치 비율 (1 - 결측 비율) | 0.20 |
| `uniqueness` | 중복 행 비율 (1 - 중복 비율) | 0.15 |
| `validity` | 타입/형식 유효성 | 0.05 |
| `consistency` | 범주형 표현 일관성 | 0.10 |
| `outlier_ratio` | IQR 기반 **non**-outlier 비율 (이름과 반대 — 값이 클수록 정상) | 0.05 |
| `class_balance` | 클래스 균형 (min ratio / ideal ratio, 값이 클수록 균형) | 0.10 |
| `feature_correlation` | 고상관 피처쌍 비율 (1 - 비율) | 0.05 |
| `label_consistency` | k-NN 라벨 일관성, chance 보정 (ADR-009) | 0.20 |
| `feature_informativeness` | MI(X; Y) / H(Y) (ADR-009) | 0.10 |

합계 1.00.

### 4-2. 회귀 (`task='regression'`)

| metric | 의미 | DEFAULT 가중치 |
|---|---|---:|
| `completeness` | 결측치 비율 | 0.20 |
| `uniqueness` | 중복 행 비율 | 0.15 |
| `validity` | 타입/형식 유효성 | 0.05 |
| `consistency` | 범주형 표현 일관성 | 0.10 |
| `outlier_ratio` | IQR 기반 non-outlier 비율 (값이 클수록 정상) | 0.05 |
| `target_distribution_quality` | target 분포 bin entropy 기반 (ADR-011) | 0.10 |
| `feature_correlation` | 고상관 피처쌍 비율 | 0.05 |
| `target_smoothness` | feature 공간 내 target 평활성 (k-NN MSE 기반) | 0.20 |
| `feature_informativeness_reg` | MI(X; Y) / H(Y_bin) | 0.10 |

합계 1.00.

## 5. Image 지표 schema (`data_type='image', task='classification'`)

| metric | 의미 | DEFAULT 가중치 |
|---|---|---:|
| `completeness_image` | 픽셀 마스킹 비율 평균 (1 - 마스킹 비율) | 0.15 |
| `uniqueness` | perceptual hash 중복 비율 | 0.10 |
| `validity` | load 성공 비율 | 0.05 |
| `consistency` | 색공간/크기 일관성 (mode + size entropy) | 0.05 |
| `outlier_ratio` | mean intensity IQR-based non-outlier 비율 (값이 클수록 정상) | 0.05 |
| `class_balance` | 클래스별 샘플 수 불균형 | 0.10 |
| `feature_correlation` | ResNet18 embedding 간 cosine 상관 | 0.05 |
| `label_consistency` | k-NN embedding 라벨 일관성 (chance 보정) | 0.20 |
| `feature_informativeness` | embedding → label MI / H(Y) | 0.10 |
| `sample_quality_image` | blur(Laplacian variance) + contrast(RMS) 결합 | 0.15 |

합계 1.00.

이미지 cell의 ResNet18 embedding 추출 경로는 torch/torchvision/PIL 의존. `dsc_framework` 패키지 import만으로는 발생하지 않음 (lazy import). 이미지 진단 호출 시점에 한해 의존 활성화.

## 6. 호출 예제 (실제 검증 출력)

### 6-1. 분류 자동감지

```python
import pandas as pd
import numpy as np
from dsc_framework import compute_dsc

rng = np.random.default_rng(0)
df = pd.DataFrame({
    'age': rng.integers(20, 70, 200),
    'income': rng.normal(50000, 12000, 200),
    'region': rng.choice(['A', 'B', 'C'], 200),
    'target': rng.integers(0, 2, 200),
})

result = compute_dsc(df=df)
# {'score': 70.06, 'grade': 'C', 'task': 'classification', 'data_type': 'tabular',
#  'completeness': ..., 'uniqueness': ..., 'validity': ..., 'consistency': ...,
#  'outlier_ratio': ..., 'class_balance': ..., 'feature_correlation': ...,
#  'label_consistency': ..., 'feature_informativeness': ...}
```

### 6-2. 회귀 자동감지

```python
df = pd.DataFrame({
    'feat1': rng.normal(0, 1, 200),
    'feat2': rng.normal(5, 2, 200),
    'cat': rng.choice(['x', 'y'], 200),
    'price': rng.normal(100, 20, 200),
})
result = compute_dsc(df=df)
# {'score': 78.62, 'grade': 'B', 'task': 'regression', 'data_type': 'tabular', ...}
```

### 6-3. 가중치 명시

```python
custom_w = {
    'completeness': 0.20, 'uniqueness': 0.15, 'validity': 0.05,
    'consistency': 0.10, 'outlier_ratio': 0.05, 'class_balance': 0.30,
    'feature_correlation': 0.05, 'label_consistency': 0.10,
    'feature_informativeness': 0.10,
}
# 가중치 dict의 키는 해당 task의 metric 키 집합과 정확히 일치해야 함. 합계 1.00 권장.
result = compute_dsc(df=df, task='classification', weights=custom_w)
```

### 6-4. 이미지

```python
from dsc_framework import compute_dsc
# images: list[np.ndarray] 또는 list[PIL.Image] 또는 list[torch.Tensor]
# labels: list[int]
result = compute_dsc(images=images, labels=labels, data_type='image')
# {'score': ..., 'grade': ..., 'task': 'classification', 'data_type': 'image', ...}
```

## 7. 의존성

`dsc_framework` 패키지 import에 필요한 최소 셋:

- `pandas`
- `numpy`
- `scikit-learn` (mutual_info, NearestNeighbors, LabelEncoder, StandardScaler)
- `scipy`

이미지 진단 호출 시 추가 (lazy):

- `torch`
- `torchvision`
- `Pillow`
- `opencv-python` (blur 지표)

웹 백엔드 측 worker가 이미지 진단을 처리하지 않는 단계에서는 위 4개를 설치하지 않아도 무방.

## 8. 비호환 변경 체크리스트

웹 백엔드 측 작업 영역. DSC 측은 인터페이스만 제공:

- MQ 메시지 schema에 `data_type`, `task` 두 optional 필드 수용. 미지정 시 DSC 측이 자동감지.
- 결과 JSON 파서: v3.2의 `value_accuracy` 키 제거, task별 metric 키셋 분기. `task`·`data_type` 키 신규.
- 프론트엔드 가중치 슬라이더: 8종 고정 → task별 9~10종 동적 schema.
- LLM 가중치 추천 프롬프트: 출력 metric 이름 집합을 task별로 분기.

## 9. 참고

- v5 framework: `dsc/dsc_framework/`
- 사전등록 문서: `dsc/documents/decisions/ADR-009`, `ADR-011`, `ADR-014`, `ADR-015`
- v4 결과 보고 (r=0.598): `dsc/documents/reports/20260427-04-v4-정식결과확정.md`
- 이미지 cell 합격선 보고: `dsc/documents/reports/20260522-01-이미지-cell-합격선-달성-진단.md`
- LLM 가중치 생성기 (옵션): `dsc/dsc_framework/llm_weight_generator.py`. ADR-015 사전등록. 웹 측 LLM 추천 흐름 대체용으로 향후 연동 검토 대상.
