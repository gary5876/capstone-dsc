# DSC 프로젝트 문서 안내

## 문서 구조

```
documents/
├── README.md          ← 이 파일 (문서 구조 안내 + 용어 정의)
├── plans/             # 계획 문서 — 무엇을, 어떻게 할 것인가
├── progress/          # 진행 기록 — 날짜별 작업 내용과 결과
└── decisions/         # 의사결정 기록(ADR) — 왜 A 대신 B를 선택했는가
```

### plans/
프로젝트 시작 전 또는 새 단계 진입 시 작성. 실행 방향을 정의한다.
- 네이밍: `YYYYMMDD-순번-요약.md`
- 수정 시 문서 내용을 직접 업데이트하고, 주요 변경은 decisions/에 기록

### progress/
작업 진행 중 작성. 그날 무엇을 했고, 결과가 어땠는지 기록한다.
- 네이밍: `YYYYMMDD-순번-요약.md`
- 포함할 것: 작업 내용, 결과/산출물, 발견한 문제, 다음 할 일

### decisions/
기술적 선택이 발생할 때 작성. 나중에 "왜 이렇게 했지?"에 대한 답.
- 네이밍: `ADR-순번-제목.md`
- 포맷: 배경 → 선택지 → 결정 → 이유

---

## 용어 정의

| 용어 | 정의 |
|---|---|
| **DSC** | Data Score Card. 데이터 품질을 0~100 점수로 산출하는 체계 |
| **DSC Score** | 7개 품질 지표의 가중 평균 × 100. 범위 0~100 |
| **DSC 등급** | A(90+), B(75-89), C(60-74), D(<60) |
| **DQ4AI** | HPI 연구실의 오픈소스 프로젝트. 데이터 품질 차원별 오염(pollution)을 적용하여 ML 성능 영향을 실험 |
| **Polluter** | DQ4AI에서 제공하는 데이터 오염 도구. 결측치 주입, 중복 생성 등 |
| **Pollution Level** | 오염 강도. 0.0(원본)~1.0(완전 오염) 사이의 비율 |
| **Metis** | HPI 연구실의 오픈소스 데이터 품질 측정 프레임워크. 향후 교차 검증용 |
| **프로파일** | 데이터 용도에 따른 가중치 세트 (ml_training, analytics, exploration, default) |
| **베이스라인** | 오염 없는 원본 데이터에 대한 DSC 점수 및 모델 성능 |

### 품질 지표 7개

| 지표 | 의미 | 가중치(default) |
|---|---|---|
| Completeness | 결측치 없이 데이터가 존재하는 비율 | 0.25 |
| Uniqueness | 중복 없는 행의 비율 | 0.20 |
| Validity | 타입/형식이 올바른 비율 | 0.20 |
| Consistency | 컬럼 간 논리적 규칙을 만족하는 비율 | 0.15 |
| Outlier Ratio | IQR 기준 이상치가 아닌 비율 | 0.10 |
| Class Balance | 타겟 클래스 분포의 균형도 (엔트로피 기반) | 0.05 |
| Feature Correlation | 고상관(>0.95) 피처 쌍이 없는 비율 | 0.05 |

### 데이터셋 3개

| 약칭 | 정식명 | 출처 |
|---|---|---|
| Telco | TelcoCustomerChurn | Kaggle |
| Credit | SouthGermanCredit | UCI #573 |
| Letter | Letter Recognition | UCI #59 |

### 모델 5개

| 약칭 | 정식명 | 계열 |
|---|---|---|
| LR | LogisticRegression | 선형 |
| SVC | Support Vector Classifier (linear) | 커널 |
| RF | RandomForestClassifier | 배깅 |
| XGB | XGBClassifier | 부스팅 |
| MLP | MLPClassifier (5층) | 신경망 |
