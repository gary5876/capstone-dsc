# aidq-platform 시스템 통신 흐름 명세

업로드부터 결과 조회까지 컴포넌트 간 통신 경로·프로토콜·페이로드. DSC 엔진(`dsc_framework`)이 끼어드는 지점과 그 외 transport의 경계를 분리하여 정리.

근거: `dsc/web.pdf` (작성 2026-04-28, 웹 백엔드 측 중간 점검) + 본 가이드 §3-0 (`20260515-01-aidq-platform-v5-통합-가이드.md`).

---

## 1. 전체 토폴로지

```
[React]  ─HTTPS REST─  [Spring Boot]  ─AMQP─  [RabbitMQ]  ─AMQP─  [Python worker]
                            │                                          │
                            ├── MySQL (메타데이터)                     ├── S3 (파일 다운로드)
                            ├── S3 (파일 저장)                         │
                            └── Groq / Claude API (LLM)                └── compute_dsc()  ← DSC 측
                                                                            (in-process Python)
```

5개 컴포넌트, 4종 통신:

| 통신 | 종류 |
|---|---|
| React ↔ Spring Boot | HTTPS REST + JWT |
| Spring Boot ↔ RabbitMQ ↔ worker | AMQP 0-9-1 |
| Spring Boot·worker → S3 | AWS S3 API (HTTPS) |
| Spring Boot → MySQL | JDBC |
| Spring Boot → Groq/Claude | HTTPS (LLM provider API) |

DSC 엔진은 네트워크 endpoint 미제공. worker.py 프로세스 내부에서 **Python 함수 호출**로 진입.

---

## 2. 단계별 명세

### ① 업로드 — React → Spring Boot

| 항목 | 값 |
|---|---|
| Protocol | HTTPS |
| Endpoint | `POST /api/jobs/submit` (web.pdf 명시) |
| Auth | JWT (Authorization 헤더, 2시간 만료) |
| Body | multipart/form-data: CSV 파일 + 작업이름 + 사용목적 + weights(JSON) |
| Response | `200 { "jobId": <int>, "status": "PENDING" }` (즉시 응답, 진단은 비동기) |

### ② S3 저장 — Spring Boot → S3

| 항목 | 값 |
|---|---|
| Protocol | AWS S3 API (HTTPS) |
| Operation | `PutObject` |
| Key | `uploads/{userId}/{uuid}_{원본파일명}` (web.pdf 명시) |
| Body | CSV 파일 바이트 |

### ③ Job 레코드 — Spring Boot → MySQL

| 항목 | 값 |
|---|---|
| Protocol | JDBC |
| Operation | `INSERT INTO jobs (...) VALUES (..., 'PENDING')` |
| 결과 | Job.id 발급 |

### ④ 진단 요청 발행 — Spring Boot → RabbitMQ

| 항목 | 값 |
|---|---|
| Protocol | AMQP 0-9-1 |
| Queue | `diagnosis.queue` (web.pdf 명시) |
| Direction | Spring Boot publish → worker consume |

Payload (web.pdf 명시 필드 + v5 신규 필드):

```json
{
  "jobId": 1,
  "userId": 42,
  "s3Key": "uploads/42/a1b2c3d4_data.csv",
  "originalFilename": "data.csv",
  "weights": { "completeness": 0.20, "...": "..." },
  "data_type": "tabular",
  "task": "classification"
}
```

- `weights`, `data_type`, `task` 모두 **optional**.
- `weights` 미지정 → `compute_dsc`가 cell별 `DEFAULT_WEIGHTS_*` 사용.
- `data_type` 미지정 → DataFrame이면 `'tabular'` 자동감지.
- `task` 미지정 → target 컬럼 dtype/cardinality로 classification/regression 자동감지.

### ⑤ S3 다운로드 — worker → S3

| 항목 | 값 |
|---|---|
| Protocol | AWS S3 API |
| Operation | `GetObject(Key=s3Key)` |
| 결과 처리 | CSV bytes → `pd.read_csv(io.BytesIO(...))` → DataFrame |

이미지 데이터셋의 경우 worker가 zip/폴더 압축 해제 후 `(images, labels)` tuple 구성. 입력 format은 통합 가이드 §3-1 참조.

### ⑥ DSC 엔진 호출 — worker.py 내부 (네트워크 없음)

**DSC 측 책임 구간.** worker.py 안의 in-process Python 함수 호출.

```python
from dsc_framework import compute_dsc

result = compute_dsc(
    df=df,                              # 또는 images=..., labels=...
    data_type=msg.get('data_type'),     # MQ 메시지 필드 그대로
    task=msg.get('task'),               # 미지정 시 자동감지
    weights=msg.get('weights'),         # 미지정 시 DEFAULT_WEIGHTS_* 적용
)
# result = {
#     'score': 70.06, 'grade': 'C',
#     'task': 'classification', 'data_type': 'tabular',
#     'completeness': 0.95, 'uniqueness': 0.88, ...,      # 평평 키 (v3.2 호환)
#     'metrics': {'completeness': 0.95, ...},             # 동일 내용, cell 무관 순회용
# }
```

특징:
- 네트워크 transport 없음. Endpoint 없음.
- 호출 실패 시 Python exception으로만 전파 → worker가 catch하여 ⑦에서 `success=false` 메시지로 변환.
- LLM 호출 없음 (결정론적).

### ⑦ 결과 발행 — worker → RabbitMQ

| 항목 | 값 |
|---|---|
| Protocol | AMQP |
| Queue | `result.queue` (web.pdf 명시) |
| Direction | worker publish → Spring Boot consume |

Payload (성공, web.pdf 명시):

```json
{
  "jobId": 1,
  "success": true,
  "dataType": "STRUCTURED",
  "totalScore": 85.50,
  "resultDetail": "<JSON string>",
  "errorMessage": null
}
```

Payload (실패):

```json
{ "jobId": 1, "success": false, "errorMessage": "..." }
```

`resultDetail`은 `compute_dsc` 반환 dict의 JSON 직렬화. v3.2 호환을 위해 `metrics`/`columns`/`summary` 등으로 재포장할지는 웹 백엔드 측 결정.

### ⑧ 결과 수신 — Spring Boot `DiagnosisResultListener`

| 항목 | 값 |
|---|---|
| Subscribe | `result.queue` |

동작 (web.pdf 명시):

1. S3 `PutObject` → `results/{jobId}/diagnosis_result.json`
2. MySQL `INSERT INTO job_results (jobId, totalScore, resultS3Key, ...)`
3. `UPDATE jobs SET dataType=?, status='DONE' WHERE id=?` (실패 시 `FAILED`)
4. LLM 리포트 생성 트리거 (⑨)

### ⑨ LLM 리포트 생성 — Spring Boot → Groq/Claude

| 항목 | 값 |
|---|---|
| Protocol | HTTPS |
| Endpoint | Groq: `https://api.groq.com/openai/v1/chat/completions` (현재 운영). Claude 전환 예정 (web.pdf §5) |
| 요청 | 진단 결과 JSON + 사용목적 → 한국어 리포트 |
| 응답 처리 | S3 `PutObject` → `reports/{jobId}/llm_report.md` |
| 실패 격리 | LLM 실패해도 진단 결과 정상 유지 (try-catch, web.pdf 명시) |

### ⑩ 폴링·결과 조회 — React → Spring Boot

기능명은 web.pdf §3에 명시. 정확한 endpoint path는 웹 백엔드 측 코드 확인 필요.

| 시점 | 기능 | 동작 |
|---|---|---|
| 3초마다 (진행 중) | "작업 상태/목록 조회" | Job.status 확인 → `DONE`이면 클릭 가능 |
| `DONE` 클릭 시 | "진단 결과 조회" | S3 `results/{jobId}/diagnosis_result.json` 읽어서 응답 |
| 리포트 탭 | "LLM 리포트 조회" | S3 `reports/{jobId}/llm_report.md` 읽어서 응답 |

---

## 3. LLM 호출 발생 지점

web.pdf 읽을 때 혼동되기 쉬운 부분. 다음 두 시점에서만 LLM 호출 발생. **Worker도 DSC 엔진도 LLM을 부르지 않음.**

| 시점 | 호출 주체 | 목적 |
|---|---|---|
| ⓐ 업로드 직전 | Spring Boot (별도 endpoint, web.pdf의 "LLM 가중치 추천") | 사용목적 텍스트 → 가중치 추천 |
| ⓑ 진단 완료 후 | Spring Boot (`DiagnosisResultListener` 내부, 단계 ⑨) | 진단 결과 + 사용목적 → 한국어 리포트 |

DSC 엔진은 결정론적이며 LLM 호출 없음. (단, `dsc_framework/llm_weight_generator.py`는 ADR-015용 별도 옵션 모듈로, 현재 웹 통합 경로에 미사용. 향후 ⓐ 대체 검토 대상.)

---

## 4. DSC 측 경계선

```
─────────────────────  네트워크 경계  ─────────────────────
[웹 측이 책임지는 모든 transport]
                  │
                  │ worker.py가 메시지 수신 → DataFrame/images 구성
                  ▼
        from dsc_framework import compute_dsc
        result = compute_dsc(df, data_type, task, weights)  ← DSC 측 책임
        # 100% Python in-process, 네트워크 없음
                  │
                  ▼
[웹 측이 result를 직렬화·발행·저장]
```

**DSC 측 책임:** `compute_dsc(...)`의 입력 contract·계산·반환 dict shape.

**DSC 측 비책임:** MQ schema·S3 키 규칙·REST endpoint·JWT·LLM 호출·DB 스키마·자동 폴링·결과 조회 응답 포맷.

이 경계가 명확해야 엔진 교체/스케일아웃이 다른 부분에 영향 없음. web.pdf "설계 핵심"의 입장과 동일.

---

## 5. 비동기 경계와 실패 격리

전체 흐름에 비동기 hop이 두 곳:

| 경계 | 동기/비동기 | 실패 격리 |
|---|---|---|
| ① → ② → ③ → ④ → ⑤ (업로드 응답까지) | 동기 (사용자가 응답 대기) | S3/MySQL/MQ 실패 시 즉시 HTTP 에러 반환 |
| ④ ⟶ ⑥ ⟶ ⑦ (worker 처리) | 비동기 | worker exception → ⑦ `success=false` 메시지로 변환, Job.status `FAILED` |
| ⑦ ⟶ ⑧ ⟶ ⑨ (리포트 생성) | 비동기 | ⑨ LLM 실패해도 ⑧ 진단 결과는 정상 (try-catch) |

worker의 in-process DSC 엔진 호출(⑥)에서 발생한 exception은 worker 프로세스 외부로 빠져나가지 않음. worker가 catch하여 ⑦의 실패 메시지로 직렬화하는 것이 책임 분담.

---

## 6. 참고

- 통합 가이드 (인터페이스 명세·지표 schema·호출 예제): [`20260515-01-aidq-platform-v5-통합-가이드.md`](20260515-01-aidq-platform-v5-통합-가이드.md)
- 웹 백엔드 측 중간 점검 문서: `dsc/web.pdf` (2026-04-28, 웹 백엔드 측 작성)
- 사전등록 ADR: `documents/decisions/ADR-009`, `ADR-011`, `ADR-014`, `ADR-015`
