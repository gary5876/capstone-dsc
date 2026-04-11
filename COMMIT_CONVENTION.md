# Commit Convention

## 형식

```
<type>(<scope>): <subject>

<body>       ← 선택
<footer>     ← 선택
```

## Type

| type | 용도 |
|---|---|
| `feat` | 새로운 기능 추가 (노트북, 스크립트, 모듈 등) |
| `fix` | 버그 수정 |
| `data` | 데이터 관련 변경 (수집, 전처리, 오염 데이터 생성) |
| `exp` | 실험 실행 및 결과 기록 |
| `docs` | 문서 추가/수정 (plans, decisions, progress 등) |
| `refactor` | 코드 리팩토링 (기능 변경 없음) |
| `chore` | 프로젝트 설정, 환경 구성, .gitignore 등 |
| `style` | 포맷팅, 세미콜론 등 코드 스타일 변경 |

## Scope (선택)

프로젝트 내 영역을 명시한다.

| scope | 대상 |
|---|---|
| `notebook` | notebooks/ 내 Jupyter 노트북 |
| `result` | results/ 내 실험 결과 파일 |
| `doc` | documents/ 내 문서 |
| `config` | 프로젝트 설정 파일 |

## Subject 규칙

- 한국어 또는 영어 사용 (프로젝트 내 혼용 허용)
- 명령형으로 작성: "추가", "수정", "삭제" (O) / "추가함", "수정했음" (X)
- 마침표 없이 끝낸다
- 50자 이내

## 예시

```
chore(config): 프로젝트 초기 설정 및 .gitignore 구성

docs(doc): DSC 검증 실험 마스터플랜 및 ADR 문서 추가

feat(notebook): 베이스라인 측정 노트북 작성

data: 3개 데이터셋 원본 수집

exp(result): DSC 점수 및 모델 성능 결과 기록
```
