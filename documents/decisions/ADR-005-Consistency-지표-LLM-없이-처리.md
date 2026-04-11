# ADR-005: Consistency 지표를 LLM 없이 처리

- **일자**: 2026-04-10
- **상태**: 검토 필요 (구현 시 확정)

## 배경

평가지표 가이드(PDF)에서 Consistency 지표는 LLM이 컬럼 간 논리 규칙을 자동 생성하는 구조이다. 그러나 DSC 검증 실험의 목적은 LLM 연동이 아니라 "오염 → DSC 변화 → 성능 변화" 상관관계 입증이다.

## 선택지

| 선택지 | 설명 | 장점 | 단점 |
|---|---|---|---|
| A. 규칙 수동 정의 | 3개 데이터셋에 대해 직접 규칙 작성 | 단순, 비용 없음 | 3개니까 가능하지만 확장성 없음 |
| B. LLM API 호출 | 가이드 원본 방식 그대로 | 가이드와 일치 | API 비용, 복잡도 증가 |
| **C. DQ4AI quality_measure 활용** | ConsistentRepresentationPolluter의 측정 함수 사용 | DQ4AI와 일관, 자동 | 가이드의 규칙 기반과 정의가 다름 |
| D. Consistency skip | 6개 지표로 운영, 가중치 재분배 | 가장 단순 | 지표 하나가 빠짐 |

## 결정

**선택지 C를 우선 시도.** DQ4AI의 ConsistentRepresentationPolluter가 오염 시 `compute_quality_measure()`로 "의미적으로 동일한 값이 다르게 표현된 비율"을 측정한다. 이것을 Consistency 점수로 사용하면 LLM 없이도 일관된 측정이 가능하다.

C가 실험 결과에서 의미 없는 수치를 보이면 A(수동 규칙)로 전환.

## 이유

- DQ4AI polluter가 오염과 측정을 모두 제공 → 오염-측정 간 정합성 보장
- LLM 의존성을 제거하여 실험 재현성 확보
- 검증 실험의 핵심은 "상관관계 입증"이지 "Consistency 측정 방법론의 완성도"가 아님
