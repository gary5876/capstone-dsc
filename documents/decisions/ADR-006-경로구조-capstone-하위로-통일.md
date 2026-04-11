# ADR-006: Google Drive 경로를 capstone/dsc/로 통일

- **일자**: 2026-04-11
- **상태**: 확정

## 배경

노트북 01을 실행한 후 결과 파일이 `G:\내 드라이브\dsc\results\`에 생성되었다.
이는 `BASE = '/content/drive/MyDrive/dsc'`로 설정되어 있었기 때문이다.
그러나 이 프로젝트의 모든 산출물은 `capstone/` 디렉토리 아래에 있어야 한다.

## 선택지

| 선택지 | 장점 | 단점 |
|---|---|---|
| A. 기존 경로 유지 (dsc/) | 수정 불필요 | capstone 프로젝트 구조와 분리됨 |
| **B. capstone/dsc/로 변경** | 프로젝트 구조 통일 | 4개 노트북 전부 수정 필요 |

## 결정

**선택지 B.** 4개 노트북의 `BASE` 경로를 `/content/drive/MyDrive/capstone/dsc`로 변경한다.

## 이유

- 프로젝트 산출물이 한 곳에 모여야 관리와 제출이 용이
- 기존에 잘못된 위치에 생성된 파일(`G:\내 드라이브\dsc\`)은 삭제 가능
- 4개 파일 모두 `BASE` 한 줄만 수정하면 되므로 영향 범위 작음

## 영향받은 파일

- `01_setup_and_baseline.ipynb` (cell-2)
- `02_pollution_and_dsc.ipynb` (cell-2)
- `03_training.ipynb` (cell-2)
- `04_scoreboard.ipynb` (cell-2)
