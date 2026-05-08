"""
회귀 cell용 데이터셋 3개를 data/raw/에 저장.

- California Housing: sklearn.datasets.fetch_california_housing → 20,640 × 9
- Bike Sharing hour: UCI dataset 275, hour.csv → 17,379 × ?
- Wine Quality: UCI dataset 186, red+white 통합 → 6,497 × 12

라이선스 (마스터플랜 sect 1-1 참조):
- California Housing: ⚠️ 형식 라이선스 미부착, Pace & Barry 1997 인용 (학술 관행 사용)
- Bike Sharing: ✅ CC BY 4.0
- Wine Quality: ✅ CC BY 4.0 (Cortez et al. 2009)

CLAUDE.md 외부 자원 검증 원칙: 다운로드 URL과 추출된 파일 행수 모두 확인.
"""
import io
import os
import sys
import urllib.request
import zipfile

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
os.makedirs(RAW_DIR, exist_ok=True)


def download_california_housing():
    """sklearn에서 가져와 CSV로 저장."""
    import pandas as pd
    from sklearn.datasets import fetch_california_housing

    print("\n[California Housing]")
    out_path = os.path.join(RAW_DIR, 'california_housing.csv')
    if os.path.exists(out_path):
        df = pd.read_csv(out_path)
        print(f"  이미 존재: {out_path}")
        print(f"  shape: {df.shape}, target=MedHouseVal range=[{df['MedHouseVal'].min():.3f}, {df['MedHouseVal'].max():.3f}]")
        return out_path

    data = fetch_california_housing(as_frame=True)
    df = data.frame
    df.to_csv(out_path, index=False)
    print(f"  저장: {out_path}")
    print(f"  shape: {df.shape}, target=MedHouseVal range=[{df['MedHouseVal'].min():.3f}, {df['MedHouseVal'].max():.3f}]")
    print(f"  인용: Pace & Barry 1997, Statistics and Probability Letters 33:291-297")
    return out_path


def download_bike_sharing():
    """UCI dataset 275 ZIP 다운로드 → hour.csv 추출."""
    import pandas as pd

    print("\n[Bike Sharing]")
    out_path = os.path.join(RAW_DIR, 'bike_sharing_hour.csv')
    if os.path.exists(out_path):
        df = pd.read_csv(out_path)
        print(f"  이미 존재: {out_path}")
        print(f"  shape: {df.shape}")
        return out_path

    url = 'https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip'
    print(f"  다운로드: {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            zip_bytes = resp.read()
    except Exception as e:
        print(f"  ❌ 다운로드 실패: {e}")
        return None

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        print(f"  ZIP 내부 파일: {names}")
        hour_name = next((n for n in names if n.endswith('hour.csv')), None)
        if hour_name is None:
            print(f"  ❌ hour.csv를 찾을 수 없음")
            return None
        with zf.open(hour_name) as f:
            df = pd.read_csv(f)

    df.to_csv(out_path, index=False)
    print(f"  저장: {out_path}")
    print(f"  shape: {df.shape}")
    print(f"  컬럼: {list(df.columns)}")
    print(f"  CC BY 4.0 (UCI dataset 275, DOI 10.24432/C5W894)")
    return out_path


def download_wine_quality():
    """UCI dataset 186 ZIP 다운로드 → red + white 통합."""
    import pandas as pd

    print("\n[Wine Quality]")
    out_path = os.path.join(RAW_DIR, 'wine_quality.csv')
    if os.path.exists(out_path):
        df = pd.read_csv(out_path)
        print(f"  이미 존재: {out_path}")
        print(f"  shape: {df.shape}")
        return out_path

    url = 'https://archive.ics.uci.edu/static/public/186/wine+quality.zip'
    print(f"  다운로드: {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            zip_bytes = resp.read()
    except Exception as e:
        print(f"  ❌ 다운로드 실패: {e}")
        return None

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        print(f"  ZIP 내부 파일: {names}")
        red_name = next((n for n in names if n.endswith('winequality-red.csv')), None)
        white_name = next((n for n in names if n.endswith('winequality-white.csv')), None)
        if red_name is None or white_name is None:
            print(f"  ❌ red/white CSV를 찾을 수 없음")
            return None
        with zf.open(red_name) as f:
            red_df = pd.read_csv(f, sep=';')
        with zf.open(white_name) as f:
            white_df = pd.read_csv(f, sep=';')

    print(f"  red: {red_df.shape}, white: {white_df.shape}")

    # 통합. wine_type 컬럼은 추가하지 않음 (마스터플랜: 11 numerical 0 categorical 유지).
    df = pd.concat([red_df, white_df], ignore_index=True)
    df.to_csv(out_path, index=False)
    print(f"  저장: {out_path}")
    print(f"  통합 shape: {df.shape}")
    print(f"  컬럼: {list(df.columns)}")
    print(f"  CC BY 4.0 (UCI dataset 186, DOI 10.24432/C56S3T, Cortez et al. 2009)")
    return out_path


def main():
    print("=" * 64)
    print("회귀 cell 데이터셋 다운로드")
    print(f"저장 경로: {RAW_DIR}")
    print("=" * 64)

    paths = []
    paths.append(download_california_housing())
    paths.append(download_bike_sharing())
    paths.append(download_wine_quality())

    print("\n" + "=" * 64)
    print("결과 요약")
    print("=" * 64)
    success = [p for p in paths if p is not None]
    print(f"\n성공: {len(success)}/3")
    for p in paths:
        if p:
            print(f"  ✅ {os.path.basename(p)}")
        else:
            print(f"  ❌ (다운로드 실패)")

    if len(success) < 3:
        sys.exit(1)


if __name__ == '__main__':
    main()
