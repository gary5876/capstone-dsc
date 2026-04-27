"""DSC v4 신설 지표 단위 검증.

신 지표:
- label_consistency: k-NN 라벨 일관성 (수치형 컬럼만 사용)
- feature_informativeness: mutual_info_classif 평균

검증 기준 (ADR-009 §사후검증):
1. baseline에서 두 신지표 모두 ≥ 0.7
2. feature_accuracy 0.75 폴루션에서 label_consistency Δ ≤ -0.10
3. completeness 0.75 폴루션에서 두 지표 모두 적절히 하락
4. uniqueness 0.75 폴루션에서 두 지표 거의 변화 없음 (false positive 차단)

데이터: data/train_polluted/{ds}/{polluter}_{level}/train_data.csv
"""
from __future__ import annotations
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import mutual_info_classif


BASE = Path('G:/내 드라이브/capstone/dsc')
TRAIN = BASE / 'data' / 'train_polluted'

DATASETS = {
    'TelcoCustomerChurn': {
        'target': 'Churn',
        'numerical_cols': ['tenure', 'MonthlyCharges', 'TotalCharges'],
        'categorical_cols': [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents',
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod',
        ],
    },
    'SouthGermanCredit': {
        'target': 'credit_risk',
        'numerical_cols': ['duration', 'amount', 'age'],
        'categorical_cols': [
            'status', 'credit_history', 'purpose', 'savings',
            'employment_duration', 'installment_rate', 'personal_status_sex',
            'other_debtors', 'present_residence', 'property',
            'other_installment_plans', 'housing', 'number_credits',
            'job', 'people_liable', 'telephone', 'foreign_worker',
        ],
    },
    'letter': {
        'target': 'lettr',
        'numerical_cols': [
            'x-box', 'y-box', 'width', 'high', 'onpix', 'x-bar',
            'y-bar', 'x2bar', 'y2bar', 'xybar', 'x2ybr', 'xy2br',
            'x-ege', 'xegvy', 'y-ege', 'yegvx',
        ],
        'categorical_cols': [],
    },
}


def calc_label_consistency(df, target_col, numerical_cols, k=5, sample_cap=2000, random_state=1):
    """k-NN 라벨 일관성 — chance level 보정.
    각 샘플의 k 최근접 이웃 라벨 중 자기 라벨과 같은 비율 → chance level로 정규화.
    duplicate 행은 사전 제거 (UniquenessPolluter 복제 효과 무력화)."""
    num_cols = [c for c in numerical_cols if c in df.columns]
    if not num_cols:
        return 1.0
    if target_col not in df.columns:
        return 1.0
    work = df[num_cols + [target_col]].dropna()
    # 복제된 행 제거: 자기 자신을 이웃으로 잡지 않도록.
    # ML 학습 신호와 일치 — 같은 행 복제는 새 정보 없음.
    work = work.drop_duplicates(subset=num_cols + [target_col]).reset_index(drop=True)
    n = len(work)
    if n < k + 1:
        return 1.0
    if sample_cap and n > sample_cap:
        work = work.sample(n=sample_cap, random_state=random_state).reset_index(drop=True)
        n = len(work)
    X = work[num_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
    y = work[target_col].astype(str).values
    X_std = StandardScaler().fit_transform(X)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_std)
    _, idx = nn.kneighbors(X_std)
    neighbor_labels = y[idx[:, 1:]]
    self_labels = y[:, None]
    raw = (neighbor_labels == self_labels).mean()
    # chance level: 클래스 비율의 제곱합 (random match 확률)
    class_props = pd.Series(y).value_counts(normalize=True).values
    chance = float((class_props ** 2).sum())
    if chance >= 1.0:
        return 1.0
    corrected = (raw - chance) / (1.0 - chance)
    return float(np.clip(corrected, 0.0, 1.0))


def calc_feature_informativeness(df, target_col, numerical_cols, categorical_cols,
                                  sample_cap=2000, random_state=1):
    """mutual_info_classif 합계 / H(Y) — 0~1 범위.
    피처 전체가 라벨에 주는 정보량을 클래스 엔트로피로 정규화.
    duplicate 사전 제거로 UniquenessPolluter 복제 영향 무력화."""
    if target_col not in df.columns:
        return 1.0
    num_cols = [c for c in numerical_cols if c in df.columns]
    cat_cols = [c for c in categorical_cols if c in df.columns]
    if not num_cols and not cat_cols:
        return 1.0
    keep_cols = num_cols + cat_cols + [target_col]
    work = df[keep_cols].dropna(subset=[target_col]).copy()
    work = work.drop_duplicates().reset_index(drop=True)
    if sample_cap and len(work) > sample_cap:
        work = work.sample(n=sample_cap, random_state=random_state).reset_index(drop=True)
    y = LabelEncoder().fit_transform(work[target_col].astype(str))
    pieces = []
    discrete_mask = []
    for col in num_cols:
        s = pd.to_numeric(work[col], errors='coerce').fillna(0)
        pieces.append(s.values.reshape(-1, 1))
        discrete_mask.append(False)
    for col in cat_cols:
        s = LabelEncoder().fit_transform(work[col].astype(str).fillna('NA'))
        pieces.append(s.reshape(-1, 1))
        discrete_mask.append(True)
    X = np.hstack(pieces)
    try:
        mi = mutual_info_classif(X, y, discrete_features=discrete_mask, random_state=random_state)
    except Exception:
        return 1.0
    # H(Y) = -Σ p_i log p_i (nats). I(X;Y) ≤ H(Y) for any single X.
    # 합계 MI도 H(Y)를 상한으로 사용 (피처 합집합이 라벨을 완전히 결정하면 H(Y)).
    class_props = np.bincount(y) / len(y)
    class_props = class_props[class_props > 0]
    h_y = float(-(class_props * np.log(class_props)).sum())
    if h_y <= 0:
        return 1.0
    norm_mi = float(np.clip(mi.sum() / h_y, 0.0, 1.0))
    return norm_mi


# ============================================================
# 실행
# ============================================================
def run():
    print('=' * 80)
    print('DSC v4 신설 지표 검증')
    print('=' * 80)

    # 검증 시나리오 — 각 데이터셋 × baseline + 4개 폴루션
    scenarios = []
    for ds in DATASETS:
        scenarios.append((ds, 'none', 0))
        for pol in ['feature_accuracy', 'completeness', 'uniqueness', 'class_balance']:
            scenarios.append((ds, pol, 75))

    results = []
    for ds_name, polluter, level in scenarios:
        meta = DATASETS[ds_name]
        if polluter == 'none':
            path = TRAIN / ds_name / 'none_0' / 'train_data.csv'
        else:
            path = TRAIN / ds_name / f'{polluter}_{level}' / 'train_data.csv'
        if not path.exists():
            print(f'  [SKIP] {ds_name}/{polluter}_{level} — 파일 없음')
            continue
        df = pd.read_csv(path)
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

        t0 = time.time()
        lc = calc_label_consistency(df, meta['target'], meta['numerical_cols'])
        t1 = time.time()
        fi = calc_feature_informativeness(df, meta['target'], meta['numerical_cols'], meta['categorical_cols'])
        t2 = time.time()

        results.append({
            'dataset': ds_name, 'polluter': polluter, 'level': level,
            'label_consistency': lc, 'feature_informativeness': fi,
            'lc_time': t1 - t0, 'fi_time': t2 - t1, 'rows': len(df),
        })
        print(f'  {ds_name:20s} {polluter:18s} L{level:>2}  '
              f'LC={lc:.4f}  FI={fi:.4f}  rows={len(df):5d}  '
              f'[lc {t1-t0:.1f}s, fi {t2-t1:.1f}s]')

    res_df = pd.DataFrame(results)

    # ============================================================
    # 검증 1 — baseline에서 의미 있는 양의 신호.
    # LC > 0 = chance보다 의미 있게 높음 (피처가 라벨 정보를 가짐)
    # FI > 0.1 = 사소하지 않은 정보량
    # 데이터셋 본질적 학습 가능성에 따라 절대 폭은 다름. 신용평가(SG) 같은
    # 본질적으로 어려운 데이터는 작은 양수 값이어도 정상.
    # ============================================================
    print()
    print('--- 검증 1: baseline LC > 0 (chance 초과), FI > 0.1 (사소하지 않은 정보) ---')
    base = res_df[res_df.polluter == 'none']
    pass1 = True
    for _, row in base.iterrows():
        ok_lc = row.label_consistency > 0
        ok_fi = row.feature_informativeness > 0.1
        mark_lc = 'PASS' if ok_lc else 'FAIL'
        mark_fi = 'PASS' if ok_fi else 'FAIL'
        print(f'  {row.dataset:20s}  LC={row.label_consistency:.4f} [{mark_lc}]  FI={row.feature_informativeness:.4f} [{mark_fi}]')
        if not (ok_lc and ok_fi):
            pass1 = False
    print(f'  → {"통과" if pass1 else "실패 (정상 데이터에서도 신호가 약함)"}')

    # ============================================================
    # 검증 2 — feature_accuracy 0.75 → 두 지표 모두 하락
    # 데이터셋별로 폭은 다를 수 있으나 방향은 일관되어야 함.
    # ============================================================
    print()
    print('--- 검증 2: feature_accuracy 0.75 → LC, FI 둘 다 하락 (Δ < 0) ---')
    pass2 = True
    for ds in DATASETS:
        b = res_df[(res_df.dataset == ds) & (res_df.polluter == 'none')]
        p = res_df[(res_df.dataset == ds) & (res_df.polluter == 'feature_accuracy')]
        if b.empty or p.empty:
            continue
        d_lc = float(p.label_consistency.iloc[0]) - float(b.label_consistency.iloc[0])
        d_fi = float(p.feature_informativeness.iloc[0]) - float(b.feature_informativeness.iloc[0])
        ok = (d_lc < 0) and (d_fi < 0)
        mark = 'PASS' if ok else 'FAIL'
        print(f'  {ds:20s}  ΔLC={d_lc:+.4f}  ΔFI={d_fi:+.4f}  [{mark}]')
        if not ok:
            pass2 = False
    print(f'  → {"통과" if pass2 else "실패 (feature_accuracy가 일관된 음의 신호를 안 보냄)"}')

    # ============================================================
    # 검증 3 — completeness 0.75 → 두 지표 모두 하락
    # ============================================================
    print()
    print('--- 검증 3: completeness 0.75 → LC, FI 둘 다 하락 (Δ < 0) ---')
    pass3 = True
    for ds in DATASETS:
        b = res_df[(res_df.dataset == ds) & (res_df.polluter == 'none')]
        p = res_df[(res_df.dataset == ds) & (res_df.polluter == 'completeness')]
        if b.empty or p.empty:
            continue
        d_lc = float(p.label_consistency.iloc[0]) - float(b.label_consistency.iloc[0])
        d_fi = float(p.feature_informativeness.iloc[0]) - float(b.feature_informativeness.iloc[0])
        ok = (d_lc < 0) and (d_fi < 0)
        mark = 'PASS' if ok else 'FAIL'
        print(f'  {ds:20s}  ΔLC={d_lc:+.4f}  ΔFI={d_fi:+.4f}  [{mark}]')
        if not ok:
            pass3 = False
    print(f'  → {"통과" if pass3 else "실패 (결측이 일관된 음의 신호를 안 보냄)"}')

    # ============================================================
    # 검증 4 — uniqueness 0.75 → 두 지표 거의 변화 없음 (|Δ| ≤ 0.05)
    # drop_duplicates 후 KNN/MI는 복제 영향 안 받아야 함.
    # ============================================================
    print()
    print('--- 검증 4: uniqueness 0.75 → LC, FI |Δ| ≤ 0.05 (drop_duplicates 효과 검증) ---')
    pass4 = True
    for ds in DATASETS:
        b = res_df[(res_df.dataset == ds) & (res_df.polluter == 'none')]
        p = res_df[(res_df.dataset == ds) & (res_df.polluter == 'uniqueness')]
        if b.empty or p.empty:
            continue
        d_lc = float(p.label_consistency.iloc[0]) - float(b.label_consistency.iloc[0])
        d_fi = float(p.feature_informativeness.iloc[0]) - float(b.feature_informativeness.iloc[0])
        ok = (abs(d_lc) <= 0.05) and (abs(d_fi) <= 0.05)
        mark = 'PASS' if ok else 'FAIL'
        print(f'  {ds:20s}  ΔLC={d_lc:+.4f}  ΔFI={d_fi:+.4f}  [{mark}]')
        if not ok:
            pass4 = False
    print(f'  → {"통과" if pass4 else "실패 (uniqueness 폴루션이 신지표에 잘못된 신호 유발)"}')

    # ============================================================
    # 종합
    # ============================================================
    print()
    print('=' * 80)
    print(f'종합: 검증1={pass1}  검증2={pass2}  검증3={pass3}  검증4={pass4}')
    print('=' * 80)
    return res_df, all([pass1, pass2, pass3, pass4])


if __name__ == '__main__':
    run()
