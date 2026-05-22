"""이미지 cell r 진단 v2 (epoch=10 pretrained 재학습 완료 후).

1. default 가중치 r (튜닝/held-out/pooled)
2. 차원별 r (개별)
3. dataset별 가중치 튜닝 (CIFAR10 fit → FashionMNIST held-out)
4. polluter hold-out + 모델별 r
5. 합격 조건 비교 (ADR-014/plan 20260511-01: r>=0.40, p<0.001, ρ>=0.4, polluter 4/5 PASS, 모델 양의 r)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

METRICS = [
    'completeness_image', 'uniqueness', 'validity', 'consistency',
    'outlier_ratio', 'class_balance', 'sample_quality_image',
    'feature_correlation', 'label_consistency', 'feature_informativeness',
]
DEFAULT_W = {
    'completeness_image': 0.15, 'uniqueness': 0.10, 'validity': 0.05,
    'consistency': 0.05, 'outlier_ratio': 0.05, 'class_balance': 0.10,
    'feature_correlation': 0.05, 'label_consistency': 0.20,
    'feature_informativeness': 0.10, 'sample_quality_image': 0.15,
}

perf = pd.read_csv(RESULTS / "model_performance_image.csv")
dsc = pd.read_csv(RESULTS / "dsc_scores_image.csv")

# 활성 DATASETS만 + epoch=10
perf = perf[(perf.epochs == 10) & perf.dataset.isin(['CIFAR10', 'FashionMNIST'])]
merged = perf.merge(dsc[['dataset', 'polluter', 'level'] + METRICS],
                    on=['dataset', 'polluter', 'level'])
merged['score_default'] = sum(merged[k] * DEFAULT_W[k] for k in DEFAULT_W) * 100

print("=" * 78)
print("[0] 데이터 요약")
print("=" * 78)
print(merged.groupby(['dataset', 'model']).size().to_string())
print()

print("=" * 78)
print("[1] default 가중치 — Pearson r / Spearman ρ")
print("=" * 78)
THR_R = 0.40

def report(sub, label):
    r, p = stats.pearsonr(sub['score'], sub['accuracy'])
    rs, ps = stats.spearmanr(sub['score'], sub['accuracy'])
    flag = "PASS" if r >= THR_R else "FAIL"
    print(f"  {label:<30} n={len(sub):>3}  r={r:+.4f} p={p:.2e}  "
          f"ρ={rs:+.4f} p_s={ps:.2e}  [{flag}]")
    return r

sub_cifar = merged[merged.dataset == 'CIFAR10']
sub_fmnist = merged[merged.dataset == 'FashionMNIST']
sub_pooled = merged

# score_default를 score로 임시 매핑
for sub in (sub_cifar, sub_fmnist, sub_pooled):
    sub.loc[:, 'score'] = sub['score_default']

r_cifar = report(sub_cifar, "CIFAR10 (튜닝)")
r_fmnist = report(sub_fmnist, "FashionMNIST (held-out)")
r_pooled = report(sub_pooled, "POOLED")
print()

print("=" * 78)
print("[2] 차원별 Pearson r (개별 차원 vs accuracy)")
print("=" * 78)
for ds_label, sub in [("CIFAR10", sub_cifar), ("FashionMNIST", sub_fmnist),
                       ("POOLED", sub_pooled)]:
    print(f"\n--- {ds_label} (n={len(sub)}) ---")
    rows = []
    for col in METRICS:
        x = sub[col].to_numpy()
        if x.std() == 0:
            rows.append({'metric': col, 'r': np.nan, 'p': np.nan, 'std': 0.0})
            continue
        r, p = stats.pearsonr(x, sub['accuracy'])
        rows.append({'metric': col, 'r': r, 'p': p, 'std': float(x.std())})
    print(pd.DataFrame(rows).sort_values('r', ascending=False, na_position='last')
          .round(4).to_string(index=False))
print()

print("=" * 78)
print("[3] 가중치 튜닝 (CIFAR10 fit → FashionMNIST held-out)")
print("=" * 78)
# 살아있는 차원만 튜닝 (constant 차원은 단지 offset이라 r에 영향 X)
LIVE = ['completeness_image', 'class_balance', 'sample_quality_image',
        'label_consistency', 'feature_informativeness']
DEAD = [m for m in METRICS if m not in LIVE]

X_t = sub_cifar[LIVE].to_numpy()
y_t = sub_cifar['accuracy'].to_numpy()
X_h = sub_fmnist[LIVE].to_numpy()
y_h = sub_fmnist['accuracy'].to_numpy()

# softmax-style positive weights summing to 1 over LIVE
def neg_r(w_raw, X, y):
    w = np.maximum(w_raw, 1e-6)
    w = w / w.sum()
    s = X @ w
    if s.std() == 0:
        return 1.0
    return -stats.pearsonr(s, y)[0]

# bound: 가중치 합 1 / WEIGHT_BOUNDS [0.01, 0.60] 준수
from scipy.optimize import minimize
best = None
for trial in range(20):
    rng = np.random.RandomState(trial)
    w0 = rng.dirichlet(np.ones(len(LIVE)))
    res = minimize(neg_r, w0, args=(X_t, y_t), method='Nelder-Mead',
                   options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5})
    if best is None or res.fun < best.fun:
        best = res
w_best = np.maximum(best.x, 1e-6)
w_best = w_best / w_best.sum()
# bound clipping
w_best = np.clip(w_best, 0.01, 0.60)
w_best = w_best / w_best.sum()

# 전체 차원 가중치 dict (dead는 default 유지 후 합 1로 재정규화 안 함 — 단지 비교용)
# 정식: LIVE 가중치 합 = 1 - sum(dead default), dead는 default
dead_sum = sum(DEFAULT_W[d] for d in DEAD)  # 0.30
live_w = {LIVE[i]: float(w_best[i]) * (1 - dead_sum) for i in range(len(LIVE))}
tuned_w = dict(DEFAULT_W)  # dead는 default 그대로
tuned_w.update(live_w)

# r 측정
def score_with(df, w):
    return sum(df[k] * w[k] for k in w) * 100

merged['score_tuned'] = score_with(merged, tuned_w)

print("튜닝 가중치 (LIVE 차원만, dead 차원은 default 유지):")
for k in METRICS:
    flag = "(dead)" if k in DEAD else "(live)"
    print(f"  {k:<26} {tuned_w[k]:.4f} {flag}")
print(f"  sum = {sum(tuned_w.values()):.4f}")
print()

for ds, sub in merged.groupby('dataset'):
    r_d, _ = stats.pearsonr(sub['score_default'], sub['accuracy'])
    r_t, _ = stats.pearsonr(sub['score_tuned'], sub['accuracy'])
    print(f"  {ds:<15} default r={r_d:+.4f}  tuned r={r_t:+.4f}  Δ={r_t-r_d:+.4f}")
r_d, _ = stats.pearsonr(merged['score_default'], merged['accuracy'])
r_t, _ = stats.pearsonr(merged['score_tuned'], merged['accuracy'])
print(f"  {'POOLED':<15} default r={r_d:+.4f}  tuned r={r_t:+.4f}  Δ={r_t-r_d:+.4f}")
print()

print("=" * 78)
print("[4] Polluter hold-out (default 가중치, POOLED)")
print("=" * 78)
hold_pass = 0
n_pol = 0
for hp in sorted(merged['polluter'].unique()):
    if hp == 'none':
        continue
    n_pol += 1
    sub = merged[merged.polluter != hp]
    r, _ = stats.pearsonr(sub['score_default'], sub['accuracy'])
    pass_ = r >= THR_R
    hold_pass += int(pass_)
    print(f"  exclude {hp:<22} r={r:+.4f}  {'PASS' if pass_ else 'FAIL'}")
print(f"\n  ▶ Polluter hold-out: {hold_pass}/{n_pol} PASS  "
      f"({'PASS' if hold_pass >= n_pol - 1 else 'FAIL'} — 4/5 기준)")
print()

print("=" * 78)
print("[5] 모델별 r (default 가중치)")
print("=" * 78)
pos_count = 0
n_models = 0
for (ds, m), sub in merged.groupby(['dataset', 'model']):
    if sub['accuracy'].std() == 0:
        continue
    n_models += 1
    r, p = stats.pearsonr(sub['score_default'], sub['accuracy'])
    if r > 0:
        pos_count += 1
    print(f"  {ds:>15} | {m:<12} n={len(sub):>2}  r={r:+.4f} p={p:.4f}")
print(f"\n  ▶ 양의 r: {pos_count}/{n_models}")

print()
print("=" * 78)
print("[6] 합격 판정 요약")
print("=" * 78)
print(f"  H1  POOLED r ≥ 0.40       : {r_d:+.4f}  {'PASS' if r_d >= THR_R else 'FAIL'}")
r_p_p, p_pp = stats.pearsonr(merged['score_default'], merged['accuracy'])
rs_pp, ps_pp = stats.spearmanr(merged['score_default'], merged['accuracy'])
print(f"  H2  POOLED ρ ≥ 0.40       : {rs_pp:+.4f}  {'PASS' if rs_pp >= THR_R else 'FAIL'}")
print(f"  H3  p < 0.001            : p={p_pp:.2e}  {'PASS' if p_pp < 0.001 else 'FAIL'}")
print(f"  H4  Polluter hold-out 4/5: {hold_pass}/{n_pol}  "
      f"{'PASS' if hold_pass >= n_pol - 1 else 'FAIL'}")
print(f"  H5  모델 모두 양의 r       : {pos_count}/{n_models}  "
      f"{'PASS' if pos_count == n_models else 'FAIL'}")

# 저장
out = {
    'date': '2026-05-22',
    'weights_default': DEFAULT_W,
    'weights_tuned_live': {k: tuned_w[k] for k in LIVE},
    'r_default': {'CIFAR10': r_cifar, 'FashionMNIST': r_fmnist, 'POOLED': r_pooled},
    'r_tuned': {ds: float(stats.pearsonr(s['score_tuned'], s['accuracy'])[0])
                 for ds, s in merged.groupby('dataset')},
    'polluter_holdout_pass': f'{hold_pass}/{n_pol}',
    'model_positive_r': f'{pos_count}/{n_models}',
    'n_total': int(len(merged)),
}
out['r_tuned']['POOLED'] = float(stats.pearsonr(merged['score_tuned'], merged['accuracy'])[0])
(RESULTS / 'image_diagnosis_v2.json').write_text(
    json.dumps(out, ensure_ascii=False, indent=2))
print(f"\n결과 저장: {RESULTS / 'image_diagnosis_v2.json'}")
