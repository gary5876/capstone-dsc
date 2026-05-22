"""04_scoreboard_image cell 3을 v5 framework 합격 단위 (cell·dataset 분리)에 맞게 패치.

POOLED 가설 1/5 PASS는 v4 잔재. v5는 dataset별 r이 합격 단위 (plan 20260511-01 §1).
"""
from __future__ import annotations

import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "04_scoreboard_image.ipynb"

NEW_CELL3 = """# ============================================================
# 2. 통계 검증 + 가설 판정 (v5 framework: dataset별 합격 단위)
# ============================================================
# plan 20260511-01 §1: 합격 단위는 cell × dataset의 Pearson r.
# POOLED 보고는 보조 (Heinrich 2018 R1: cross-dataset 비교 부정당).

print('=' * 64)
print('[A] dataset별 r — v5 framework 합격 단위')
print('=' * 64)
THR_R = 0.40
dataset_verdict = {}
for ds, sub in merged.groupby('dataset'):
    r, p = pearsonr(sub['dsc_score'], sub['accuracy'])
    rs, ps = spearmanr(sub['dsc_score'], sub['accuracy'])
    pass_r = r >= THR_R
    pass_rho = rs >= THR_R
    dataset_verdict[ds] = {'r': r, 'rho': rs, 'p': p, 'pass_r': pass_r}
    flag = 'PASS' if pass_r else 'FAIL'
    print(f'  {ds:<15} n={len(sub):>3}  r={r:+.4f} p={p:.2e}  '
          f'ρ={rs:+.4f} p_s={ps:.2e}  [{flag}]')

print()
print('=' * 64)
print('[B] POOLED — 보조 (cross-dataset, 합격 단위 아님)')
print('=' * 64)
x = merged['dsc_score'].values; y = merged['accuracy'].values
r_p, p_p = pearsonr(x, y); r_s, p_s = spearmanr(x, y)
print(f'  POOLED n={len(merged)}  Pearson r = {r_p:+.4f}, Spearman ρ = {r_s:+.4f}')

# 비선형 (POOLED, 참고용)
X = merged[['dsc_score']].values; rf_folds = []
for tr, te in KFold(5, shuffle=True, random_state=42).split(X):
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X[tr], y[tr])
    rf_folds.append(r2_score(y[te], rf.predict(X[te])))
print(f'  POOLED 비선형 RF 5-fold R² = {np.mean(rf_folds):.4f} ± {np.std(rf_folds):.4f}')
print('  (R²<0 흔함: 두 dataset 분포 차로 단일 RF 학습 한계, 합격 판정 외)')

print()
print('=' * 64)
print('[C] 모델별 r (dataset×model)')
print('=' * 64)
model_pos = 0; model_n = 0
for (ds, m), sub in merged.groupby(['dataset', 'model']):
    if len(sub) < 2:
        continue
    model_n += 1
    rr, pp = pearsonr(sub['dsc_score'], sub['accuracy'])
    if rr > 0:
        model_pos += 1
    flag = 'PASS' if rr >= THR_R else ('+' if rr > 0 else 'NEG')
    print(f'  {ds:>15} | {m:<12s}  n={len(sub):>2}  r={rr:+.4f} p={pp:.2e}  [{flag}]')
print(f'  ▶ 양의 r: {model_pos}/{model_n}')

print()
print('=' * 64)
print('[D] Polluter hold-out — dataset별')
print('=' * 64)
for ds, sub_ds in merged.groupby('dataset'):
    hold_pass = 0; n_pol = 0
    print(f'\\n  --- {ds} ---')
    for hp in sorted(sub_ds['polluter'].unique()):
        if hp == 'none':
            continue
        n_pol += 1
        sub = sub_ds[sub_ds.polluter != hp]
        if len(sub) < 3:
            continue
        rr, _ = pearsonr(sub['dsc_score'], sub['accuracy'])
        pass_ = rr >= THR_R
        hold_pass += int(pass_)
        print(f'    exclude {hp:<22} r={rr:+.4f}  {"PASS" if pass_ else "FAIL"}')
    print(f'    ▶ {ds} polluter hold-out: {hold_pass}/{n_pol} PASS')

print()
print('=' * 64)
print('[E] 합격 판정 — v5 framework 단위')
print('=' * 64)
ds_pass_count = sum(1 for v in dataset_verdict.values() if v['pass_r'])
ds_total = len(dataset_verdict)
print(f'  dataset r≥{THR_R}      : {ds_pass_count}/{ds_total}  '
      f'{"PASS" if ds_pass_count == ds_total else "PARTIAL"}')
print(f'  모델 양의 r          : {model_pos}/{model_n}  '
      f'{"PASS" if model_pos == model_n else "FAIL"}')
print()
print('  ※ POOLED 가설/비선형 R²는 v4 잔재. v5는 cell·dataset별 r이 합격 단위.')
print('  ※ 정식 held-out (CIFAR-100/SVHN/Caltech-101)에서 1회 측정 필요 — 본 결과는 튜닝-운영 cross-validation.')
"""

nb = json.loads(NB.read_text(encoding="utf-8"))
nb["cells"][3]["source"] = NEW_CELL3.splitlines(keepends=True)
nb["cells"][3]["outputs"] = []
nb["cells"][3]["execution_count"] = None
NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"04 cell 3 patched: {NB}")
