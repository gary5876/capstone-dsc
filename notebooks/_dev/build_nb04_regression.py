"""노트북 04 회귀 버전(04_scoreboard_regression.ipynb) 생성.

회귀 cell의 분석·검증·차트:
1. 환경 + 데이터 로드 (dsc_scores_regression + model_performance_regression)
2. 차트 1: DSC vs R² 산점도
3. 차트 2: 오염 강도별 DSC & R² 라인
4. 차트 3: 오염 차원 × 모델 민감도 히트맵
5. 차트 4: DSC 등급별 R² 박스플롯
6. 차트 5: 단일 오염 시 DSC 지표 변화 레이더
7. 통계 검증: Pearson r, Spearman ρ, 비선형 RF 5-fold R²
8. Polluter hold-out (F1)
9. Degradation index 절대 vs preservation r 비교 (ADR-012)
10. 결과 요약 + 검증 기준 (r ≥ 0.4) 자동 판정

분류 cell의 메타검증 F1~F10 중 회귀에 의미 있는 것만 포함.
"""
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NB_OUT = REPO / 'notebooks' / '04_scoreboard_regression.ipynb'


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)}


def code(text):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {},
            'outputs': [], 'source': text.splitlines(keepends=True)}


CELLS = []

CELLS.append(md("""# 04. Scoreboard & Visualization (Regression Cell)

**Phase 2 검증**: DSC ↔ R² 상관관계 측정 + 메타 검증

검증 기준 (v5 마스터플랜 sect 2.2):
- Pearson r(DSC, R²_clipped) ≥ 0.4
- Polluter hold-out 4/5 PASS
- 모델 5/5 모두 양의 r

ADR-012 보조 지표: absolute r + preservation r (degradation index 기반) 둘 다 보고.

---"""))

CELLS.append(md("""## 0. 환경 + 데이터 로드"""))

CELLS.append(code("""# ============================================================
# 0-1. Drive 마운트 + 결과 로드
# ============================================================
from google.colab import drive
drive.mount('/content/drive')

import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

BASE = '/content/drive/MyDrive/capstone/dsc'
RESULTS_DIR = f'{BASE}/results'
CHARTS_DIR = f'{RESULTS_DIR}/charts_regression'
os.makedirs(CHARTS_DIR, exist_ok=True)

if BASE not in sys.path:
    sys.path.insert(0, BASE)

dsc = pd.read_csv(f'{RESULTS_DIR}/dsc_scores_regression.csv')
perf = pd.read_csv(f'{RESULTS_DIR}/model_performance_regression.csv')
print(f'DSC: {len(dsc)}건, perf: {len(perf)}건')
print()
print('DSC 데이터셋 분포:'); print(dsc['dataset'].value_counts())
print('\\nperf 데이터셋 × 모델:'); print(perf.pivot_table(values='r2_clipped', index='dataset', columns='model', aggfunc='count'))"""))

CELLS.append(code("""# ============================================================
# 0-2. DSC × Model 결합 (key: dataset, polluter, level)
# ============================================================
merged = perf.merge(dsc[['dataset', 'polluter', 'level', 'score', 'grade']],
                    on=['dataset', 'polluter', 'level'],
                    suffixes=('', '_dsc'))
merged = merged.rename(columns={'score': 'dsc_score'})
print(f'merged: {len(merged)}건')
merged.head()"""))

CELLS.append(md("""## 1. 차트 1: DSC vs R² 산점도 (핵심 증거)"""))

CELLS.append(code("""# ============================================================
# 1-1. 산점도: DSC × R² (모델별 색상)
# ============================================================
plt.figure(figsize=(10, 6))
for model_name, sub in merged.groupby('model'):
    plt.scatter(sub['dsc_score'], sub['r2_clipped'], label=model_name, alpha=0.6, s=30)
plt.xlabel('DSC Score (regression cell)')
plt.ylabel('R² (clipped)')
plt.title('DSC ↔ R² 산점도 (회귀 cell)')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{CHARTS_DIR}/01_dsc_vs_r2_scatter.png', dpi=150)
plt.show()"""))

CELLS.append(md("""## 2. 차트 2: 오염 강도별 DSC & R² 라인"""))

CELLS.append(code("""# ============================================================
# 2-1. 오염 강도 × DSC, R² 라인 (데이터셋별 패널)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
datasets = sorted(merged['dataset'].unique())

for ax, ds in zip(axes, datasets):
    sub = merged[merged.dataset == ds]
    # DSC: polluter 평균 (level별)
    dsc_by_level = sub.groupby('level')['dsc_score'].mean()
    r2_by_level = sub.groupby('level')['r2_clipped'].mean()

    ax2 = ax.twinx()
    ax.plot(dsc_by_level.index, dsc_by_level.values, 'o-', color='steelblue', label='DSC')
    ax2.plot(r2_by_level.index, r2_by_level.values, 's-', color='coral', label='R²_clipped')
    ax.set_xlabel('오염 강도')
    ax.set_ylabel('DSC', color='steelblue')
    ax2.set_ylabel('R²_clipped', color='coral')
    ax.set_title(ds)
    ax.grid(True, alpha=0.3)

plt.suptitle('오염 강도별 DSC & R² 동반 하락', fontsize=14)
plt.tight_layout()
plt.savefig(f'{CHARTS_DIR}/02_dsc_r2_by_level.png', dpi=150)
plt.show()"""))

CELLS.append(md("""## 3. 차트 3: 오염 차원 × 모델 민감도 히트맵"""))

CELLS.append(code("""# ============================================================
# 3-1. 폴루터 × 모델 R² 최대 하락폭 히트맵
# ============================================================
baseline_perf = merged[merged.polluter == 'none'].groupby(['dataset', 'model'])['r2_clipped'].first()
heatmap_rows = []
for (ds, model), base in baseline_perf.items():
    sub = merged[(merged.dataset == ds) & (merged.model == model) & (merged.polluter != 'none')]
    for polluter in sub['polluter'].unique():
        worst_r2 = sub[sub.polluter == polluter]['r2_clipped'].min()
        heatmap_rows.append({'dataset': ds, 'model': model, 'polluter': polluter,
                             'r2_drop': base - worst_r2})
hm = pd.DataFrame(heatmap_rows)
hm_pivot = hm.pivot_table(values='r2_drop', index='polluter', columns='model', aggfunc='mean')

plt.figure(figsize=(10, 6))
sns.heatmap(hm_pivot, annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label': 'R² drop'})
plt.title('오염 차원 × 모델 R² 최대 하락폭 (3 데이터셋 평균)')
plt.tight_layout()
plt.savefig(f'{CHARTS_DIR}/03_polluter_model_heatmap.png', dpi=150)
plt.show()"""))

CELLS.append(md("""## 4. 차트 4: DSC 등급별 R² 박스플롯"""))

CELLS.append(code("""# ============================================================
# 4-1. DSC 등급(A/B/C/D) × R² 박스플롯
# ============================================================
plt.figure(figsize=(10, 6))
order = ['A', 'B', 'C', 'D']
sns.boxplot(data=merged, x='grade', y='r2_clipped', order=order, palette='RdYlGn_r')
plt.xlabel('DSC 등급')
plt.ylabel('R²_clipped')
plt.title('DSC 등급별 모델 R² 분포 (회귀 cell)')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{CHARTS_DIR}/04_grade_r2_box.png', dpi=150)
plt.show()

# 등급별 R² 평균
print()
print('등급별 R² 평균:')
print(merged.groupby('grade')['r2_clipped'].agg(['count', 'mean', 'std']).round(4).reindex(order))"""))

CELLS.append(md("""## 5. 차트 5: 단일 오염 (target_distribution_skew) DSC 지표 변화 레이더"""))

CELLS.append(code("""# ============================================================
# 5-1. target_distribution_skew 레벨별 DSC 9지표 레이더
# ============================================================
metric_keys = ['completeness', 'uniqueness', 'validity', 'consistency', 'outlier_ratio',
               'target_distribution_quality', 'feature_correlation',
               'target_smoothness', 'feature_informativeness_reg']

ds_pick = 'CaliforniaHousing'  # 가장 데이터 많은 데이터셋
sub_dsc = dsc[(dsc.dataset == ds_pick) & (dsc.polluter.isin(['none', 'target_distribution_skew']))]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
angles = np.linspace(0, 2 * np.pi, len(metric_keys), endpoint=False)
angles = np.concatenate([angles, [angles[0]]])

for _, row in sub_dsc.iterrows():
    values = [row[k] for k in metric_keys] + [row[metric_keys[0]]]
    label = 'baseline' if row['polluter'] == 'none' else f'skew={row[\"level\"]}'
    ax.plot(angles, values, 'o-', label=label, alpha=0.7)
    ax.fill(angles, values, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metric_keys, fontsize=8)
ax.set_ylim(0, 1)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
ax.set_title(f'{ds_pick} — target_distribution_skew 레벨별 DSC 지표', pad=20)
plt.tight_layout()
plt.savefig(f'{CHARTS_DIR}/05_radar_target_skew.png', dpi=150)
plt.show()"""))

CELLS.append(md("""## 6. 통계적 검증

핵심 가설 검증:
- H1: Pearson r(DSC, R²_clipped) ≥ 0.4
- H2: Spearman ρ ≥ 0.4
- H3: 비선형 RF 5-fold R² > 선형 r²

분류 cell v4 결과 비교 기준:
- Pearson r = 0.598
- Spearman ρ = 0.628
- 비선형 R² = 0.632 ± 0.091"""))

CELLS.append(code("""# ============================================================
# 6-1. Pearson r, Spearman ρ
# ============================================================
from scipy.stats import pearsonr, spearmanr

x = merged['dsc_score'].values
y = merged['r2_clipped'].values

r_p, p_p = pearsonr(x, y)
r_s, p_s = spearmanr(x, y)

print(f'Pearson r  = {r_p:+.4f} (p={p_p:.2e})')
print(f'Spearman ρ = {r_s:+.4f} (p={p_s:.2e})')
print()
PASS_R = r_p >= 0.4
print(f'H1 PASS (r ≥ 0.4): {PASS_R}')"""))

CELLS.append(code("""# ============================================================
# 6-2. 비선형 RF 5-fold R²
# ============================================================
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

X = merged[['dsc_score']].values
y = merged['r2_clipped'].values

rf_r2_folds = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for tr_idx, te_idx in kf.split(X):
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X[tr_idx], y[tr_idx])
    pred = rf.predict(X[te_idx])
    from sklearn.metrics import r2_score
    rf_r2_folds.append(r2_score(y[te_idx], pred))

print(f'비선형 RF 5-fold R² = {np.mean(rf_r2_folds):.4f} ± {np.std(rf_r2_folds):.4f}')
print(f'  fold: {[round(r, 4) for r in rf_r2_folds]}')
print(f'선형 r² = {r_p**2:.4f}')
print(f'비선형 우위: {np.mean(rf_r2_folds) - r_p**2:+.4f}')"""))

CELLS.append(md("""## 7. 모델별 r — DSC가 모델 무관 점수인지 검증"""))

CELLS.append(code("""# ============================================================
# 7-1. 모델 5개 각각의 r(DSC, R²_clipped)
# ============================================================
print('모델별 Pearson r:')
print('-' * 50)
for model_name, sub in merged.groupby('model'):
    r, p = pearsonr(sub['dsc_score'], sub['r2_clipped'])
    print(f'  {model_name:<22s} r={r:+.4f} (p={p:.2e}, n={len(sub)})')

# 검증 기준: 모든 모델에서 양의 r
all_positive = all(pearsonr(sub['dsc_score'], sub['r2_clipped'])[0] > 0
                   for _, sub in merged.groupby('model'))
print()
print(f'H4 PASS (모든 모델 양의 r): {all_positive}')"""))

CELLS.append(md("""## 8. Polluter Hold-out (F1 검증)

각 폴루터를 hold-out으로 빼고도 r ≥ 0.4 유지하는지. 5개 중 4개 이상 PASS 기대."""))

CELLS.append(code("""# ============================================================
# 8-1. Polluter hold-out
# ============================================================
polluters = sorted(merged['polluter'].unique())
print('Polluter hold-out r (해당 폴루터 제외):')
print('-' * 60)
hold_pass = 0
for hp in polluters:
    if hp == 'none':
        continue
    sub = merged[merged.polluter != hp]
    r, p = pearsonr(sub['dsc_score'], sub['r2_clipped'])
    pass_ = r >= 0.4
    hold_pass += int(pass_)
    print(f'  {hp:<28s} r={r:+.4f} (p={p:.2e})  {\"PASS\" if pass_ else \"FAIL\"}')

n_polluters = len([p for p in polluters if p != 'none'])
print()
print(f'F1 검증: {hold_pass}/{n_polluters} PASS')"""))

CELLS.append(md("""## 9. Degradation Index — 절대 vs preservation r 비교 (ADR-012)

데이터셋 별 floor effect 회피용 보조 지표. compute_dsc_degradation 사용."""))

CELLS.append(code("""# ============================================================
# 9-1. Preservation score 계산 + r 비교
# ============================================================
from dsc_framework import compute_dsc_degradation, DEFAULT_WEIGHTS_REGRESSION

# baseline DSC dict per dataset
baseline_dsc_per_ds = {}
metric_full_keys = list(DEFAULT_WEIGHTS_REGRESSION.keys())
for ds, sub in dsc[dsc.polluter == 'none'].groupby('dataset'):
    row = sub.iloc[0]
    baseline_dsc_per_ds[ds] = {k: row[k] for k in metric_full_keys}

# preservation 계산
preserve_rows = []
for _, row in dsc.iterrows():
    if row['polluter'] == 'none':
        preserve_rows.append({**row.to_dict(), 'preservation_score': 100.0})
        continue
    polluted = {k: row[k] for k in metric_full_keys}
    clean = baseline_dsc_per_ds[row['dataset']]
    deg = compute_dsc_degradation(polluted, clean)
    preserve_rows.append({**row.to_dict(), 'preservation_score': deg['preservation_score']})

dsc_deg = pd.DataFrame(preserve_rows)
merged_deg = perf.merge(dsc_deg[['dataset', 'polluter', 'level', 'score', 'preservation_score']],
                        on=['dataset', 'polluter', 'level'])
merged_deg = merged_deg.rename(columns={'score': 'dsc_score'})

r_abs, _ = pearsonr(merged_deg['dsc_score'], merged_deg['r2_clipped'])
r_pres, _ = pearsonr(merged_deg['preservation_score'], merged_deg['r2_clipped'])
print(f'absolute  DSC ↔ R²:           r = {r_abs:+.4f}')
print(f'preservation DSC ↔ R²:        r = {r_pres:+.4f}')
print(f'preservation 우위: {r_pres - r_abs:+.4f}')"""))

CELLS.append(md("""## 10. 결과 요약 + 검증 기준 자동 판정"""))

CELLS.append(code("""# ============================================================
# 10-1. 검증 기준 종합
# ============================================================
summary = {
    'Pearson_r':                round(r_p, 4),
    'Spearman_rho':             round(r_s, 4),
    'r_squared':                round(r_p**2, 4),
    'nonlinear_RF_5fold_R2':    round(np.mean(rf_r2_folds), 4),
    'preservation_r':           round(r_pres, 4),
    'polluter_holdout_pass':    f'{hold_pass}/{n_polluters}',
    'all_models_positive_r':    all_positive,
}

verdict = {
    'H1 r ≥ 0.4':              r_p >= 0.4,
    'H2 ρ ≥ 0.4':              r_s >= 0.4,
    'H3 비선형 우위':            np.mean(rf_r2_folds) > r_p**2,
    'H4 모든 모델 양의 r':       all_positive,
    'H5 polluter hold-out 4/5': hold_pass >= 4,
}

print('=== 회귀 cell Phase 2 검증 결과 ===')
for k, v in summary.items():
    print(f'  {k:<30s} {v}')
print()
print('=== 가설 검증 ===')
for k, v in verdict.items():
    mark = '✅' if v else '❌'
    print(f'  {mark} {k}: {v}')

n_pass = sum(verdict.values())
n_total = len(verdict)
print()
print(f'종합: {n_pass}/{n_total} PASS')
if n_pass == n_total:
    print('🎉 회귀 cell Phase 2 검증 통과 — Phase 3 (framework 통합)으로 진행 가능')
else:
    print('⚠️  실패 가설 있음 — 조사 후 사전등록 가중치·정의식은 변경 금지 (F1 순환 논증 회피)')"""))

CELLS.append(code("""# ============================================================
# 11. 실행 로그 저장
# ============================================================
from datetime import datetime

log_lines = []
log_lines.append('# 노트북 04 회귀 버전 실행 로그 — Scoreboard')
log_lines.append('')
log_lines.append(f'- **실행 시각**: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}')
log_lines.append(f'- **DSC 데이터**: {len(dsc)}건')
log_lines.append(f'- **모델 성능**: {len(perf)}건')
log_lines.append(f'- **merged**: {len(merged)}건')
log_lines.append('')

log_lines.append('## 1. 통계 요약')
log_lines.append('')
for k, v in summary.items():
    log_lines.append(f'- **{k}**: {v}')
log_lines.append('')

log_lines.append('## 2. 가설 검증')
log_lines.append('')
log_lines.append('| 가설 | 결과 |')
log_lines.append('|---|---|')
for k, v in verdict.items():
    log_lines.append(f'| {k} | {\"✅ PASS\" if v else \"❌ FAIL\"} |')
log_lines.append('')

log_lines.append(f'**종합**: {n_pass}/{n_total} PASS')
log_lines.append('')

log_lines.append('## 3. 모델별 r')
log_lines.append('')
log_lines.append('| 모델 | r | n |')
log_lines.append('|---|---|---|')
for model_name, sub in merged.groupby('model'):
    r, _ = pearsonr(sub['dsc_score'], sub['r2_clipped'])
    log_lines.append(f'| {model_name} | {r:+.4f} | {len(sub)} |')
log_lines.append('')

log_lines.append('## 4. Polluter hold-out')
log_lines.append('')
log_lines.append('| Polluter (제외) | r | 결과 |')
log_lines.append('|---|---|---|')
for hp in polluters:
    if hp == 'none':
        continue
    sub = merged[merged.polluter != hp]
    r, _ = pearsonr(sub['dsc_score'], sub['r2_clipped'])
    log_lines.append(f'| {hp} | {r:+.4f} | {\"✅\" if r >= 0.4 else \"❌\"} |')
log_lines.append('')

log_path = f'{RESULTS_DIR}/04_regression_execution_log.md'
with open(log_path, 'w', encoding='utf-8') as f:
    f.write('\\n'.join(log_lines))
print(f'실행 로그 저장: {log_path}')"""))


nb = {
    'cells': CELLS,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.11'},
    },
    'nbformat': 4,
    'nbformat_minor': 4,
}

os.makedirs(NB_OUT.parent, exist_ok=True)
with open(NB_OUT, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

n_md = sum(1 for c in CELLS if c['cell_type'] == 'markdown')
n_code = sum(1 for c in CELLS if c['cell_type'] == 'code')
print(f'생성: {NB_OUT}')
print(f'총 {len(CELLS)} 셀 (markdown {n_md}, code {n_code})')
