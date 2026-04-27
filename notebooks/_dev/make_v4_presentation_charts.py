"""v4 정식 결과로 발표용 차트 5장 생성.

입력:
  - results/dsc_scores.csv (87건, v4 점수)
  - results/model_performance.csv (435건)
  - results/merged_results.csv (435건, v4 머지)
  - notebooks/_dev/simulated_v4_merged.csv (v3.2 시점 데이터 보존: v3.2 비교용)

출력: documents/reports/charts/20260427/*.png

차트 5장:
  1. 산점도 v3.2 vs v4 (좌우 비교)
  2. 등급별 박스플롯 (v4 정식)
  3. Polluter × Dataset r 매트릭스 (v3.2 vs v4)
  4. Polluter Hold-out 5/5 PASS 막대 (v3.2 vs v4)
  5. 선형 r² vs 비선형 R² 비교
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score


rcParams['font.family'] = 'Malgun Gothic'
rcParams['axes.unicode_minus'] = False
rcParams['figure.dpi'] = 110

BASE = Path('G:/내 드라이브/capstone/dsc')
OUT = BASE / 'documents' / 'reports' / 'charts' / '20260427'
OUT.mkdir(parents=True, exist_ok=True)

# v4 정식
m_v4 = pd.read_csv(BASE / 'results' / 'merged_results.csv')
# v3.2 비교용 (시뮬에 보존)
m_v32 = pd.read_csv(BASE / 'notebooks' / '_dev' / 'simulated_v4_merged.csv')
# 잠깐 — simulated_v4_merged는 v4 dsc + v3.2 perf. v3.2 score는 다른 곳에서 가져와야.
# v3.2 시점 정식 결과 백업이 있는지 확인 — 진단보고서의 r=0.42, ANOVA F=32.9 등은 v3.2 실측치.
# 차트 비교용 v3.2는 시뮬에 보존된 score 컬럼이 v4 score이므로 적합치 않음.
# v3.2 결과 재구성: 진단/개선 보고서의 정식 수치 + simulated_v32_scores.csv 활용

# 옵션: simulated_v32_scores.csv (v3 시점 시뮬)와 model_performance를 합쳐 v3.2 정식 재현
m_v32_dsc = pd.read_csv(BASE / 'notebooks' / '_dev' / 'simulated_v32_scores.csv')
m_v32_dsc = m_v32_dsc.rename(columns={'v3.2_score': 'score', 'v3.2_grade': 'grade'})
perf = pd.read_csv(BASE / 'results' / 'model_performance.csv')
m_v32 = perf.merge(m_v32_dsc[['dataset', 'polluter', 'level', 'score', 'grade']],
                   on=['dataset', 'polluter', 'level'], how='inner')
print(f'v3.2 재구성: {len(m_v32)}건, v4 정식: {len(m_v4)}건')


# ============================================================
# 차트 1: DSC vs F1 산점도 (v3.2 vs v4)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
ds_colors = {'TelcoCustomerChurn': '#1f77b4', 'SouthGermanCredit': '#ff7f0e', 'letter': '#2ca02c'}
for ax, data, title in [(axes[0], m_v32, 'v3.2 (이전 정식)'), (axes[1], m_v4, 'v4 (개선 후 정식)')]:
    for ds, c in ds_colors.items():
        sub = data[data['dataset'] == ds]
        ax.scatter(sub['score'], sub['f1_macro'], c=c, alpha=0.55, s=24, label=ds, edgecolors='none')
    if len(data) >= 3:
        slope, intercept, r, p, _ = stats.linregress(data['score'], data['f1_macro'])
        x = np.linspace(data['score'].min(), data['score'].max(), 50)
        ax.plot(x, intercept + slope * x, 'k--', linewidth=1.2, alpha=0.6)
        ax.set_title(f'{title}  (n={len(data)})\nPearson r = {r:.3f}, p = {p:.2e}', fontsize=12)
    ax.set_xlabel('DSC Score', fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc='lower right')
axes[0].set_ylabel('F1 (macro)', fontsize=11)
fig.suptitle('DSC Score ↔ ML 성능 — v3.2 → v4 회복', fontsize=14, y=1.00)
plt.tight_layout()
plt.savefig(OUT / '01_scatter_v32_vs_v4.png', bbox_inches='tight', dpi=130)
plt.close()
print('saved: 01_scatter_v32_vs_v4.png')


# ============================================================
# 차트 2: 등급별 박스플롯 (v4 정식)
# ============================================================
fig, ax = plt.subplots(figsize=(8.5, 5))
order = ['A', 'B', 'C', 'D']
data = [m_v4[m_v4['grade'] == g]['f1_macro'].values for g in order]
counts = [len(d) for d in data]
bp = ax.boxplot(
    data, tick_labels=[f'{g}\n(n={c})' for g, c in zip(order, counts)],
    patch_artist=True, widths=0.55, showmeans=True,
    meanprops=dict(marker='D', markerfacecolor='red', markersize=6),
)
colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
for patch, c in zip(bp['boxes'], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.6)
ax.set_xlabel('DSC 등급', fontsize=11)
ax.set_ylabel('F1 (macro)', fontsize=11)
valid = [d for d in data if len(d) > 0]
f, p = stats.f_oneway(*valid)
means = [d.mean() for d in valid]
ax.set_title(
    f'DSC 등급별 ML 성능 분포 (v4 정식, n={len(m_v4)})\n'
    f'평균 F1: A={means[0]:.3f}, B={means[1]:.3f}, C={means[2]:.3f}, D={means[3]:.3f}  |  '
    f'ANOVA F={f:.2f}, p={p:.2e}',
    fontsize=11,
)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / '02_boxplot_grade_v4.png', bbox_inches='tight', dpi=130)
plt.close()
print('saved: 02_boxplot_grade_v4.png')


# ============================================================
# 차트 3: Polluter × Dataset r 매트릭스 (v3.2 vs v4)
# ============================================================
def slice_r(df):
    rows = []
    for pol in sorted([p for p in df['polluter'].unique() if p != 'none']):
        for ds in sorted(df['dataset'].unique()):
            sub = df[(df['polluter'] == pol) & (df['dataset'] == ds)]
            if len(sub) < 5:
                continue
            try:
                r, _ = stats.pearsonr(sub['score'], sub['f1_macro'])
            except Exception:
                r = np.nan
            rows.append({'polluter': pol, 'dataset': ds, 'r': r})
    return pd.DataFrame(rows)

r32 = slice_r(m_v32).pivot(index='polluter', columns='dataset', values='r')
r4 = slice_r(m_v4).pivot(index='polluter', columns='dataset', values='r')

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, data, title in [(axes[0], r32, 'v3.2 (이전)'), (axes[1], r4, 'v4 (개선)')]:
    im = ax.imshow(data.values, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=20, ha='right', fontsize=10)
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index, fontsize=10)
    for i in range(len(data.index)):
        for j in range(len(data.columns)):
            v = data.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f'{v:+.2f}', ha='center', va='center', fontsize=10,
                        color='white' if abs(v) > 0.5 else 'black', fontweight='bold')
    ax.set_title(title, fontsize=12)
fig.suptitle('Polluter × Dataset 슬라이스 상관계수 — 부호 회복', fontsize=14, y=1.02)
fig.colorbar(im, ax=axes, fraction=0.025, pad=0.04, label='Pearson r')
plt.savefig(OUT / '03_slice_r_matrix.png', bbox_inches='tight', dpi=130)
plt.close()
print('saved: 03_slice_r_matrix.png')


# ============================================================
# 차트 4: Polluter Hold-out 5/5 PASS (v3.2 vs v4)
# ============================================================
def holdout_r(df, polluter):
    sub = df[df['polluter'] == polluter]
    if len(sub) < 10:
        return np.nan, np.nan
    r, p = stats.pearsonr(sub['score'], sub['f1_macro'])
    return r, p

polluters = ['completeness', 'uniqueness', 'consistent_repr', 'class_balance', 'feature_accuracy']
v32_rs = [holdout_r(m_v32, p)[0] for p in polluters]
v4_rs = [holdout_r(m_v4, p)[0] for p in polluters]

fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.arange(len(polluters))
w = 0.38
b1 = ax.bar(x - w/2, v32_rs, w, label='v3.2', color='#7f8c8d', alpha=0.8)
b2 = ax.bar(x + w/2, v4_rs, w, label='v4', color='#27ae60', alpha=0.85)
ax.axhline(0.3, color='red', linestyle='--', linewidth=1, alpha=0.6, label='PASS 기준 (r=0.3)')
ax.axhline(0, color='black', linewidth=0.5)
for bars, vals in [(b1, v32_rs), (b2, v4_rs)]:
    for bar, v in zip(bars, vals):
        if pd.notna(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02 if v > 0 else v - 0.05,
                    f'{v:+.2f}', ha='center', fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(polluters, rotation=15, ha='right', fontsize=10)
ax.set_ylabel('Hold-out Pearson r', fontsize=11)
n_pass_v32 = sum(1 for r in v32_rs if pd.notna(r) and r > 0.3)
n_pass_v4 = sum(1 for r in v4_rs if pd.notna(r) and r > 0.3)
ax.set_title(f'Polluter Hold-out 검증 — 순환 논증 회피 (v3.2: {n_pass_v32}/5  →  v4: {n_pass_v4}/5)',
             fontsize=12)
ax.set_ylim(-0.1, 1.0)
ax.legend(loc='lower right', fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / '04_holdout_passes.png', bbox_inches='tight', dpi=130)
plt.close()
print('saved: 04_holdout_passes.png')


# ============================================================
# 차트 5: 선형 r² vs 비선형 R² (분산 설명력 정직화)
# ============================================================
metric_cols = [
    'completeness', 'uniqueness', 'validity', 'consistency',
    'outlier_ratio', 'class_balance', 'feature_correlation',
    'label_consistency', 'feature_informativeness',
]
metric_cols = [c for c in metric_cols if c in m_v4.columns]
valid = m_v4.dropna(subset=metric_cols + ['f1_macro'])
X = valid[metric_cols].values
y = valid['f1_macro'].values

r_lin, _ = stats.pearsonr(valid['score'], valid['f1_macro'])
r2_lin = r_lin ** 2
rf = RandomForestRegressor(n_estimators=200, random_state=1, n_jobs=-1)
cv = KFold(n_splits=5, shuffle=True, random_state=1)
r2_cv = cross_val_score(rf, X, y, cv=cv, scoring='r2')
r2_nonlin = r2_cv.mean()

fig, ax = plt.subplots(figsize=(8, 5))
labels = ['선형 결합\n(DSC = Σ wᵢ sᵢ)', '비선형 결합\n(RandomForest)']
vals = [r2_lin, r2_nonlin]
errs = [0, r2_cv.std()]
colors = ['#3498db', '#27ae60']
bars = ax.bar(labels, vals, yerr=errs, color=colors, alpha=0.8, capsize=8, width=0.5)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f'{v:.3f}',
            ha='center', fontsize=12, fontweight='bold')
ax.set_ylabel('R² (분산 설명력)', fontsize=11)
ax.set_ylim(0, max(vals) * 1.3)
ax.set_title(f'선형 vs 비선형 분산 설명력 — DSC 차원의 잠재 정보량\n'
             f'r² = {r2_lin:.3f} → R² = {r2_nonlin:.3f} ± {r2_cv.std():.3f} (RF 5-fold CV)',
             fontsize=12)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / '05_r2_linear_vs_nonlinear.png', bbox_inches='tight', dpi=130)
plt.close()
print('saved: 05_r2_linear_vs_nonlinear.png')

print()
print(f'완료. 차트 5장 → {OUT}')
