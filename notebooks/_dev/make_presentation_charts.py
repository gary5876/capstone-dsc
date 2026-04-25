"""발표용 핵심 차트 4장 생성 (정식 결과 — 노트북 03 재실행 후 데이터).
입력: dsc_scores.csv (v3.2 정식, 87건) + model_performance.csv (정식 학습, 405건)
출력: documents/reports/charts/20260425/*.png
참고: letter × uniqueness 6단계는 학습 누락(데이터 확장으로 timeout 추정).
v2 비교 수치는 진단 보고서에서 인용 (옛 merged_results.csv 기준)."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats

rcParams['font.family'] = 'Malgun Gothic'
rcParams['axes.unicode_minus'] = False
rcParams['figure.dpi'] = 110

OUT = 'G:/내 드라이브/capstone/dsc/documents/reports/charts/20260425'

# 정식 데이터 로드
dsc = pd.read_csv('G:/내 드라이브/capstone/dsc/results/dsc_scores.csv')
perf = pd.read_csv('G:/내 드라이브/capstone/dsc/results/model_performance.csv')
m = perf.merge(dsc, on=['dataset','polluter','level'])

# v2 비교용: 옛 merged_results
m_v2 = pd.read_csv('G:/내 드라이브/capstone/dsc/results/merged_results.csv')

print(f'정식 m shape: {m.shape}, 옛 m_v2 shape: {m_v2.shape}')

# ============================================================
# 차트 1: DSC vs F1 산점도 (v2 vs v3.2)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
ds_colors = {'TelcoCustomerChurn':'#1f77b4', 'SouthGermanCredit':'#ff7f0e', 'letter':'#2ca02c'}
for ax, data, title in [(axes[0], m_v2, 'v2 (개선 전)'), (axes[1], m, 'v3.2 (개선 후 — 정식)')]:
    for ds, c in ds_colors.items():
        sub = data[data['dataset']==ds]
        ax.scatter(sub['score'], sub['f1_macro'], c=c, alpha=0.55, s=24, label=ds, edgecolors='none')
    if len(data) >= 3:
        slope, intercept, r, p, _ = stats.linregress(data['score'], data['f1_macro'])
        x = np.linspace(data['score'].min(), data['score'].max(), 50)
        ax.plot(x, intercept + slope*x, 'k--', linewidth=1.2, alpha=0.6)
        ax.set_title(f'{title}  (n={len(data)})\nPearson r = {r:.3f}, p = {p:.2e}', fontsize=12)
    ax.set_xlabel('DSC Score', fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc='lower right')
axes[0].set_ylabel('F1 (macro)', fontsize=11)
fig.suptitle('DSC Score ↔ ML 성능 상관관계 — v2 vs v3.2', fontsize=14, y=1.00)
plt.tight_layout()
plt.savefig(f'{OUT}/01_scatter_v2_vs_v32.png', bbox_inches='tight', dpi=130)
plt.close()
print('saved: 01_scatter_v2_vs_v32.png')

# ============================================================
# 차트 2: 등급별 F1 박스플롯 (v3.2 정식)
# ============================================================
fig, ax = plt.subplots(figsize=(8.5, 5))
order = ['A','B','C','D']
data = [m[m['grade']==g]['f1_macro'].values for g in order]
counts = [len(d) for d in data]
bp = ax.boxplot(data, tick_labels=[f'{g}\n(n={c})' for g, c in zip(order, counts)],
                patch_artist=True, widths=0.55, showmeans=True,
                meanprops=dict(marker='D', markerfacecolor='red', markersize=6))
colors = ['#2ecc71','#3498db','#f39c12','#e74c3c']
for patch, c in zip(bp['boxes'], colors):
    patch.set_facecolor(c); patch.set_alpha(0.6)
ax.set_xlabel('DSC 등급', fontsize=11)
ax.set_ylabel('F1 (macro)', fontsize=11)
valid = [d for d in data if len(d) > 0]
f, p = stats.f_oneway(*valid)
means = [d.mean() for d in valid]
ax.set_title(f'DSC 등급별 ML 성능 분포 (v3.2 정식, n={len(m)})\n'
             f'평균 F1: A={means[0]:.3f}, B={means[1]:.3f}, C={means[2]:.3f}, D={means[3]:.3f}  |  '
             f'ANOVA F={f:.2f}, p={p:.2e}', fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/02_boxplot_grade_v32.png', bbox_inches='tight', dpi=130)
plt.close()
print('saved: 02_boxplot_grade_v32.png')

# ============================================================
# 차트 3: Polluter × Dataset 별 r 비교 (v2 vs v3.2 정식)
# ============================================================
fig, ax = plt.subplots(figsize=(11, 6.5))
rows = []
for ds in m['dataset'].unique():
    for pol in m['polluter'].unique():
        s2 = m_v2[(m_v2['dataset']==ds)&(m_v2['polluter']==pol)]
        s3 = m[(m['dataset']==ds)&(m['polluter']==pol)]
        if len(s3) < 5 or s3['score'].std()==0 or pol=='none':
            continue
        if len(s2) < 5 or s2['score'].std()==0:
            r2 = 0
        else:
            r2, _ = stats.pearsonr(s2['score'], s2['f1_macro'])
        r3, _ = stats.pearsonr(s3['score'], s3['f1_macro'])
        rows.append({'label': f'{pol[:14]} × {ds[:12]}', 'v2': r2, 'v3.2': r3, 'pol': pol, 'ds': ds})
rdf = pd.DataFrame(rows).sort_values('v3.2', ascending=True).reset_index(drop=True)
y = np.arange(len(rdf))
w = 0.4
bars2 = ax.barh(y-w/2, rdf['v2'], height=w, color='#bbbbbb', label='v2 (개선 전)')
bars3 = ax.barh(y+w/2, rdf['v3.2'], height=w, color='#2980b9', label='v3.2 (개선 후 정식)')
ax.set_yticks(y)
ax.set_yticklabels(rdf['label'], fontsize=9)
ax.axvline(0, color='k', linewidth=0.8)
ax.set_xlabel('Pearson r (DSC vs F1)', fontsize=11)
ax.set_xlim(-1.0, 1.05)
ax.set_title('Polluter × Dataset 별 상관계수 — 모든 슬라이스가 음수→양수로 회복',
             fontsize=12)
ax.legend(loc='lower right', fontsize=10)
ax.grid(axis='x', alpha=0.3)
# 부호 반전 강조
for i, row in rdf.iterrows():
    if row['v2'] < -0.3 and row['v3.2'] > 0.3:
        bars2[i].set_color('#e74c3c')
        bars3[i].set_color('#27ae60')
        ax.text(row['v3.2']+0.02, i+w/2, '부호 회복', va='center', fontsize=8.5, color='#27ae60', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/03_polluter_r_comparison.png', bbox_inches='tight', dpi=130)
plt.close()
print('saved: 03_polluter_r_comparison.png')

# ============================================================
# 차트 4: P1 사례 — outlier_ratio가 더 이상 역반응 안 함
# ============================================================
fig, ax = plt.subplots(figsize=(8.5, 5))
levels = [0.0, 0.1, 0.25, 0.5, 0.75]
v2_outlier = {
    'letter':              [0.9671, 0.9772, 0.9867, 0.9906, 0.9918],  # 상승 ↑
    'SouthGermanCredit':  [0.9450, 0.9540, 0.9670, 0.9740, 0.9873],  # 상승 ↑
}
v32_outlier_letter = dsc[(dsc['dataset']=='letter')&(dsc['polluter'].isin(['none','feature_accuracy']))&(dsc['level']<=0.75)].sort_values('level')['outlier_ratio'].tolist()
v32_outlier_sg = dsc[(dsc['dataset']=='SouthGermanCredit')&(dsc['polluter'].isin(['none','feature_accuracy']))&(dsc['level']<=0.75)].sort_values('level')['outlier_ratio'].tolist()

ax.plot(levels, v2_outlier['letter'], 'o-', color='#e74c3c', linewidth=2.2, label='v2 letter (역방향 ↑)')
ax.plot(levels, v2_outlier['SouthGermanCredit'], 's-', color='#c0392b', linewidth=2.2, label='v2 SouthGerman (역방향 ↑)')
ax.plot(levels, v32_outlier_letter, 'o--', color='#27ae60', linewidth=2.2, label='v3.2 letter (정방향 ↓)')
ax.plot(levels, v32_outlier_sg, 's--', color='#16a085', linewidth=2.2, label='v3.2 SouthGerman (정방향 ↓)')
ax.set_xlabel('feature_accuracy 오염 강도', fontsize=11)
ax.set_ylabel('outlier_ratio (1.0 = outlier 없음)', fontsize=11)
ax.set_title('P1 해결 — outlier_ratio가 더 이상 노이즈에 역반응 안 함\n'
             '(reference 분포 IQR 고정으로 자기참조 함정 회피)', fontsize=11)
ax.grid(alpha=0.3)
ax.legend(loc='lower left', fontsize=9.5)
plt.tight_layout()
plt.savefig(f'{OUT}/04_outlier_p1_fix.png', bbox_inches='tight', dpi=130)
plt.close()
print('saved: 04_outlier_p1_fix.png')

print()
print(f'4개 차트 → {OUT}/')
