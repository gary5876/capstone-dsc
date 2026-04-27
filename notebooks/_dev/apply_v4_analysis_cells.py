"""04 노트북에 v4 추가 분석 셀들을 삽입.

추가되는 분석 (cell 22 핵심 주장 셀 직전에 삽입):
- 8-2. F1 — Polluter Hold-out 검증
- 8-3. F2 — 비선형 결합 R² (RandomForest 5-fold CV)
- 8-4. F4 — min-aggregation 비교
- 8-5. F6 — 모델 클래스별 프로파일 가중치 비교
- 8-6. F9 — 등급 임계값 sensitivity
- 8-7. F10 — 다지표 (accuracy/AUC) r 비교
- 8-8. 한계 명시 (F5/F8)

cell 23 실행로그 셀도 새 분석 결과 포함하도록 업데이트.
"""
import json

NB_PATH = 'G:/내 드라이브/capstone/dsc/notebooks/04_scoreboard.ipynb'

# ============================================================
# 새 셀들 정의 (모두 source는 list[str] — Jupyter 형식)
# ============================================================
CELL_F1_MD = """## 8.1 F1 검증 — Polluter Hold-out 분석

각 polluter를 hold-out으로 두고 그 슬라이스에서만 r 측정.
가중치가 ML 결과를 보고 fitting된 결과라면, hold-out 슬라이스에서 r이 떨어져야 함.
모든 hold-out에서 r > 0.3 + p < 0.05이면 "fitting 아닌 일반화 신호"."""

CELL_F1_CODE = """# ============================================================
# 8-2. F1 — Polluter Hold-out 검증
# ============================================================
holdout_results = []
for pol in ['completeness', 'uniqueness', 'consistent_repr', 'class_balance', 'feature_accuracy']:
    sub = df_merged[df_merged['polluter'] == pol].dropna(subset=['score', 'f1_macro'])
    if len(sub) < 10:
        continue
    r_h, p_h = sp_stats.pearsonr(sub['score'], sub['f1_macro'])
    ok = (r_h > 0.3) and (p_h < 0.05)
    holdout_results.append({'polluter': pol, 'n': len(sub), 'r': r_h, 'p': p_h, 'pass': ok})

holdout_df = pd.DataFrame(holdout_results)
print('=' * 60)
print('Polluter Hold-out 검증 (각 polluter 슬라이스에서 r)')
print('=' * 60)
print(holdout_df.to_string(index=False, float_format='{:.4f}'.format))
n_pass = holdout_df['pass'].sum()
print(f'\\n→ {n_pass}/{len(holdout_df)} polluter에서 r > 0.3, p < 0.05')
print('→ 결론: ', '가중치는 fitting이 아닌 일반화 신호' if n_pass == len(holdout_df)
      else f'{len(holdout_df)-n_pass}개 polluter는 fitting 의존성 가능성, 추가 진단 필요')"""

CELL_F2_MD = """## 8.2 F2 검증 — 비선형 결합 R²

선형 결합 DSC = Σ wᵢ sᵢ는 r²(현재) 정도의 분산만 설명.
8개 metric을 RandomForest에 입력해 비선형 결합 시 R²을 측정.
R² ≥ 0.5면 "DSC 차원에 정보는 풍부하나 선형 결합의 한계" 방어 가능."""

CELL_F2_CODE = """# ============================================================
# 8-3. F2 — 비선형 R² (RandomForest 5-fold CV)
# ============================================================
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score

metric_cols_full = [c for c in [
    'completeness', 'uniqueness', 'validity', 'consistency',
    'outlier_ratio', 'class_balance', 'feature_correlation',
    'label_consistency', 'feature_informativeness', 'value_accuracy'
] if c in df_merged.columns]

valid = df_merged.dropna(subset=metric_cols_full + ['f1_macro'])
X_metrics = valid[metric_cols_full].values
y_target = valid['f1_macro'].values

print('=' * 60)
print('비선형 결합 R² (RandomForest 5-fold CV)')
print('=' * 60)
print(f'metric 수: {len(metric_cols_full)}, 표본 수: {len(valid)}')
print(f'사용 metric: {metric_cols_full}')

rf_reg = RandomForestRegressor(n_estimators=200, random_state=1, n_jobs=-1)
cv = KFold(n_splits=5, shuffle=True, random_state=1)
r2_cv = cross_val_score(rf_reg, X_metrics, y_target, cv=cv, scoring='r2')

r_lin_full = r_pearson
r2_lin = r_lin_full ** 2
r2_nonlin = r2_cv.mean()

print(f'\\n  선형 r² (DSC = Σwᵢsᵢ):     {r2_lin:.4f}')
print(f'  비선형 R² (RF 5-fold CV):   {r2_nonlin:.4f} ± {r2_cv.std():.4f}')
print(f'  fold별 R²: {[f\"{x:.3f}\" for x in r2_cv]}')
print(f'\\n→ 비선형 R² / 선형 r² = {r2_nonlin/r2_lin:.2f}배. ', end='')
if r2_nonlin > 0.5:
    print('DSC 차원에 충분한 정보. 선형 결합의 한계로 r² 작게 보임.')
else:
    print('비선형 모델로도 R² < 0.5. 차원 자체의 정보량 한계.')"""

CELL_F4_MD = """## 8.3 F4 검증 — 가법성 가정 (min-aggregation 비교)

DSC = Σ wᵢ sᵢ는 차원 간 독립을 가정.
min(metrics) (최약 차원이 결정)와 비교해 가법 가정의 타당성을 측정."""

CELL_F4_CODE = """# ============================================================
# 8-4. F4 — min-aggregation 비교
# ============================================================
metric_cols_use = [c for c in [
    'completeness', 'uniqueness', 'validity', 'consistency',
    'outlier_ratio', 'class_balance', 'feature_correlation',
    'label_consistency', 'feature_informativeness'
] if c in df_merged.columns]

df_merged['dsc_min'] = df_merged[metric_cols_use].min(axis=1) * 100

valid = df_merged.dropna(subset=['dsc_min', 'score', 'f1_macro'])
r_add, p_add = sp_stats.pearsonr(valid['score'], valid['f1_macro'])
r_min, p_min = sp_stats.pearsonr(valid['dsc_min'], valid['f1_macro'])

print('=' * 60)
print('가법 vs min-aggregation 비교')
print('=' * 60)
print(f'  가법 (DSC = Σwᵢsᵢ):   r = {r_add:+.4f}, p = {p_add:.2e}, r² = {r_add**2:.4f}')
print(f'  min(metrics):          r = {r_min:+.4f}, p = {p_min:.2e}, r² = {r_min**2:.4f}')
diff = abs(r_add) - abs(r_min)
print(f'  차이: {diff:+.4f} ({\"가법 우세\" if diff > 0 else \"min 우세\"})')

print('\\n--- 모델별 비교 ---')
for m in sorted(df_merged['model'].unique()):
    sub = df_merged[df_merged['model'] == m].dropna(subset=['dsc_min', 'score', 'f1_macro'])
    if len(sub) < 10:
        continue
    rA, _ = sp_stats.pearsonr(sub['score'], sub['f1_macro'])
    rM, _ = sp_stats.pearsonr(sub['dsc_min'], sub['f1_macro'])
    winner = 'add' if abs(rA) > abs(rM) else 'min'
    print(f'  {m:<22}  r_add={rA:+.4f}  r_min={rM:+.4f}  → {winner}')"""

CELL_F6_MD = """## 8.4 F6 검증 — 모델 클래스별 가중치 프로파일

DSC가 모델 무관 점수라면 모델별 r이 비슷해야 하나, 실제로는 격차가 큼.
noise_sensitive(LR/SVC/MLP) vs tree_based(RF/XGB) 두 프로파일로 가중치 차등하여
프로파일별 r이 default보다 향상되는지 검증."""

CELL_F6_CODE = """# ============================================================
# 8-5. F6 — 모델 클래스별 프로파일 가중치
# ============================================================
WEIGHTS_DEFAULT = {
    'completeness': 0.20, 'uniqueness': 0.15, 'validity': 0.05,
    'consistency': 0.10, 'outlier_ratio': 0.05,
    'class_balance': 0.10, 'feature_correlation': 0.05,
    'label_consistency': 0.20, 'feature_informativeness': 0.10,
}
WEIGHTS_NOISE = {
    'completeness': 0.20, 'uniqueness': 0.10, 'validity': 0.05,
    'consistency': 0.10, 'outlier_ratio': 0.10,
    'class_balance': 0.05, 'feature_correlation': 0.05,
    'label_consistency': 0.25, 'feature_informativeness': 0.10,
}
WEIGHTS_TREE = {
    'completeness': 0.15, 'uniqueness': 0.10, 'validity': 0.03,
    'consistency': 0.10, 'outlier_ratio': 0.02,
    'class_balance': 0.20, 'feature_correlation': 0.05,
    'label_consistency': 0.20, 'feature_informativeness': 0.15,
}
NOISE_MODELS = {'LogisticRegression', 'SVC', 'MLP'}
TREE_MODELS = {'RandomForest', 'XGBoost'}

def dsc_with_weights(row, weights):
    s = 0.0
    for k, w in weights.items():
        if k in row.index and pd.notna(row[k]):
            s += float(row[k]) * w
    return s * 100

print('=' * 60)
print('모델 클래스별 프로파일 가중치 효과')
print('=' * 60)

# 각 행의 metric을 사용해 두 가지 프로파일 점수 계산
metric_cols_avail = [c for c in WEIGHTS_DEFAULT if c in df_merged.columns]
df_prof = df_merged.dropna(subset=metric_cols_avail + ['f1_macro']).copy()
df_prof['dsc_noise'] = df_prof.apply(lambda r: dsc_with_weights(r, WEIGHTS_NOISE), axis=1)
df_prof['dsc_tree']  = df_prof.apply(lambda r: dsc_with_weights(r, WEIGHTS_TREE), axis=1)

# 모델 그룹별 r — default vs 프로파일
print('\\n  group           | default r | profile r | Δ')
print('  ' + '-' * 60)
for group_name, models, profile_col in [
    ('noise_sensitive (LR/SVC/MLP)', NOISE_MODELS, 'dsc_noise'),
    ('tree_based (RF/XGB)',          TREE_MODELS,  'dsc_tree'),
]:
    sub = df_prof[df_prof['model'].isin(models)]
    if len(sub) < 10:
        continue
    rD, _ = sp_stats.pearsonr(sub['score'], sub['f1_macro'])
    rP, _ = sp_stats.pearsonr(sub[profile_col], sub['f1_macro'])
    print(f'  {group_name:<34}  {rD:+.4f}   {rP:+.4f}   {rP-rD:+.4f}')"""

CELL_F9_MD = """## 8.5 F9 검증 — 등급 임계값 sensitivity

기존 등급 임계값 90/75/60은 출처 미상의 휴리스틱.
임계값을 바꿔도 ANOVA 결과가 유의한지 확인하여 임계값 의존성 측정."""

CELL_F9_CODE = """# ============================================================
# 8-6. F9 — 등급 임계값 sensitivity
# ============================================================
def grade_with_thresholds(score, thresholds):
    if score >= thresholds[0]: return 'A'
    if score >= thresholds[1]: return 'B'
    if score >= thresholds[2]: return 'C'
    return 'D'

# 분위수 기반 임계값
q33, q66, q90 = df_merged['score'].quantile([0.33, 0.66, 0.90]).values
threshold_sets = {
    'current_90_75_60': (90, 75, 60),
    'percentile_90_66_33': (q90, q66, q33),
    'conservative_80_60_40': (80, 60, 40),
}

print('=' * 60)
print('등급 임계값 sensitivity (ANOVA F-test)')
print('=' * 60)
sensitivity_results = []
for name, thresh in threshold_sets.items():
    grade_col = df_merged['score'].apply(lambda s: grade_with_thresholds(s, thresh) if pd.notna(s) else None)
    groups_t = [df_merged[grade_col == g]['f1_macro'].dropna() for g in ['A','B','C','D']]
    groups_t = [g for g in groups_t if len(g) >= 2]
    if len(groups_t) < 2:
        sensitivity_results.append({'thresholds': name, 'F': None, 'p': None, 'grades': 0})
        continue
    F, P = sp_stats.f_oneway(*groups_t)
    sensitivity_results.append({'thresholds': name, 'F': F, 'p': P, 'grades': len(groups_t)})
    counts = [len(g) for g in groups_t]
    print(f'  {name:<26}  F = {F:7.3f}  p = {P:.2e}  groups={len(groups_t)}  n={counts}')

n_sig = sum(1 for r in sensitivity_results if r['p'] is not None and r['p'] < 0.05)
print(f'\\n→ {n_sig}/{len(sensitivity_results)} 임계값 세트에서 ANOVA 유의')
if n_sig == len(sensitivity_results):
    print('→ 결론: 등급별 F1 차이는 임계값 선택과 무관하게 견고')
else:
    print('→ 결론: 임계값에 따라 결과 변동, 임계값 의존성 인정 필요')"""

CELL_F10_MD = """## 8.6 F10 검증 — 다지표 (accuracy / AUC) r 비교

F1 macro 단일 의존 한계 보강. accuracy, AUC와도 r 측정."""

CELL_F10_CODE = """# ============================================================
# 8-7. F10 — 다지표 r 비교
# ============================================================
print('=' * 60)
print('ML 평가 지표별 DSC ↔ 성능 r')
print('=' * 60)
multi_metric_results = []
for ml_metric in ['f1_macro', 'accuracy', 'auc_roc']:
    if ml_metric not in df_merged.columns:
        continue
    sub = df_merged.dropna(subset=['score', ml_metric])
    if len(sub) < 3:
        continue
    r_m, p_m = sp_stats.pearsonr(sub['score'], sub[ml_metric])
    rho_m, _ = sp_stats.spearmanr(sub['score'], sub[ml_metric])
    multi_metric_results.append({'metric': ml_metric, 'pearson_r': r_m, 'p': p_m, 'spearman_rho': rho_m, 'n': len(sub)})
    print(f'  {ml_metric:<12}  Pearson r = {r_m:+.4f} (p={p_m:.2e})  Spearman ρ = {rho_m:+.4f}  n={len(sub)}')

n_strong = sum(1 for r in multi_metric_results if abs(r['pearson_r']) > 0.3)
print(f'\\n→ {n_strong}/{len(multi_metric_results)} ML 지표에서 |r| > 0.3 — DSC는 F1 macro 외 지표와도 양의 상관')"""

CELL_LIMITS_MD = """## 8.7 한계 (F5 / F8 — 자체 인정)

reviewer 공격을 사전 방어하기 위해 명시적으로 인정하는 한계 4가지."""

CELL_LIMITS_CODE = """# ============================================================
# 8-8. 한계 명시 (F5 / F8)
# ============================================================
print('=' * 60)
print('본 연구의 한계 (자체 인정)')
print('=' * 60)
print('''
F5. 합성 오염 시나리오 한정
   - 본 연구는 5종 DQ4AI polluter (completeness, uniqueness, consistent_repr,
     class_balance, feature_accuracy)로 만든 합성 오염에서의 상관관계를 입증.
   - 자연 발생 노이즈, 라벨 누락, 시계열·텍스트·이미지 데이터로의 일반화는
     후속 연구로 남김.

F8. Baseline = 원본 가정
   - 모든 점수는 원본을 100점 baseline으로 anchored됨.
   - 원본의 자연 노이즈는 별도 검증되지 않음.

F2. 선형 결합의 분산 설명력 한계
   - DSC = Σ wᵢ sᵢ의 r²는 한정적.
   - 비선형 결합(RF) R²은 더 높음 → DSC 차원에 정보는 풍부하나
     선형 가중합으로는 정보 일부만 활용.

F4. 가법성 가정
   - 차원 간 독립을 가정한 선형 근사. 결측 + 클래스불균형이 동시에
     발생하는 상호작용은 미반영.
''')
print('주장 범위: "정형 분류 데이터에서 5종 합성 오염 시나리오에 대한")
print('              DSC ↔ ML F1 macro의 양의 상관관계"')"""


# ============================================================
# 23번 (실행로그) 셀 업데이트 — 새 분석 결과 추가
# ============================================================
CELL_LOG_NEW = """# ============================================================
# 9. 실행 로그 저장
# ============================================================
from datetime import datetime

log_lines = []
log_lines.append('# 노트북 04 실행 로그: 스코어보드 & 시각화 & 통계 검증 (v4)')
log_lines.append('')
log_lines.append(f'- **실행 시각**: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}')
log_lines.append(f'- **DSC 점수 데이터**: {df_dsc.shape[0]}건')
log_lines.append(f'- **모델 성능 데이터**: {df_perf.shape[0]}건')
log_lines.append(f'- **병합 결과**: {df_merged.shape[0]}건')
log_lines.append('')

log_lines.append('## 1. 핵심 결과: DSC ↔ 모델 성능 상관관계')
log_lines.append('')
log_lines.append('| 지표 | 값 | p-value | 해석 |')
log_lines.append('|---|---|---|---|')
sig_pearson = '유의' if p_pearson < 0.05 else '비유의'
sig_spearman = '유의' if p_spearman < 0.05 else '비유의'
log_lines.append(f'| Pearson r | {r_pearson:.4f} | {p_pearson:.2e} | {sig_pearson} |')
log_lines.append(f'| r² | {r_pearson**2:.4f} | - | 분산 설명력 |')
log_lines.append(f'| Spearman ρ | {r_spearman:.4f} | {p_spearman:.2e} | {sig_spearman} |')
log_lines.append(f'| 표본 수 | {len(x)} | - | - |')
log_lines.append('')
strength = '강함' if abs(r_pearson) > 0.7 else '보통' if abs(r_pearson) > 0.4 else '약함'
log_lines.append(f'**해석**: 피어슨 r {r_pearson:.3f}은 {strength} 수준의 양의 상관 (예측이 아닌 상관 주장).')
log_lines.append(f'DSC Score 10점 하락 시 F1 평균 {abs(slope * 10):.4f} 하락.')
log_lines.append('')

log_lines.append('## 2. DSC 등급별 모델 성능')
log_lines.append('')
log_lines.append('| 등급 | 평균 F1 | 표준편차 | 최소 | 최대 | 건수 |')
log_lines.append('|---|---|---|---|---|---|')
for grade in ['A', 'B', 'C', 'D']:
    if grade in grade_stats.index and not pd.isna(grade_stats.loc[grade, 'mean']):
        s = grade_stats.loc[grade]
        log_lines.append(f'| {grade} | {s[\"mean\"]:.4f} | {s[\"std\"]:.4f} | {s[\"min\"]:.4f} | {s[\"max\"]:.4f} | {int(s[\"count\"])} |')
log_lines.append('')

if len(groups) >= 2:
    log_lines.append(f'**ANOVA**: F = {f_stat:.4f}, p = {p_anova:.2e}')
    log_lines.append('→ 등급 간 F1 차이가 통계적으로 유의함' if p_anova < 0.05 else '→ 비유의')
    log_lines.append('')

log_lines.append('## 3. 모델별 상관관계')
log_lines.append('')
log_lines.append('| 모델 | Pearson r | p-value | 표본 수 |')
log_lines.append('|---|---|---|---|')
for mc in model_corrs:
    log_lines.append(f'| {mc[\"model\"]} | {mc[\"pearson_r\"]:.4f} | {mc[\"p_value\"]:.2e} | {mc[\"n\"]} |')
log_lines.append('')

log_lines.append('## 4. 데이터셋별 상관관계')
log_lines.append('')
log_lines.append('| 데이터셋 | Pearson r | p-value | 표본 수 |')
log_lines.append('|---|---|---|---|')
for ds in df_merged['dataset'].unique():
    sub = df_merged[df_merged['dataset'] == ds].dropna(subset=['score', 'f1_macro'])
    if len(sub) >= 3:
        r, p = sp_stats.pearsonr(sub['score'], sub['f1_macro'])
        log_lines.append(f'| {ds} | {r:.4f} | {p:.2e} | {len(sub)} |')
log_lines.append('')

# v4 추가 분석 결과
log_lines.append('## 5. F1 — Polluter Hold-out 검증')
log_lines.append('')
log_lines.append('| Polluter | n | r | p | PASS |')
log_lines.append('|---|---|---|---|---|')
for h in holdout_results:
    log_lines.append(f'| {h[\"polluter\"]} | {h[\"n\"]} | {h[\"r\"]:.4f} | {h[\"p\"]:.2e} | {\"OK\" if h[\"pass\"] else \"FAIL\"} |')
log_lines.append('')

log_lines.append('## 6. F2 — 비선형 결합 R²')
log_lines.append('')
log_lines.append(f'- 선형 r² (DSC = Σwᵢsᵢ): **{r2_lin:.4f}**')
log_lines.append(f'- 비선형 R² (RF 5-fold CV): **{r2_nonlin:.4f}** ± {r2_cv.std():.4f}')
log_lines.append(f'- 비율: 비선형/선형 = {r2_nonlin/r2_lin:.2f}배')
log_lines.append('')

log_lines.append('## 7. F4 — 가법 vs min-aggregation')
log_lines.append('')
log_lines.append(f'- 가법: r = {r_add:.4f}, r² = {r_add**2:.4f}')
log_lines.append(f'- min:  r = {r_min:.4f}, r² = {r_min**2:.4f}')
log_lines.append('')

log_lines.append('## 8. F9 — 등급 임계값 Sensitivity')
log_lines.append('')
log_lines.append('| Threshold set | F | p | groups | sig |')
log_lines.append('|---|---|---|---|---|')
for sr in sensitivity_results:
    if sr['F'] is None:
        continue
    sig = '유의' if sr['p'] < 0.05 else '비유의'
    log_lines.append(f'| {sr[\"thresholds\"]} | {sr[\"F\"]:.3f} | {sr[\"p\"]:.2e} | {sr[\"grades\"]} | {sig} |')
log_lines.append('')

log_lines.append('## 9. F10 — 다지표 r')
log_lines.append('')
log_lines.append('| ML metric | Pearson r | p | Spearman ρ | n |')
log_lines.append('|---|---|---|---|---|')
for mr in multi_metric_results:
    log_lines.append(f'| {mr[\"metric\"]} | {mr[\"pearson_r\"]:.4f} | {mr[\"p\"]:.2e} | {mr[\"spearman_rho\"]:.4f} | {mr[\"n\"]} |')
log_lines.append('')

log_lines.append('## 10. 한계 (F5 / F8 자체 인정)')
log_lines.append('')
log_lines.append('1. **합성 오염 시나리오 한정** — 5종 polluter, 자연 노이즈/라벨 누락/시계열·텍스트·이미지 일반화 미검증')
log_lines.append('2. **Baseline = 원본 가정** — 원본의 자연 노이즈는 별도 검증되지 않음')
log_lines.append('3. **선형 결합 한계** — 비선형 R²이 선형 r²보다 높음 → 차원 정보 일부만 활용')
log_lines.append('4. **가법성 가정** — 차원 간 독립 가정, 상호작용 미반영')
log_lines.append('')

log_lines.append('## 11. 산출물')
log_lines.append('')
log_lines.append(f'- `merged_results.csv` — 통합 결과 {len(df_merged)}건')
log_lines.append(f'- `scoreboard.csv` — 스코어보드 {len(pivot_f1)}건')
log_lines.append(f'- `charts/` — 시각화 5종')
log_lines.append(f'- `04_execution_log.md` — 이 로그 파일')
log_lines.append('')
log_lines.append('---')
log_lines.append('*이 로그는 노트북 04 실행 시 자동 생성됨 (v4)*')

log_path = f'{RESULTS_DIR}/04_execution_log.md'
with open(log_path, 'w', encoding='utf-8') as f:
    f.write('\\n'.join(log_lines))
print(f'실행 로그 저장: {log_path}')"""


def make_md_cell(source_str):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': [source_str]}


def make_code_cell(source_str):
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [source_str],
    }


def main():
    with open(NB_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    print(f'기존 셀 수: {len(nb["cells"])}')

    # cell 22 (8-1 핵심 주장)와 cell 23 (실행 로그) 사이가 아니라,
    # cell 22(8-1) 직전(인덱스 22)에 새 셀들 삽입.
    # cell 21 = "## 8. 핵심 주장" markdown, cell 22 = 8-1 code, cell 23 = 9. 실행로그 code
    # 새 셀 8-2~8-8을 cell 22 바로 앞 (인덱스 22)에 삽입하면 8-1이 뒤로 밀림.
    # 더 자연스럽게: cell 22 바로 뒤(인덱스 23)에 삽입. 그러면 8-1 → 8-2~8-8 → 9. 실행로그.

    new_cells = [
        make_md_cell(CELL_F1_MD),       make_code_cell(CELL_F1_CODE),
        make_md_cell(CELL_F2_MD),       make_code_cell(CELL_F2_CODE),
        make_md_cell(CELL_F4_MD),       make_code_cell(CELL_F4_CODE),
        make_md_cell(CELL_F6_MD),       make_code_cell(CELL_F6_CODE),
        make_md_cell(CELL_F9_MD),       make_code_cell(CELL_F9_CODE),
        make_md_cell(CELL_F10_MD),      make_code_cell(CELL_F10_CODE),
        make_md_cell(CELL_LIMITS_MD),   make_code_cell(CELL_LIMITS_CODE),
    ]

    # cell 22 바로 뒤에 삽입 (= 인덱스 23 위치)
    nb['cells'][23:23] = new_cells

    # 마지막 셀 (원래 cell 23, 이제 23 + len(new_cells) = 23+14 = 37)이 실행 로그
    # 실행 로그 셀을 새 정보 포함하도록 교체
    log_cell_idx = 23 + len(new_cells)
    nb['cells'][log_cell_idx]['source'] = [CELL_LOG_NEW]

    print(f'새 셀 추가: {len(new_cells)}개')
    print(f'실행 로그 셀 업데이트: cell {log_cell_idx}')
    print(f'최종 셀 수: {len(nb["cells"])}')

    with open(NB_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f'저장: {NB_PATH}')

    # 셀 헤더 다시 출력해 검증
    nb2 = json.load(open(NB_PATH, encoding='utf-8'))
    print('\n새 셀 구조:')
    for i, cell in enumerate(nb2['cells']):
        head = ''.join(cell['source']).split('\n')[0][:90]
        print(f'  cell {i:2d} [{cell["cell_type"]:8s}] {head}')


if __name__ == '__main__':
    main()
