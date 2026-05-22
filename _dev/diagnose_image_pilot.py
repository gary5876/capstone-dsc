"""이미지 cell pilot 진단 — 차원별 r 기여도 + dead 차원 + 가중치 비교."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

scores = pd.read_csv(RESULTS / "dsc_scores_image.csv")
perf = pd.read_csv(RESULTS / "model_performance_image.csv")
pilot = json.loads((RESULTS / "tuned_weights_image_pilot.json").read_text())

METRIC_COLS = [
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
TUNED_W = pilot["weights"]

# (dataset, polluter, level) 단위로 score+metric, model accuracy 병합
merged = perf.merge(scores, on=["dataset", "polluter", "level"], how="inner")

# CIFAR10에서 학습한 모델 = Flowers102 row 제외, ResNet18+CNNSimple 둘다 사용
merged = merged[merged["model"].isin(["ResNet18", "CNNSimple"])].copy()
merged["set_label"] = merged["dataset"]  # tune=CIFAR10, heldout=FashionMNIST

print("=" * 78)
print("[STEP 0] 데이터 요약")
print("=" * 78)
print(f"merged rows: {len(merged)}")
print(merged.groupby(["dataset", "model"]).size().to_string())
print()

# =================================================================
# STEP 1. 차원별 표준편차 — dead 차원 식별 (set별 + 전체)
# =================================================================
print("=" * 78)
print("[STEP 1] 차원별 표준편차 (dataset별 + 전체)")
print("=" * 78)
std_table = pd.DataFrame({
    "all": scores[METRIC_COLS].std(),
    "CIFAR10": scores[scores.dataset == "CIFAR10"][METRIC_COLS].std(),
    "FashionMNIST": scores[scores.dataset == "FashionMNIST"][METRIC_COLS].std(),
})
print(std_table.round(4).to_string())
print()
dead = std_table[(std_table["all"] < 0.01)].index.tolist()
print(f"dead 차원 (std<0.01, 전체): {dead}")
print()

# =================================================================
# STEP 2. 차원별 Pearson r — vs accuracy
# =================================================================
def per_dim_r(df, label):
    rows = []
    for col in METRIC_COLS:
        x = df[col].to_numpy()
        y = df["accuracy"].to_numpy()
        if x.std() == 0:
            rows.append({"metric": col, "r": np.nan, "p": np.nan, "std": x.std()})
            continue
        r, p = stats.pearsonr(x, y)
        rows.append({"metric": col, "r": r, "p": p, "std": float(x.std())})
    return pd.DataFrame(rows).assign(set=label)

print("=" * 78)
print("[STEP 2] 차원별 Pearson r (개별 차원 vs accuracy)")
print("=" * 78)
for label, sub in [
    ("CIFAR10 (튜닝)", merged[merged.dataset == "CIFAR10"]),
    ("FashionMNIST (held-out)", merged[merged.dataset == "FashionMNIST"]),
    ("전체 pooled", merged),
]:
    print(f"\n--- {label} (n={len(sub)}) ---")
    r_df = per_dim_r(sub, label).sort_values("r", ascending=False)
    print(r_df.round(4).to_string(index=False))
print()

# =================================================================
# STEP 3. default vs tuned 가중치로 score 재계산 후 r
# =================================================================
def score_from_weights(row, w):
    return sum(row[k] * w[k] for k in w) * 100

def eval_weights(w, label):
    df = merged.copy()
    df["score_w"] = df.apply(lambda r: score_from_weights(r, w), axis=1)
    out = []
    for ds, sub in df.groupby("dataset"):
        r, p = stats.pearsonr(sub["score_w"], sub["accuracy"])
        rs, ps = stats.spearmanr(sub["score_w"], sub["accuracy"])
        out.append({"dataset": ds, "n": len(sub), "pearson_r": r, "p": p,
                    "spearman_rho": rs, "spearman_p": ps})
    # pooled
    r, p = stats.pearsonr(df["score_w"], df["accuracy"])
    rs, ps = stats.spearmanr(df["score_w"], df["accuracy"])
    out.append({"dataset": "POOLED", "n": len(df), "pearson_r": r, "p": p,
                "spearman_rho": rs, "spearman_p": ps})
    return pd.DataFrame(out).assign(weights=label)

print("=" * 78)
print("[STEP 3] default vs tuned 가중치로 r 재현")
print("=" * 78)
res_default = eval_weights(DEFAULT_W, "default")
res_tuned = eval_weights(TUNED_W, "tuned_pilot")
out = pd.concat([res_default, res_tuned])
print(out.round(4).to_string(index=False))
print()

# =================================================================
# STEP 4. ResNet18만 / CNNSimple만 분리해서 r — 모델 효과 분리
# =================================================================
print("=" * 78)
print("[STEP 4] model별 r (default 가중치)")
print("=" * 78)
df_d = merged.copy()
df_d["score_w"] = df_d.apply(lambda r: score_from_weights(r, DEFAULT_W), axis=1)
for (ds, model), sub in df_d.groupby(["dataset", "model"]):
    if sub["accuracy"].std() == 0:
        continue
    r, p = stats.pearsonr(sub["score_w"], sub["accuracy"])
    print(f"{ds:>15} | {model:<12} | n={len(sub):>3} | r={r:+.4f} | p={p:.4f}")
print()

# =================================================================
# STEP 5. polluter별 score-acc rank — 어떤 오염이 진단 가능한가
# =================================================================
print("=" * 78)
print("[STEP 5] polluter별 평균 accuracy + score (default w, ResNet18만)")
print("=" * 78)
r18 = df_d[df_d.model == "ResNet18"].copy()
agg = r18.groupby(["dataset", "polluter"]).agg(
    acc_mean=("accuracy", "mean"),
    score_mean=("score_w", "mean"),
    n=("accuracy", "count"),
).round(4)
print(agg.to_string())
