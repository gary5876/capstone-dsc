# -*- coding: utf-8 -*-
"""이미지 분류·회귀 산점도에 '데이터셋별' 회귀선 + Pearson r 추가.
이미지 셀은 합격 판정이 dataset 단위(배치효과로 pooled 선은 결과 왜곡) → 데이터셋별 선.
각 셀의 공식 집계 미러: 분류=전체 모델 원시점 / 회귀=probe_mean(config별 4모델 평균).
덮어쓰기: results/charts_image/01_scatter.png, results/charts_image_regression/01_scatter.png
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from pathlib import Path

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
BASE = Path(__file__).resolve().parents[2]
R = BASE / "results"


def scatter_by_dataset(m, xcol, ycol, ylabel, title, out, clip0=False):
    m = m.copy()
    if clip0:
        m[ycol] = m[ycol].clip(lower=0)
    fig, ax = plt.subplots(figsize=(10, 7))
    dsets = sorted(m["dataset"].unique())
    colors = sns.color_palette("Set1", len(dsets))
    lines = []
    for ds, color in zip(dsets, colors):
        sub = m[m["dataset"] == ds]
        x = sub[xcol].values.astype(float)
        y = sub[ycol].values.astype(float)
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]
        ax.scatter(x, y, label=ds, color=color, alpha=0.6, s=50,
                   edgecolors="white", linewidth=0.5)
        if len(x) >= 3 and x.std() > 1e-9:
            z = np.polyfit(x, y, 1)
            xl = np.linspace(x.min(), x.max(), 100)
            ax.plot(xl, np.poly1d(z)(xl), "--", color=color, alpha=0.9, linewidth=1.8)
            r, p = pearsonr(x, y)
            lines.append(f"{ds}: r={r:+.3f} (p={p:.1e}, n={len(x)})")
    ax.text(0.05, 0.95, "데이터셋별 Pearson r\n" + "\n".join(lines),
            transform=ax.transAxes, fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6))
    ax.set_xlabel("DSC Score", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.legend(title="Dataset", loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out.parent.name}/{out.name}  " + " | ".join(lines))


# 이미지 분류 — 전체 모델 원시점, accuracy
dsc_c = pd.read_csv(R / "dsc_scores_image.csv")
perf_c = pd.read_csv(R / "model_performance_image.csv")
mc = perf_c.merge(dsc_c[["dataset", "polluter", "level", "score", "grade"]],
                  on=["dataset", "polluter", "level"], how="inner")
scatter_by_dataset(mc, "score", "accuracy", "Accuracy",
                   "DSC Score vs Accuracy — image classification cell",
                   R / "charts_image" / "01_scatter.png")

# 이미지 회귀 — probe_mean(config별 4모델 평균 R²)
dsc_r = pd.read_csv(R / "dsc_scores_image_regression.csv")
perf_r = pd.read_csv(R / "model_performance_image_regression.csv")
probe = perf_r[perf_r["method"] == "probe"]
pm = probe.groupby(["dataset", "polluter", "level"])["score"].mean().reset_index(name="probe_R2")
mr = pm.merge(dsc_r[["dataset", "polluter", "level", "score", "grade"]],
              on=["dataset", "polluter", "level"], how="inner").rename(columns={"score": "dsc_score"})
scatter_by_dataset(mr, "dsc_score", "probe_R2", "probe 평균 R² (0 클립)",
                   "DSC Score vs R² — image regression cell",
                   R / "charts_image_regression" / "01_scatter.png", clip0=True)
