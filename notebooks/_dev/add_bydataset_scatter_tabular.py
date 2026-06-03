# -*- coding: utf-8 -*-
"""정형 분류·회귀 산점도의 '데이터셋별 회귀선' 버전 신규 생성 (기존 pooled 파일은 유지).
새 파일: *_bydataset.png. 점은 데이터셋별 색, 선도 데이터셋별, 박스에 per-dataset r + pooled r.
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


def by_dataset(m, xcol, ycol, ylabel, title, out):
    fig, ax = plt.subplots(figsize=(10, 7))
    dsets = sorted(m["dataset"].unique())
    colors = sns.color_palette("Set1", len(dsets))
    lines = []
    for ds, color in zip(dsets, colors):
        sub = m[m["dataset"] == ds]
        x = sub[xcol].values.astype(float)
        y = sub[ycol].values.astype(float)
        msk = ~(np.isnan(x) | np.isnan(y))
        x, y = x[msk], y[msk]
        ax.scatter(x, y, label=ds, color=color, alpha=0.55, s=40,
                   edgecolors="white", linewidth=0.5)
        if len(x) >= 3 and x.std() > 1e-9:
            z = np.polyfit(x, y, 1)
            xl = np.linspace(x.min(), x.max(), 100)
            ax.plot(xl, np.poly1d(z)(xl), "--", color=color, alpha=0.95, linewidth=2.0)
            r, p = pearsonr(x, y)
            lines.append(f"{ds}: r={r:+.3f} (n={len(x)})")
    xp = m[xcol].values.astype(float); yp = m[ycol].values.astype(float)
    mk = ~(np.isnan(xp) | np.isnan(yp))
    rp, _ = pearsonr(xp[mk], yp[mk])
    txt = "데이터셋별 Pearson r\n" + "\n".join(lines) + f"\n──────\nPOOLED: r={rp:+.3f} (n={mk.sum()})"
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=11, va="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.65))
    ax.set_xlabel("DSC Score", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.legend(title="Dataset", loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out.parent.name}/{out.name}  " + " | ".join(lines))


# 정형 분류
mc = pd.read_csv(R / "merged_results.csv")
by_dataset(mc, "score", "f1_macro", "F1-score (macro)",
           "DSC Score vs F1 — 정형 분류 cell (데이터셋별 회귀선)",
           R / "charts" / "chart1_scatter_dsc_vs_f1_bydataset.png")

# 정형 회귀
perf = pd.read_csv(R / "model_performance_regression.csv")
dsc = pd.read_csv(R / "dsc_scores_regression.csv")
mr = perf.merge(dsc[["dataset", "polluter", "level", "score"]],
                on=["dataset", "polluter", "level"]).rename(columns={"score": "dsc_score"})
by_dataset(mr, "dsc_score", "r2_clipped", "R² (clipped)",
           "DSC Score vs R² — 정형 회귀 cell (데이터셋별 회귀선)",
           R / "charts_regression" / "01_dsc_vs_r2_scatter_bydataset.png")
