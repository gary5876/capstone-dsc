# -*- coding: utf-8 -*-
"""텍스트 셀 DEV 임시 차트 생성 (정식 full sweep 아님 — dev 사이즈).
입력: results/text_dsc_sweep.csv + results/text_train_metrics_dev.csv
출력: results/charts_text_dev/{분류,회귀}_*.png
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).resolve().parents[2]
OUT = BASE / "results" / "charts_text_dev"
OUT.mkdir(parents=True, exist_ok=True)

dsc = pd.read_csv(BASE / "results" / "text_dsc_sweep.csv")
met = pd.read_csv(BASE / "results" / "text_train_metrics_dev.csv")

keys = ["dataset", "task", "polluter", "level", "seed"]
df = met.merge(dsc[keys + ["dsc_score"]], on=keys, how="inner")
df = df.dropna(subset=["dsc_score", "metric"])
# 회귀 R²는 음수 가능 → 0 클립(차트 가독성), 분류 F1은 그대로
df["perf"] = df["metric"].clip(lower=0)

TAG = "DEV — 임시(정식 full sweep 아님)"


def scatter(sub, perf_label, title, fname):
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for model, g in sub.groupby("model"):
        ax.scatter(g["dsc_score"], g["perf"], s=70, alpha=0.8, label=model, edgecolor="k", linewidth=0.4)
    if len(sub) >= 3:
        r, p = pearsonr(sub["dsc_score"], sub["perf"])
        z = np.polyfit(sub["dsc_score"], sub["perf"], 1)
        xs = np.linspace(sub["dsc_score"].min(), sub["dsc_score"].max(), 50)
        ax.plot(xs, np.polyval(z, xs), "r--", lw=1.3)
        ax.text(0.04, 0.94, f"Pearson r = {r:.3f} (p={p:.3g}, n={len(sub)})",
                transform=ax.transAxes, va="top", fontsize=10,
                bbox=dict(boxstyle="round", fc="#fff7d6", ec="#caa"))
    ax.set_xlabel("DSC Score")
    ax.set_ylabel(perf_label)
    ax.set_title(f"{title}\n[{TAG}]", fontsize=11)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=130)
    plt.close(fig)
    print("saved", fname, "rows", len(sub))


def level_trend(sub, perf_label, title, fname):
    fig, ax1 = plt.subplots(figsize=(6.4, 4.8))
    # 평균 DSC·성능을 level별로
    agg = sub.groupby("level").agg(dsc=("dsc_score", "mean"), perf=("perf", "mean")).reset_index()
    ax1.plot(agg["level"], agg["dsc"], "o-", color="#2c7fb8", label="DSC Score")
    ax1.set_xlabel("오염 강도 (level)")
    ax1.set_ylabel("DSC Score", color="#2c7fb8")
    ax1.tick_params(axis="y", labelcolor="#2c7fb8")
    ax2 = ax1.twinx()
    ax2.plot(agg["level"], agg["perf"], "s--", color="#d95f0e", label=perf_label)
    ax2.set_ylabel(perf_label, color="#d95f0e")
    ax2.tick_params(axis="y", labelcolor="#d95f0e")
    ax1.set_title(f"{title}\n[{TAG}]", fontsize=11)
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=130)
    plt.close(fig)
    print("saved", fname, "rows", len(sub))


clf = df[df["task"] == "classification"]
reg = df[df["task"] == "regression"]

scatter(clf, "F1 (macro)", "텍스트 분류: DSC vs F1 (ag_news)", "분류_01_scatter.png")
level_trend(clf, "F1 (macro)", "텍스트 분류: 오염강도별 DSC·F1", "분류_02_level_trend.png")
scatter(reg, "R² (0 클립)", "텍스트 회귀: DSC vs R² (sst5)", "회귀_01_scatter.png")
level_trend(reg, "R² (0 클립)", "텍스트 회귀: 오염강도별 DSC·R²", "회귀_02_level_trend.png")

print("\n=== 요약 ===")
for name, sub in [("분류", clf), ("회귀", reg)]:
    if len(sub) >= 3:
        r, p = pearsonr(sub["dsc_score"], sub["perf"])
        print(f"{name}: n={len(sub)}, r={r:.3f}, p={p:.3g}, datasets={list(sub.dataset.unique())}, models={list(sub.model.unique())}")
