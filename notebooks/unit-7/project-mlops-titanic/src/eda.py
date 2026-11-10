"""Exploratory data analysis: writes summary stats and figures to reports/eda."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import EDA_DIR, TARGET
from src.data import load_raw
from src.features import engineer_features

BLUE, ORANGE, GREY = "#2f6f9f", "#d9822b", "#8a8f98"


def _rate_bar(ax, series, title, xlabel):
    series.plot.bar(ax=ax, color=BLUE, width=0.7)
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Survival rate")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=0)
    for i, v in enumerate(series):
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    df = load_raw()
    feats = engineer_features(df).assign(**{TARGET: df[TARGET].values})

    q1, q3 = df["Fare"].quantile([0.25, 0.75])
    fare_fence = q3 + 1.5 * (q3 - q1)
    summary = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "survival_rate": round(float(df[TARGET].mean()), 4),
        "missing_counts": {k: int(v) for k, v in df.isna().sum().items() if v > 0},
        "missing_pct": {k: round(float(v) * 100, 1) for k, v in df.isna().mean().items() if v > 0},
        "fare_iqr_upper_fence": round(float(fare_fence), 2),
        "fare_outliers_above_fence": int((df["Fare"] > fare_fence).sum()),
        "fare_max": float(df["Fare"].max()),
        "fare_skew": round(float(df["Fare"].skew()), 2),
        "survival_by_sex": df.groupby("Sex")[TARGET].mean().round(3).to_dict(),
        "survival_by_pclass": {int(k): v for k, v in df.groupby("Pclass")[TARGET].mean().round(3).items()},
        "survival_by_title": feats.groupby("Title")[TARGET].mean().round(3).to_dict(),
        "survival_by_is_alone": {int(k): v for k, v in feats.groupby("IsAlone")[TARGET].mean().round(3).items()},
        "survival_by_has_cabin": {int(k): v for k, v in feats.groupby("HasCabin")[TARGET].mean().round(3).items()},
    }
    (EDA_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2))

    # Figure 1: missing values
    fig, ax = plt.subplots(figsize=(6, 3))
    miss = df.isna().mean().mul(100).loc[lambda s: s > 0].sort_values()
    miss.plot.barh(ax=ax, color=BLUE)
    for i, v in enumerate(miss):
        ax.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Missing (%)")
    ax.set_title("Columns with missing values", loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(EDA_DIR / "missing_values.png", dpi=150); plt.close(fig)

    # Figure 2: survival rate by key groups
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    _rate_bar(axes[0], df.groupby("Sex")[TARGET].mean(), "By sex", "")
    _rate_bar(axes[1], df.groupby("Pclass")[TARGET].mean(), "By passenger class", "")
    _rate_bar(axes[2], feats.groupby("Title")[TARGET].mean().sort_values(), "By title (engineered)", "")
    fig.tight_layout(); fig.savefig(EDA_DIR / "survival_by_group.png", dpi=150); plt.close(fig)

    # Figure 3: age and fare distributions, fare before and after log transform
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for label, color in [(0, GREY), (1, ORANGE)]:
        axes[0].hist(df.loc[df[TARGET] == label, "Age"].dropna(), bins=25, alpha=0.65, color=color,
                     label="Survived" if label else "Did not survive")
    axes[0].set_title("Age by outcome", loc="left", fontsize=11); axes[0].set_xlabel("Age"); axes[0].legend(frameon=False, fontsize=8)
    axes[1].hist(df["Fare"], bins=40, color=BLUE)
    axes[1].axvline(fare_fence, color=ORANGE, ls="--", lw=1)
    axes[1].text(fare_fence + 8, axes[1].get_ylim()[1] * 0.85, "IQR fence", color=ORANGE, fontsize=8)
    axes[1].set_title("Fare (raw, right-skewed)", loc="left", fontsize=11); axes[1].set_xlabel("Fare")
    axes[2].hist(feats["FareLog"], bins=40, color=BLUE)
    axes[2].set_title("Fare after log1p", loc="left", fontsize=11); axes[2].set_xlabel("log1p(Fare)")
    for a in axes:
        a.set_ylabel("Passengers"); a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(EDA_DIR / "distributions.png", dpi=150); plt.close(fig)

    print(f"EDA done: {summary['rows']} rows, survival rate {summary['survival_rate']:.1%}, "
          f"missing {summary['missing_pct']}. Figures in {EDA_DIR.name}/.")


if __name__ == "__main__":
    main()
