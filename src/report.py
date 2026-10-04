"""Charts for the README.  Usage:  python report.py robyn|edtech"""
import json
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CHANNEL_LABELS, DATASETS, OUTPUTS
from optimizer import load_params
from transforms import steady_state_response

matplotlib.use("Agg")

# Validated categorical slots, fixed per channel position (never cycled).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, MUTED, GRID, SURFACE, BASE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb", "#c9c8c2"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "figure.dpi": 150,
})


def money(cur):
    def fmt(v, _=None):
        if cur == "₹":
            return f"₹{v / 1e7:.1f} Cr" if abs(v) >= 1e7 else f"₹{v / 1e5:.1f} L"
        return f"${v / 1e6:.1f}M" if abs(v) >= 1e6 else f"${v / 1e3:g}K"
    return fmt


def fit_chart(name, cfg, out):
    f = pd.read_csv(out / "fit.csv", parse_dates=["date"])
    m = json.load(open(out / "metrics.json"))
    fig, ax = plt.subplots(figsize=(10, 4))
    h0 = f["holdout_pred"].first_valid_index()
    ax.axvspan(f.date[h0], f.date.iloc[-1], color=GRID, alpha=0.5, lw=0)
    ax.plot(f.date, f.actual, color=INK, lw=1.5, label="Actual revenue")
    ax.plot(f.date[:h0], f.fitted[:h0], color=SERIES[0], lw=2, label="Model fit")
    ax.fill_between(f.date, f.holdout_lo, f.holdout_hi, color=SERIES[1], alpha=0.2, lw=0)
    ax.plot(f.date, f.holdout_pred, color=SERIES[1], lw=2, label="Forecast on unseen weeks (94% band)")
    ax.text(f.date[h0], ax.get_ylim()[1], "  holdout", va="top", color=INK2)
    ax.yaxis.set_major_formatter(money(cfg["currency"]))
    ax.set_title(f"Model fit: R² {m['in_sample_r2']:.2f}, holdout error (MAPE) {m['holdout_mape']:.1%}")
    ax.legend(loc="upper left", ncol=3, bbox_to_anchor=(0, -0.08))
    fig.tight_layout()
    fig.savefig(out / "fit_holdout.png")


def roi_chart(name, cfg, out):
    r = pd.read_csv(out / "roi.csv").iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    idx = {c: i for i, c in enumerate(cfg["channels"])}
    for i, row in r.iterrows():
        col = SERIES[idx[row.channel]]
        ax.plot([row.roi_lo, row.roi_hi], [i, i], color=col, lw=3, solid_capstyle="round", alpha=0.45)
        ax.plot(row.roi, i, "o", color=col, ms=9, mec=SURFACE, mew=2)
        ax.text(row.roi_hi, i, f"  {row.roi:.2f}", va="center", color=INK2)
        if "true_roi" in r:
            ax.plot(row.true_roi, i, "D", color=INK, ms=7, mec=SURFACE, mew=1.5,
                    label="True ROI (known in simulation)" if i == 0 else None)
    ax.axvline(1, color=MUTED, lw=1, ls="--")
    ax.text(1, -0.45, " break-even", color=MUTED, fontsize=9, va="center")
    ax.set_yticks(range(len(r)), [CHANNEL_LABELS[c] for c in r.channel])
    ax.set_xlabel("Revenue per unit of spend (dot = estimate, bar = 94% credible interval)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Return on spend by channel")
    if "true_roi" in r:
        ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "roi.png")


def decomposition_chart(name, cfg, out):
    d = pd.read_csv(out / "decomposition.csv", parse_dates=["date"])
    base = d[["baseline", "seasonality", "controls"]].sum(axis=1)
    layers = [base] + [d[c] for c in cfg["channels"]]
    labels = ["Base (non-media)"] + [CHANNEL_LABELS[c] for c in cfg["channels"]]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.stackplot(d.date, *layers, labels=labels, colors=[BASE] + SERIES,
                 edgecolor=SURFACE, linewidth=0.6)
    ax.plot(d.date, d.actual, color=INK, lw=1.2, label="Actual revenue")
    ax.yaxis.set_major_formatter(money(cfg["currency"]))
    ax.set_title("What drove weekly revenue: base demand vs each channel")
    ax.legend(loc="upper left", ncol=4, bbox_to_anchor=(0, -0.08))
    fig.tight_layout()
    fig.savefig(out / "decomposition.png")


def response_chart(name, cfg, out):
    p = load_params(name)
    plan = pd.read_csv(out / "budget_plan.csv")
    n = len(cfg["channels"])
    fig, axes = plt.subplots(1, n, figsize=(13, 3.2))
    fmt = money(cfg["currency"])
    for i, (ax, c) in enumerate(zip(axes, cfg["channels"])):
        # Show the decision-relevant range rather than rare spike weeks.
        xmax = 2.5 * max(plan.current_weekly_spend[i], plan.optimized_weekly_spend[i])
        grid = np.linspace(0, xmax, 80)
        resp = np.array([steady_state_response(np.where(np.arange(n) == i, g, 0), p)[:, i] for g in grid])
        ax.fill_between(grid, np.percentile(resp, 3, 1), np.percentile(resp, 97, 1), color=SERIES[i], alpha=0.18, lw=0)
        ax.plot(grid, resp.mean(1), color=SERIES[i], lw=2)
        for x, mk, lab in [(plan.current_weekly_spend[i], "o", "Current"), (plan.optimized_weekly_spend[i], "D", "Optimized")]:
            y = steady_state_response(np.where(np.arange(n) == i, x, 0), p)[:, i].mean()
            ax.plot(x, y, mk, color=INK if mk == "D" else SERIES[i], ms=8, mec=SURFACE, mew=2)
        ax.set_title(CHANNEL_LABELS[c], fontsize=11)
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)
        ax.tick_params(labelsize=8)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    axes[0].set_ylabel("Weekly revenue from channel")
    fig.supxlabel("Weekly spend", color=INK2, fontsize=10)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=MUTED, ms=8, label="Current spend"),
                        Line2D([], [], marker="D", ls="", color=INK, ms=7, label="Optimized spend")],
               loc="upper right", ncol=2)
    fig.suptitle("Response curves: diminishing returns per channel (94% band)", x=0.01, ha="left",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "response_curves.png")


def budget_chart(name, cfg, out):
    plan = pd.read_csv(out / "budget_plan.csv").iloc[::-1].reset_index(drop=True)
    s = json.load(open(out / "budget_summary.json"))
    fmt = money(cfg["currency"])
    y = np.arange(len(plan))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.barh(y + 0.2, plan.current_weekly_spend, 0.38, color=BASE, label="Current", edgecolor=SURFACE, lw=2)
    ax.barh(y - 0.2, plan.optimized_weekly_spend, 0.38, color=SERIES[0], label="Optimized", edgecolor=SURFACE, lw=2)
    for i, row in plan.iterrows():
        ax.text(row.optimized_weekly_spend, i - 0.2,
                f"  {row.change_pct:+.0f}%", va="center", color=INK2)
    ax.set_yticks(y, [CHANNEL_LABELS[c] for c in plan.channel])
    ax.xaxis.set_major_formatter(fmt)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Weekly spend (same total budget)")
    ax.set_xlim(0, plan[["current_weekly_spend", "optimized_weekly_spend"]].values.max() * 1.15)
    ax.set_title(f"Same budget, re-split: +{fmt(s['uplift_mean'])} media revenue per week\n"
                 f"{s['prob_uplift_positive']:.0%} of posterior draws show a gain")
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig(out / "budget_shift.png")


def main(name):
    cfg, out = DATASETS[name], OUTPUTS / name
    for chart in (fit_chart, roi_chart, decomposition_chart, response_chart, budget_chart):
        chart(name, cfg, out)
        plt.close("all")
    print(f"charts saved to {out}")


if __name__ == "__main__":
    main(sys.argv[1])
