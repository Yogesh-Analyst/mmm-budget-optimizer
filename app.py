"""Streamlit budget simulator.  Run:  streamlit run app.py

Reads the posterior draws saved by src/fit_mmm.py, so it needs only
NumPy/SciPy at runtime (no PyMC), which keeps it deployable for free.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from config import CHANNEL_LABELS, DATASETS, OUTPUTS  # noqa: E402
from optimizer import compare, load_params, marginal_roi, optimize  # noqa: E402

st.set_page_config(page_title="MMM Budget Optimizer", layout="wide")
st.title("Marketing Mix Model: budget optimizer")
st.caption("Bayesian MMM (PyMC-Marketing) with adstock and saturation. "
           "Move the sliders to re-split a weekly budget and see the expected revenue change, with uncertainty.")

name = st.sidebar.radio("Dataset", list(DATASETS), format_func=lambda k: DATASETS[k]["title"])
cfg = DATASETS[name]
cur_sym = cfg["currency"]
p = load_params(name)
channels = list(p["channels"])
labels = [CHANNEL_LABELS[c] for c in channels]
current = p["current_weekly_spend"]


def money(v):
    if cur_sym == "₹":
        return f"₹{v / 1e5:,.2f} L"
    return f"${v:,.0f}"


st.sidebar.header("Budget")
budget_pct = st.sidebar.slider("Total weekly budget vs current", 50, 150, 100, 5, format="%d%%")
budget = current.sum() * budget_pct / 100
st.sidebar.write(f"Weekly budget: **{money(budget)}** (current {money(current.sum())})")

st.sidebar.header("Channel limits (vs current spend)")
lo, hi = np.zeros(len(channels)), np.zeros(len(channels))
for i, (c, lab) in enumerate(zip(channels, labels)):
    lo_pct, hi_pct = st.sidebar.slider(lab, 0, 200, (50, 150), 10, format="%d%%", key=f"{name}-{c}")
    lo[i] = current[i] * lo_pct / 100
    # Never go past the biggest week in the data: beyond that the curve is a guess.
    hi[i] = max(min(current[i] * hi_pct / 100, p["max_weekly_spend"][i]), lo[i])

if lo.sum() > budget or hi.sum() < budget:
    st.error(f"The channel limits allow {money(lo.sum())}-{money(hi.sum())} per week, "
             f"so a budget of {money(budget)} cannot be met. Widen the limits or change the budget.")
    st.stop()

plan = optimize(p, budget, lo, hi)
same_mix = current * budget / current.sum()   # today's channel mix at the chosen budget
res = compare(p, plan, baseline=same_mix)

c1, c2, c3 = st.columns(3)
c1.metric("Media revenue / week, current mix", money(res["baseline_media_revenue"]))
c2.metric("Media revenue / week, optimized", money(res["plan_media_revenue"]),
          f"{res['uplift_mean'] / res['baseline_media_revenue']:+.1%}")
c3.metric("Probability it beats the current mix", f"{res['prob_uplift_positive']:.0%}")
st.caption(f"94% credible interval for the weekly change: {money(res['uplift_lo'])} to {money(res['uplift_hi'])}.")

table = pd.DataFrame({
    "Channel": labels,
    "Current mix": same_mix,
    "Recommended spend": plan,
    "Change": plan / same_mix - 1,
    "Marginal ROI now": marginal_roi(p, same_mix),
    "Marginal ROI after": marginal_roi(p, plan),
})
st.subheader("Recommended weekly split")
st.dataframe(
    table.style.format({"Current mix": money, "Recommended spend": money, "Change": "{:+.0%}",
                        "Marginal ROI now": "{:.2f}", "Marginal ROI after": "{:.2f}"}),
    hide_index=True, width="stretch")
st.caption("Marginal ROI = revenue from the next unit of spend. A good split moves money until these even out.")
st.bar_chart(table.set_index("Channel")[["Current mix", "Recommended spend"]], stack=False, horizontal=True)

out = OUTPUTS / name
st.subheader("Model results")
m = json.load(open(out / "metrics.json"))
k1, k2, k3, k4 = st.columns(4)
k1.metric("R²", f"{m['in_sample_r2']:.2f}")
k2.metric("Holdout error (MAPE)", f"{m['holdout_mape']:.1%}")
k3.metric("Revenue driven by media", f"{m['media_share_of_revenue']:.0%}")
k4.metric("Max R-hat (convergence)", f"{m['max_rhat']:.3f}")
for img, cap in [("roi.png", "Return on spend by channel"), ("response_curves.png", "Response curves"),
                 ("decomposition.png", "Revenue decomposition"), ("fit_holdout.png", "Fit and holdout forecast")]:
    st.image(str(out / img), caption=cap, width="stretch")
