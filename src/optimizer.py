"""Reallocate a fixed weekly budget across channels to maximise expected revenue.

Uses the posterior draws saved by fit_mmm.py, so the answer comes with
uncertainty: for every draw we compute the revenue change of the new plan vs
the current one, and report the mean, a 94% interval and P(uplift > 0).

Usage:  python optimizer.py robyn|edtech
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config import DATASETS, OUTPUTS
from transforms import logistic_saturation, steady_state_response

MIN_SHARE = 0.5  # never cut a channel below 50% of its current spend
MAX_SHARE = 1.5  # nor raise it above 150% (or its historical max week)


def load_params(name):
    z = np.load(OUTPUTS / name / "posterior_params.npz")
    return {k: z[k] for k in z.files}


def default_bounds(p, min_share=MIN_SHARE, max_share=MAX_SHARE):
    cur = p["current_weekly_spend"]
    # Cap at the largest week seen: beyond it the response curve is extrapolation.
    hi = np.maximum(np.minimum(cur * max_share, p["max_weekly_spend"]), cur)
    return cur * min_share, hi


def solve(revenue_fn, budget, lo, hi, start):
    """Maximise revenue_fn(x) s.t. sum(x) == budget, lo <= x <= hi.

    Works in budget shares (0-1) with a normalised objective: SLSQP is
    unreliable when variables and objective are in raw currency units.
    """
    norm = abs(revenue_fn(start)) or 1.0
    res = minimize(
        lambda z: -revenue_fn(z * budget) / norm, start / budget, method="SLSQP",
        bounds=list(zip(lo / budget, hi / budget)),
        constraints=[{"type": "eq", "fun": lambda z: z.sum() - 1}],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not res.success:
        raise RuntimeError(res.message)
    return res.x * budget


def optimize(p, budget, lo, hi):
    """Best weekly allocation summing to `budget` within [lo, hi] per channel."""
    cur = p["current_weekly_spend"]
    start = np.clip(cur * budget / cur.sum(), lo, hi)
    return solve(lambda x: steady_state_response(x, p).sum(axis=1).mean(), budget, lo, hi, start)


def compare(p, plan, baseline=None):
    """Revenue of `plan` vs `baseline` (default: current spend), per posterior draw."""
    baseline = p["current_weekly_spend"] if baseline is None else baseline
    r_plan = steady_state_response(plan, p)
    r_base = steady_state_response(baseline, p)
    diff = r_plan.sum(1) - r_base.sum(1)
    return {
        "baseline_media_revenue": float(r_base.sum(1).mean()),
        "plan_media_revenue": float(r_plan.sum(1).mean()),
        "uplift_mean": float(diff.mean()),
        "uplift_lo": float(np.percentile(diff, 3)),
        "uplift_hi": float(np.percentile(diff, 97)),
        "prob_uplift_positive": float((diff > 0).mean()),
        "channel_revenue_baseline": r_base.mean(0),
        "channel_revenue_plan": r_plan.mean(0),
    }


def marginal_roi(p, spend, step=0.01):
    """Revenue from the next 1% of spend in each channel, per unit spent."""
    base = steady_state_response(spend, p).mean(0)
    bumped = steady_state_response(spend * (1 + step), p).mean(0)
    return (bumped - base) / (spend * step)


def true_weekly_revenue(truth, spend, channels):
    """Weekly media revenue under the simulation's true response curves."""
    return sum(truth["effect_max"][c] * logistic_saturation(s / truth["channel_scale"][c], truth["lam"][c])
               for c, s in zip(channels, spend))


def truth_check(truth, p, plan, lo, hi):
    """How much the model's plan really earns, vs the best plan possible with the true curves."""
    channels = list(p["channels"])
    cur = p["current_weekly_spend"]
    rev = lambda x: true_weekly_revenue(truth, x, channels)
    best = solve(rev, cur.sum(), lo, hi, cur)
    return {
        "true_uplift_of_model_plan": float(rev(plan) - rev(cur)),
        "true_uplift_of_best_plan": float(rev(best) - rev(cur)),
        "true_best_plan": dict(zip(channels, best.round(0).tolist())),
    }


def main(name):
    p = load_params(name)
    cur = p["current_weekly_spend"]
    lo, hi = default_bounds(p)
    plan = optimize(p, cur.sum(), lo, hi)
    cmp_ = compare(p, plan)
    table = pd.DataFrame({
        "channel": p["channels"],
        "current_weekly_spend": cur,
        "optimized_weekly_spend": plan,
        "change_pct": (plan / cur - 1) * 100,
        "lower_bound": lo, "upper_bound": hi,
        "marginal_roi_current": marginal_roi(p, cur),
        "marginal_roi_optimized": marginal_roi(p, plan),
        "revenue_current": cmp_["channel_revenue_baseline"],
        "revenue_optimized": cmp_["channel_revenue_plan"],
    })
    table.to_csv(OUTPUTS / name / "budget_plan.csv", index=False)
    summary = {k: v for k, v in cmp_.items() if not k.startswith("channel_")}
    summary["weekly_budget"] = float(cur.sum())
    summary["uplift_pct_of_media_revenue"] = summary["uplift_mean"] / summary["baseline_media_revenue"] * 100
    if DATASETS[name]["truth"]:
        summary.update(truth_check(json.load(open(DATASETS[name]["truth"])), p, plan, lo, hi))
    json.dump(summary, open(OUTPUTS / name / "budget_summary.json", "w"), indent=2)
    print(table.round(2).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
