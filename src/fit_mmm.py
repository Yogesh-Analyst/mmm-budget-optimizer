"""Fit the Bayesian MMM for one dataset and save everything the report and app need.

Usage:  python fit_mmm.py robyn|edtech

Steps
1. Holdout check: fit on the first 80% of weeks, forecast the last 20%.
2. Full fit on all weeks.
3. Save posterior parameters, ROI per channel, revenue decomposition,
   fit diagnostics and (for simulated data) estimated-vs-true ROI.
"""
import json
import sys
import warnings

import arviz as az
import numpy as np
import pandas as pd
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

from config import DATASETS, HOLDOUT_SHARE, L_MAX, OUTPUTS, SEED
from transforms import geometric_adstock, logistic_saturation

warnings.filterwarnings("ignore")
SAMPLER = dict(chains=4, draws=1000, tune=1500, target_accept=0.95)


def build_model(cfg):
    return MMM(
        date_column="date",
        channel_columns=cfg["channels"],
        target_column=cfg["target"],
        adstock=GeometricAdstock(l_max=L_MAX),
        saturation=LogisticSaturation(),
        control_columns=cfg["controls"],
        yearly_seasonality=2,
    )


def fit(cfg, X, y):
    model = build_model(cfg)
    model.fit(X, y, random_seed=SEED, progressbar=False, **SAMPLER)
    return model


def mape(actual, pred):
    return float(np.mean(np.abs((actual - pred) / actual)))


def diagnostics(model):
    post = model.idata.posterior.to_dataset()
    core = ["adstock_alpha", "saturation_lam", "saturation_beta", "gamma_control",
            "intercept_contribution", "y_sigma"]
    summ = az.summary(post[core])
    divergences = int(model.idata.sample_stats.to_dataset()["diverging"].sum())
    return {
        "max_rhat": float(summ["r_hat"].max()),
        "min_ess_bulk": float(summ["ess_bulk"].min()),
        "divergences": divergences,
    }


def interval(a, axis=0):
    """Mean and central 94% interval (3rd-97th percentile)."""
    return a.mean(axis=axis), np.percentile(a, 3, axis=axis), np.percentile(a, 97, axis=axis)


def main(name):
    cfg = DATASETS[name]
    out = OUTPUTS / name
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(cfg["file"], parse_dates=["date"])
    channels = cfg["channels"]
    X = df[["date"] + channels + cfg["controls"]]
    y = df[cfg["target"]]
    n_train = int(len(df) * (1 - HOLDOUT_SHARE))

    # 1. Holdout check
    print(f"[{name}] fitting on first {n_train} of {len(df)} weeks ...")
    m_hold = fit(cfg, X.iloc[:n_train], y.iloc[:n_train])
    scale = float(m_hold.idata.constant_data["target_scale"])
    pp = m_hold.sample_posterior_predictive(
        X.iloc[n_train:], include_last_observations=True, extend_idata=False, progressbar=False)
    test_draws = pp["y"].values.T * scale                       # (samples, weeks)
    test_mean, test_lo, test_hi = interval(test_draws)
    holdout_mape = mape(y.iloc[n_train:].values, test_mean)

    # 2. Full fit
    print(f"[{name}] fitting on all weeks ...")
    m = fit(cfg, X, y)
    post = m.idata.posterior.to_dataset().stack(sample=("chain", "draw"))
    cd = m.idata.constant_data.to_dataset()
    target_scale = float(cd["target_scale"])
    channel_scale = cd["channel_scale"].values

    mu = post["mu"].transpose("sample", "date").values * target_scale
    fit_mean, fit_lo, fit_hi = interval(mu)
    in_sample_mape = mape(y.values, fit_mean)
    r2 = 1 - np.sum((y.values - fit_mean) ** 2) / np.sum((y.values - y.mean()) ** 2)

    fit_df = pd.DataFrame({
        "date": df["date"], "actual": y,
        "fitted": fit_mean, "fitted_lo": fit_lo, "fitted_hi": fit_hi,
        "holdout_pred": np.r_[np.full(n_train, np.nan), test_mean],
        "holdout_lo": np.r_[np.full(n_train, np.nan), test_lo],
        "holdout_hi": np.r_[np.full(n_train, np.nan), test_hi],
    })
    fit_df.to_csv(out / "fit.csv", index=False)

    # Posterior parameters (thinned to 1,000 draws) for the optimizer and app
    take = np.random.default_rng(SEED).choice(post.sizes["sample"], 1000, replace=False)
    params = {
        "alpha": post["adstock_alpha"].transpose("sample", "channel").values[take],
        "lam": post["saturation_lam"].transpose("sample", "channel").values[take],
        "beta": post["saturation_beta"].transpose("sample", "channel").values[take],
        "channel_scale": channel_scale,
        "target_scale": np.array(target_scale),
    }

    # Check our NumPy transforms reproduce the model's channel contributions
    contrib = post["channel_contribution"].transpose("sample", "date", "channel").values * target_scale
    xs = df[channels].values / channel_scale
    a, l, b = (params[k].mean(0) for k in ("alpha", "lam", "beta"))
    s0 = take[0]
    manual = np.column_stack([
        post["saturation_beta"].values[i, s0] * logistic_saturation(
            geometric_adstock(xs[:, i], post["adstock_alpha"].values[i, s0], L_MAX),
            post["saturation_lam"].values[i, s0])
        for i in range(len(channels))]) * target_scale
    max_gap = float(np.abs(manual - contrib[s0]).max() / target_scale)
    assert max_gap < 1e-6, f"NumPy transform mismatch: {max_gap}"

    # ROI per channel = revenue contributed / spend, per posterior draw
    spend_total = df[channels].sum().values
    roi_draws = contrib.sum(axis=1) / spend_total               # (samples, channels)
    roi_mean, roi_lo, roi_hi = interval(roi_draws)
    contrib_total = contrib.sum(axis=1).mean(0)
    roi_df = pd.DataFrame({
        "channel": channels, "spend": spend_total, "contribution": contrib_total,
        "roi": roi_mean, "roi_lo": roi_lo, "roi_hi": roi_hi,
        "alpha": a, "lam": l, "beta": b,
    })
    if cfg["truth"]:
        truth = json.load(open(cfg["truth"]))
        roi_df["true_roi"] = [truth["roi"][c] for c in channels]
        roi_df["true_alpha"] = [truth["alpha"][c] for c in channels]
        roi_df["roi_error_pct"] = (roi_df.roi / roi_df.true_roi - 1) * 100
        roi_df["true_in_interval"] = (roi_df.true_roi >= roi_df.roi_lo) & (roi_df.true_roi <= roi_df.roi_hi)
    roi_df.to_csv(out / "roi.csv", index=False)

    # Revenue decomposition (posterior means per week)
    def mean_of(var):
        v = post[var]
        extra = [d for d in v.dims if d not in ("sample", "date")]
        v = v.sum(extra) if extra else v
        return v.mean("sample").values * target_scale

    decomp = pd.DataFrame({"date": df["date"]})
    decomp["baseline"] = mean_of("intercept_contribution") * np.ones(len(df))
    decomp["seasonality"] = mean_of("yearly_seasonality_contribution")
    decomp["controls"] = mean_of("control_contribution")
    for i, c in enumerate(channels):
        decomp[c] = contrib[:, :, i].mean(0)
    decomp["actual"] = y.values
    decomp.to_csv(out / "decomposition.csv", index=False)

    np.savez(out / "posterior_params.npz", **params,
             channels=np.array(channels),
             current_weekly_spend=df[channels].iloc[-52:].mean().values,
             max_weekly_spend=df[channels].max().values)

    metrics = {
        "dataset": cfg["title"],
        "weeks": len(df), "train_weeks": n_train, "holdout_weeks": len(df) - n_train,
        "in_sample_mape": in_sample_mape, "in_sample_r2": float(r2),
        "holdout_mape": holdout_mape,
        "media_share_of_revenue": float(contrib.sum(axis=(1, 2)).mean() / y.sum()),
        **diagnostics(m),
    }
    json.dump(metrics, open(out / "metrics.json", "w"), indent=2)
    print(json.dumps(metrics, indent=2))
    print(roi_df.round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
