"""Build the two modelling datasets.

1. robyn_weekly.csv: Meta Robyn's public simulated weekly dataset (208 weeks).
2. edtech_india_synthetic.csv: a made-up Indian ed-tech channel mix generated
   from KNOWN adstock, saturation and ROI values, so we can check whether the
   model recovers the truth. Contains no real company data.
"""
import json
import urllib.request

import holidays
import numpy as np
import pandas as pd
import pyreadr

from config import DATA, L_MAX, SEED
from transforms import geometric_adstock, logistic_saturation

ROBYN_URL = ("https://raw.githubusercontent.com/facebookexperimental/Robyn/"
             "main/R/data/dt_simulated_weekly.RData")


def build_robyn():
    raw = DATA / "robyn.RData"
    if not raw.exists():
        urllib.request.urlretrieve(ROBYN_URL, raw)
    df = pyreadr.read_r(raw)["dt_simulated_weekly"]
    out = pd.DataFrame({
        "date": pd.to_datetime(df["DATE"]),
        "revenue": df["revenue"],
        "tv": df["tv_S"],
        "ooh": df["ooh_S"],
        "print": df["print_S"],
        "facebook": df["facebook_S"],
        "search": df["search_S"],
        # Controls are scaled to 0-1 because PyMC-Marketing does not scale them.
        "competitor_sales": df["competitor_sales_B"] / df["competitor_sales_B"].max(),
        "newsletter": df["newsletter"] / df["newsletter"].max(),
    })
    out["trend"] = np.linspace(0, 1, len(out))
    # The data is German (Robyn's demo uses DE holidays): flag weeks containing a public holiday.
    de = holidays.Germany(years=range(2015, 2020))
    out["holiday"] = [float(any(d + pd.Timedelta(days=k) in de for k in range(7))) for d in out["date"]]
    out.to_csv(DATA / "robyn_weekly.csv", index=False)
    print(f"robyn_weekly.csv: {len(out)} weeks")


def build_edtech():
    rng = np.random.default_rng(SEED)
    n = 156  # three years of weeks
    dates = pd.date_range("2023-10-02", periods=n, freq="W-MON")
    doy = dates.dayofyear.values
    # Admission seasons: Jan-Mar and Jun-Jul push spend up.
    admissions = np.isin(dates.month, [1, 2, 3, 6, 7]).astype(float)

    def lognoise(sd):
        return rng.lognormal(0, sd, n)

    lakh = 1e5
    spend = pd.DataFrame(index=dates)
    spend["meta"] = 9 * lakh * (1 + 0.35 * admissions) * lognoise(0.30)
    spend["google_search"] = 6 * lakh * (1 + 0.20 * admissions) * lognoise(0.25)
    pmax = 4 * lakh * lognoise(0.35)
    pmax[:30] = 0  # Performance Max campaigns launched ~7 months in
    spend["pmax"] = pmax
    flight = ((np.arange(n) // 4) % 2 == 0).astype(float)  # YouTube runs 4 weeks on, 4 off
    spend["youtube"] = 5 * lakh * flight * lognoise(0.25)
    spend["linkedin"] = 1.5 * lakh * lognoise(0.30)
    channels = list(spend.columns)

    true = {
        "alpha": {"meta": 0.25, "google_search": 0.10, "pmax": 0.20, "youtube": 0.55, "linkedin": 0.35},
        "lam": {"meta": 2.5, "google_search": 3.5, "pmax": 2.0, "youtube": 1.5, "linkedin": 2.0},
        "roi": {"meta": 2.0, "google_search": 3.0, "pmax": 1.5, "youtube": 0.9, "linkedin": 0.6},
    }

    contrib = pd.DataFrame(index=dates)
    scales = spend.max()
    true["effect_max"], true["channel_scale"] = {}, scales.to_dict()
    for c in channels:
        ad = geometric_adstock(spend[c].values / scales[c], true["alpha"][c], L_MAX)
        shape = logistic_saturation(ad, true["lam"][c])
        # Pick the channel's max weekly effect so its realised ROI equals the target.
        effect_max = true["roi"][c] * spend[c].sum() / shape.sum()
        contrib[c] = effect_max * shape
        true["effect_max"][c] = float(effect_max)

    fee_offer = np.zeros(n)
    fee_offer[[14, 15, 40, 41, 66, 67, 92, 93, 118, 119, 144, 145]] = 1  # two-week fee-discount offers
    trend = np.linspace(0, 1, n)
    base = (45 * lakh + 10 * lakh * trend
            + 4 * lakh * np.sin(2 * np.pi * doy / 365.25)
            + 3 * lakh * np.cos(2 * np.pi * doy / 365.25))
    revenue = base + 8 * lakh * fee_offer + contrib.sum(axis=1).values
    revenue = revenue + rng.normal(0, 0.04 * revenue.mean(), n)

    out = spend.round(0)
    out.insert(0, "revenue", revenue.round(0))
    out["fee_offer"] = fee_offer
    out["trend"] = trend
    out.index.name = "date"
    out.reset_index().to_csv(DATA / "edtech_india_synthetic.csv", index=False)

    true["contribution_total"] = contrib.sum().round(0).to_dict()
    true["spend_total"] = spend.sum().round(0).to_dict()
    with open(DATA / "edtech_true_params.json", "w") as f:
        json.dump(true, f, indent=2)
    share = contrib.sum().sum() / revenue.sum()
    print(f"edtech_india_synthetic.csv: {n} weeks, paid media = {share:.0%} of revenue")


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    build_robyn()
    build_edtech()
