"""Shared paths and dataset definitions."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"

L_MAX = 8            # weeks of carryover the adstock can model
HOLDOUT_SHARE = 0.2  # last 20% of weeks held out for validation
SEED = 42

DATASETS = {
    "robyn": {
        "title": "Meta Robyn benchmark (public, USD)",
        "file": DATA / "robyn_weekly.csv",
        "target": "revenue",
        "channels": ["tv", "ooh", "print", "facebook", "search"],
        "controls": ["competitor_sales", "newsletter", "trend", "holiday"],
        "currency": "$",
        "truth": None,
    },
    "edtech": {
        "title": "Indian ed-tech mix (simulated, INR)",
        "file": DATA / "edtech_india_synthetic.csv",
        "target": "revenue",
        "channels": ["meta", "google_search", "pmax", "youtube", "linkedin"],
        "controls": ["fee_offer", "trend"],
        "currency": "₹",
        "truth": DATA / "edtech_true_params.json",
    },
}

CHANNEL_LABELS = {
    "tv": "TV", "ooh": "Outdoor", "print": "Print", "facebook": "Facebook",
    "search": "Search", "meta": "Meta", "google_search": "Google Search",
    "pmax": "Performance Max", "youtube": "YouTube", "linkedin": "LinkedIn",
}
