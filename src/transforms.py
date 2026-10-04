"""NumPy versions of the media transforms used by the model.

They match PyMC-Marketing's GeometricAdstock(normalize=True) and
LogisticSaturation, so the simulator, the optimizer and the Streamlit app
can compute responses without loading PyMC.
"""
import numpy as np


def geometric_adstock(x, alpha, l_max):
    """Carry each week's spend into the next l_max - 1 weeks with decay alpha.

    x: (weeks,) or (weeks, channels); alpha: scalar or (channels,).
    Weights are normalised to sum to 1, so a constant spend level is unchanged.
    """
    x = np.asarray(x, dtype=float)
    alpha = np.atleast_1d(alpha)
    weights = alpha[None, :] ** np.arange(l_max)[:, None]   # (l_max, channels)
    weights = weights / weights.sum(axis=0)
    x2 = x.reshape(len(x), -1)
    out = np.zeros_like(x2)
    for lag in range(l_max):
        out[lag:] += weights[lag] * x2[: len(x2) - lag]
    return out.reshape(x.shape)


def logistic_saturation(x, lam):
    """Diminishing returns: 0 at x=0, approaching 1 as x grows."""
    return (1 - np.exp(-lam * x)) / (1 + np.exp(-lam * x))


def steady_state_response(weekly_spend, params):
    """Weekly revenue from holding `weekly_spend` constant, per posterior draw.

    weekly_spend: (channels,) in currency. params: dict with
    alpha/lam/beta of shape (draws, channels), channel_scale (channels,)
    and target_scale (scalar). With normalised adstock a constant spend
    level passes through unchanged, so only saturation matters.
    Returns (draws, channels) revenue contribution per week.
    """
    x = np.asarray(weekly_spend, dtype=float) / params["channel_scale"]
    return params["beta"] * logistic_saturation(x[None, :], params["lam"]) * params["target_scale"]
