# Polymarket "Next CEO of NVIDIA?" price forecaster (make.fun bounty
# polymarket-nvidia-ceo-forecast-01). Python 3.13, standard library only.
#
# Model (tuned on the six practice cases, practice pinball loss ~0.272 vs the
# house baseline's 0.363589; 90% coverage ~0.91, i.e. honest calibration):
#
#   * Median: persistence -- the price at the checkpoint. Prediction-market
#     prices are (near-)martingales at a 4h horizon; every drift we tried
#     (mean reversion to a 60-min mean, pull toward the cross-market sum of 1)
#     made the practice loss worse, so the median stays put.
#   * Spread: per-market, adaptive. sigma is the mean absolute 1-minute
#     return (x1.2533, the normal equivalent), blended 70% from the last 6h
#     with 30% from the last 24h. The 6h half tracks the current volatility
#     regime; the 24h half steadies it. A plain standard deviation is useless
#     here: one 20-point jump in 24h inflates it so far that the intervals
#     are far too wide at every quiet minute. Mean absolute move is
#     jump-resistant and won on all six practice cases, early and late.
#     Floor 0.0003, cap 0.008 price units per minute.
#   * Horizon shape: quantiles widen as h**0.45, slightly slower than a
#     random walk's sqrt(h). These are bounded probabilities that sit still
#     for long stretches, so long-horizon spread grows slower than sqrt(h);
#     0.45 beat 0.5 and 0.4 on the practice, and the 0.42-0.48 band is flat.
#   * Shape: plain normal quantiles. Heavier (Student-t) tails and
#     logit-space forecasts both lost on the practice once the blend was in.
#   * Every quantile is clamped to [0, 1] and rounded to 6 decimals.
#
# Deterministic: pure arithmetic on the case file, no randomness, no network,
# no clock. Two runs on the same case write the same bytes.

import json
import math
import os

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at the five levels.
Z = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]

# Blend weights for the two volatility windows (minutes).
SHORT_WINDOW = 360    # 6h: tracks the current regime
LONG_WINDOW = 1440    # 24h: steadies the estimate
SHORT_WEIGHT = 0.7
LONG_WEIGHT = 0.3

SIGMA_FLOOR = 0.0003
SIGMA_CAP = 0.008
SIGMA_FALLBACK = 0.0025  # for a market with too little history
ALPHA = 0.45             # spread grows as h ** ALPHA


def price6(x):
    """Clamp to [0, 1] and round to 6 decimals (0, never -0)."""
    return round(min(1.0, max(0.0, x)), 6) + 0.0


def mean_abs_sigma(prices, window):
    """Mean absolute 1-minute move over the last `window` minutes, as a
    normal-equivalent sigma. Jump-resistant: one big jump barely moves it,
    where a standard deviation would explode."""
    seg = prices[-window:] if len(prices) > window else prices
    if len(seg) < 10:
        return None
    total = 0.0
    for i in range(len(seg) - 1):
        total += abs(seg[i + 1] - seg[i])
    return total / (len(seg) - 1) * 1.2533


def market_sigma(prices):
    """Blended per-market sigma, floored and capped."""
    short = mean_abs_sigma(prices, SHORT_WINDOW)
    long = mean_abs_sigma(prices, LONG_WINDOW)
    if short is None and long is None:
        return SIGMA_FALLBACK
    if short is None:
        return min(SIGMA_CAP, max(SIGMA_FLOOR, long))
    if long is None:
        return min(SIGMA_CAP, max(SIGMA_FLOOR, short))
    blended = SHORT_WEIGHT * short + LONG_WEIGHT * long
    return min(SIGMA_CAP, max(SIGMA_FLOOR, blended))


def forecast_market(market, horizon):
    """240 rows of five quantiles for one market, median = last price."""
    prices = market.get("prices") or []
    if not prices:
        # No history at all: stay valid with a flat 0.5 forecast.
        return [[0.5, 0.5, 0.5, 0.5, 0.5] for _ in range(horizon)]
    x = prices[-1]
    sigma = market_sigma(prices)
    # Horizon growth factors, computed once per market instead of per row.
    growth = [h ** ALPHA for h in range(1, horizon + 1)]
    rows = []
    for h in range(1, horizon + 1):
        spread = sigma * growth[h - 1]
        rows.append([price6(x + z * spread) for z in Z])
    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inputs = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    for filename in inputs:
        case_id = filename[: -len(".in")]
        with open(os.path.join(INPUT_DIR, filename), "r", encoding="utf-8") as f:
            case = json.load(f)
        horizon = case.get("horizon_minutes", 240)
        quantiles = case.get("quantiles", QUANTILES)
        if list(quantiles) != QUANTILES:
            raise ValueError(
                "%s asks for quantiles %s, not %s" % (filename, quantiles, QUANTILES)
            )
        forecasts = [
            {"id": m["id"], "q": forecast_market(m, horizon)}
            for m in case["markets"]
        ]
        out_path = os.path.join(OUTPUT_DIR, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"forecasts": forecasts}, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
