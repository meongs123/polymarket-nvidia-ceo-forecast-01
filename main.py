# Polymarket "Next CEO of NVIDIA?" price forecaster (make.fun bounty
# polymarket-nvidia-ceo-forecast-01). Python 3.13, standard library only.
#
# Model (tuned on the six practice cases, practice pinball loss 0.253213 vs
# the house baseline's 0.363589; median error 0.7262, 90% coverage ~0.88):
#
#   * Median: the last TWO minutes averaged, plus a sum correction. The
#     1-minute CLOB midpoint carries microstructure noise (a single tick
#     can sit a point or two off the true price), so the base price is the
#     mean of the last two minutes -- worth ~0.006 of practice loss over
#     the raw last price, and the 0.5-0.7 weighting plateau is flat, so it
#     is not a knife-edge fit. The 24 Yes prices are mutually exclusive
#     outcomes, so they must sum to 1 -- but at the practice checkpoints
#     they summed to 0.81-1.09. When the sum S is off, the median drifts
#     toward x / S with a 120-minute time constant, i.e. the market's
#     mispricing is assumed to unwind over ~2h. This helps exactly where
#     the sum is farthest from 1 (p00, p02, p04) and does nothing where
#     the sum is already 1 (p01). Plain mean reversion to a 60-min mean
#     and reverting after big jumps both made the loss WORSE, so there is
#     no other drift: prices are martingales except for the sum constraint.
#   * Spread: per-market, adaptive. sigma is the mean absolute 1-minute
#     return (x1.2533, the normal equivalent), blended 70% from the last 6h
#     with 30% from the last 24h. The 6h half tracks the current volatility
#     regime; the 24h half steadies it. A plain standard deviation is useless
#     here: one 20-point jump in 24h inflates it so far that the intervals
#     are far too wide at every quiet minute. Mean absolute move is
#     jump-resistant and won on all six practice cases, early and late.
#     Floor 0.0003, cap 0.008 price units per minute. The sigma scales by
#     1/S with the median: if the price level is corrected by 1/S, its
#     moves scale the same way.
#   * Horizon shape: quantiles widen as h**0.40, slower than a random
#     walk's sqrt(h). These are bounded probabilities that sit still for
#     long stretches, so long-horizon spread grows slower than sqrt(h).
#   * Shape: Laplace quantiles, not normal. One-minute returns are
#     zero-inflated -- the price sits still for many minutes, then jumps --
#     so the forecast distribution has a sharper peak and heavier tails
#     than a normal. Laplace z's are ln(2*tau): -2.3026, -0.6931, 0,
#     0.6931, 2.3026, with scale = mean absolute move (sigma / 1.2533).
#     Worth ~0.001 over normal on the practice, and the same shape shows
#     up in the board's #3 entry, so it is not a one-off fit.
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
# Laplace quantiles at the five levels: ln(2*tau) for tau < 0.5,
# -ln(2*(1-tau)) for tau > 0.5.
Z = [-2.302585, -0.693147, 0.0, 0.693147, 2.302585]

# Blend weights for the two volatility windows (minutes).
SHORT_WINDOW = 360    # 6h: tracks the current regime
LONG_WINDOW = 1440    # 24h: steadies the estimate
SHORT_WEIGHT = 0.7
LONG_WEIGHT = 0.3

SIGMA_FLOOR = 0.0003
SIGMA_CAP = 0.008
SIGMA_FALLBACK = 0.0025  # for a market with too little history
ALPHA = 0.40             # spread grows as h ** ALPHA
SUM_TAU = 120            # minutes: median unwinds toward x / S on this scale
NORMAL_EQUIV = 1.2533    # mean absolute move -> normal-equivalent sigma


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
    return total / (len(seg) - 1) * NORMAL_EQUIV


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


def forecast_market(market, horizon, total):
    """240 rows of five quantiles for one market. The base price is the mean
    of the last two minutes (microstructure noise); the median unwinds
    toward x / total (the sum-to-one correction); the spread scales by
    1 / total with it and follows Laplace quantiles."""
    prices = market.get("prices") or []
    if not prices:
        # No history at all: stay valid with a flat 0.5 forecast.
        return [[0.5, 0.5, 0.5, 0.5, 0.5] for _ in range(horizon)]
    if len(prices) >= 2:
        x = (prices[-1] + prices[-2]) / 2.0
    else:
        x = prices[-1]
    sigma = market_sigma(prices)
    if total > 0:
        target = x / total
        sigma = sigma / total
    else:
        target = x
    # Laplace scale is the mean absolute move, i.e. sigma / 1.2533.
    scale = sigma / NORMAL_EQUIV
    # Horizon growth factors, computed once per market instead of per row.
    growth = [h ** ALPHA for h in range(1, horizon + 1)]
    unwind = [1.0 - math.exp(-h / SUM_TAU) for h in range(1, horizon + 1)]
    rows = []
    for h in range(1, horizon + 1):
        median = x + (target - x) * unwind[h - 1]
        spread = scale * growth[h - 1]
        rows.append([price6(median + z * spread) for z in Z])
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
        markets = case["markets"]
        total = sum((m.get("prices") or [0.0])[-1] for m in markets)
        forecasts = [
            {"id": m["id"], "q": forecast_market(m, horizon, total)}
            for m in markets
        ]
        out_path = os.path.join(OUTPUT_DIR, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"forecasts": forecasts}, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
