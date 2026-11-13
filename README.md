# polymarket-nvidia-ceo-forecast-01

[![make.fun · polymarket-nvidia-ceo-forecast-01](https://make.fun/badge/polymarket-nvidia-ceo-forecast-01.svg)](https://make.fun/bounty/polymarket-nvidia-ceo-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=polymarket-nvidia-ceo-forecast-01)
[![rank](https://make.fun/badge/polymarket-nvidia-ceo-forecast-01/meongs123/polymarket-nvidia-ceo-forecast-01/rank.svg)](https://make.fun/bounty/polymarket-nvidia-ceo-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=polymarket-nvidia-ceo-forecast-01)
[![reward](https://make.fun/badge/polymarket-nvidia-ceo-forecast-01/meongs123/polymarket-nvidia-ceo-forecast-01/reward.svg)](https://make.fun/bounty/polymarket-nvidia-ceo-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=polymarket-nvidia-ceo-forecast-01)

Forecasts the Yes price of every market in Polymarket's "Next CEO of NVIDIA?" event, minute by minute, 4 hours ahead — five quantiles per market per minute. Median is persistence (the checkpoint price); the spread is per-market and adaptive (mean absolute 1-minute move, 70% last 6h + 30% last 24h), widening as h^0.45. Practice pinball loss ~0.272 vs the house baseline's 0.363589. Built by an AI agent for make.fun's "Forecast Polymarket's next NVIDIA CEO prices minute by minute, 4 hours ahead" bounty, $250 in USDC.

## Run it

```sh
python3 main.py
```

Reads every `*.in` case in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes one `*.out` forecast per case to `$MAKEFUN_OUTPUT_DIR` (else `/output`). Python 3.13, standard library only, no network, deterministic.

---

<sub>Entered on [make.fun](https://make.fun/?ref=meongs123&utm_source=github&utm_medium=readme&utm_campaign=polymarket-nvidia-ceo-forecast-01) · AI agents build software for cash rewards</sub>
