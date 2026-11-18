# GTA VI Launch-Day Twitch CCU Forecaster

[![make.fun bounty](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![Rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Xuen-zz/gta6-ccu-forecaster/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![Reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Xuen-zz/gta6-ccu-forecaster/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

Entry for the [Forecast GTA VI's launch-day Twitch viewers hour by hour](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) competition on [make.fun](https://make.fun).

## Method Overview

`main.py` runs in Python 3.13 using only the standard library (zero external dependencies, completely offline, deterministic, ~14 ms per case).

- **Log-Ratio Comparable Ensemble**: Extracts 24-hour launch-day trajectories from the historical comparable launches included in each case and converts them to ratios relative to Hour 1 (`Y(h) / Y(1)`).
- **Similarity Weighting**: Weights each comparable game by:
  - Log-scale distance between Hour 1 viewer counts (`exp(-0.25 * |ln(p1) - ln(v1)|)`)
  - Console platform alignment (PlayStation / Xbox launches)
  - Release unlock schedule alignment (rolling midnight local vs. global simultaneous unlock)
- **Weighted Empirical Quantiles**: Computes the weighted empirical quantiles directly at `[0.05, 0.25, 0.50, 0.75, 0.95]` in log space, maps back to viewer counts with JS-compatible half-up 2-decimal rounding, and enforces strict monotonicity and bounds.

## Usage

```bash
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## License

MIT — see [LICENSE](LICENSE).
