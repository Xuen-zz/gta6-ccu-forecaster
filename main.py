#!/usr/bin/env python3
"""
GTA VI Twitch CCU Launch-Day Probabilistic Forecaster
Competition: gta6-twitch-ccu-forecast-01 (make.fun)

Architecture: Unlock-Aligned Log-Ratio Comparable Ensemble with Empirical Quantiles
Runtime: Python 3.13 (Standard Library Only, 0 external dependencies)
Execution Time: ~15ms per case (Targeting 1st place tie-breaker on run_ms)
"""

import json
import math
import os
import sys
from pathlib import Path

# Fix sys.path for sandbox environment
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
DEFAULT_Z = [-1.645, -0.674, 0.0, 0.674, 1.645]


def round2(x: float) -> float | int:
    """Rounds x to 2 decimal places, half up, identical to Math.round(x * 100) / 100 in JS."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if value.is_integer() else value


def is_rolling_unlock(unlock_str: str) -> bool:
    """Detects if an unlock schedule is rolling midnight local or global simultaneous."""
    s = str(unlock_str).lower()
    return "rolling" in s or "midnight local" in s


def weighted_quantile(values: list[float], weights: list[float], quantiles: list[float]) -> list[float]:
    """Computes exact weighted empirical quantiles from a continuous sample distribution."""
    sorter = sorted(range(len(values)), key=lambda k: values[k])
    sorted_vals = [values[i] for i in sorter]
    sorted_weights = [weights[i] for i in sorter]

    total_w = sum(sorted_weights)
    if total_w <= 0.0:
        total_w = 1.0
        sorted_weights = [1.0 / len(values)] * len(values)

    cum_w = []
    curr = 0.0
    for w in sorted_weights:
        curr += w
        cum_w.append(curr / total_w)

    res = []
    for q in quantiles:
        idx = 0
        while idx < len(cum_w) and cum_w[idx] < q:
            idx += 1
        if idx >= len(cum_w):
            idx = len(cum_w) - 1
        res.append(sorted_vals[idx])
    return res


def forecast_case(case: dict, case_name: str) -> dict:
    """Generates 23-hour 5-quantile forecast for a single competition case."""
    if case.get("levels") != LEVELS:
        raise ValueError(f"{case_name} requested levels {case.get('levels')}, expected {LEVELS}")

    known_list = case.get("known", [])
    first_hour = next((k for k in known_list if k.get("hour") == 1), None)
    if first_hour is None:
        raise ValueError(f"{case_name} has no valid hour 1 record")

    p1 = max(1.0, float(first_hour.get("viewers", 1.0)))

    # Extract target case context
    notes = str(case.get("notes", ""))
    target_is_rolling = is_rolling_unlock(notes)

    # Parse and extract launch trajectories from the comparables dataset
    comparables = case.get("comparables", [])
    trajectories = []

    for comp in comparables:
        day_start = str(comp.get("day_start", ""))[:10]
        hourly = comp.get("hourly", [])
        # Launch day starts at day_start 00:00 UTC
        launch_hours = [x for x in hourly if str(x.get("from", "")).startswith(day_start)]

        if len(launch_hours) >= 24:
            v_init = float(launch_hours[0].get("viewers", 0.0))
            if v_init > 0.0:
                y_curve = {h: max(1.0, float(launch_hours[h - 1]["viewers"])) for h in range(1, 25)}
                trajectories.append({
                    "name": comp.get("name", ""),
                    "platforms": str(comp.get("platforms", "")),
                    "unlock": str(comp.get("unlock", "")),
                    "is_rolling": is_rolling_unlock(str(comp.get("unlock", ""))),
                    "y": y_curve,
                    "v1": v_init,
                })

    # Robust Fallback to Persistence Baseline if no comparables found
    if not trajectories:
        return {
            "hours": [
                {
                    "hour": row["hour"],
                    "q": [round2(p1 * math.exp(z * 0.15 * math.sqrt(row["hour"] - 1))) for z in DEFAULT_Z],
                }
                for row in case["forecast"]
            ]
        }

    # Similarity weighting: scale distance + console platform + unlock alignment
    log_p1 = math.log(p1)
    weights = []
    for t in trajectories:
        dist_scale = abs(log_p1 - math.log(t["v1"]))
        w = math.exp(-0.25 * dist_scale)

        # Platform match (consoles vs PC)
        if "PS" in t["platforms"] or "Xbox" in t["platforms"]:
            w *= 1.4

        # Unlock schedule alignment
        if t["is_rolling"] == target_is_rolling:
            w *= 1.8

        weights.append(w)

    # Forecast hours 2 to 24
    hours_output = []
    for row in case.get("forecast", []):
        h = row["hour"]
        log_ratios = [math.log(t["y"][h] / t["v1"]) for t in trajectories]
        quantiles_log = weighted_quantile(log_ratios, weights, LEVELS)

        # Map back to viewer counts
        q_vals = [round2(p1 * math.exp(val)) for val in quantiles_log]

        # Enforce strict non-decreasing order and positive boundaries
        for i in range(1, len(q_vals)):
            if q_vals[i] < q_vals[i - 1]:
                q_vals[i] = q_vals[i - 1]
        q_vals = [max(0.01, min(1_000_000_000.0, float(v))) for v in q_vals]

        hours_output.append({"hour": h, "q": q_vals})

    return {"hours": hours_output}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Process all input cases in deterministic lexicographical order
    case_paths = sorted(INPUT_DIR.glob("*.in"))
    for path in case_paths:
        try:
            case_data = json.loads(path.read_text(encoding="utf-8"))
            result = forecast_case(case_data, path.name)
            output_file = OUTPUT_DIR / f"{path.stem}.out"
            output_file.write_text(json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8")
        except Exception as e:
            sys.stderr.write(f"Error processing {path.name}: {e}\n")


if __name__ == "__main__":
    main()
