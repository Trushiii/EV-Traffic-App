"""
Trains the EV Traffic Grid surrogate model from one or two NetLogo
BehaviorSpace "spreadsheet" exports.

Usage:
    python train_model.py ev_experiment.csv congestion_experiment.csv

Each CSV can have a different set of swept constants (the parser detects
this automatically). Runs from a CSV are aggregated (mean) over their
repetitions, then all runs are pooled into one dataframe. Any of the 5
model features missing from a given CSV (because that experiment held it
fixed rather than sweeping it) is filled with the value in DEFAULTS below,
which mirrors this model's interface slider defaults.

Produces ev_model.pkl in the current directory, which app.py loads.
"""

import csv
import json
import sys

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict

FEATURE_COLS = [
    "ev_percentage",
    "num_charging_stations",
    "charging_bays",
    "num_cars",
    "ticks_per_cycle",
]
TARGET_COLS = [
    "mean_wait_time",
    "mean_speed",
    "count_charging",
    "num_cars_stopped",
    "mean_ev_battery",
]

# Interface slider defaults, used to fill in a feature that a given
# BehaviorSpace experiment held fixed (and therefore didn't log as a
# swept constant) rather than varied.
DEFAULTS = {
    "ev_percentage": 20,
    "num_charging_stations": 6,
    "charging_bays": 2,
    "num_cars": 200,
    "ticks_per_cycle": 20,
}

# Maps a BehaviorSpace constant-row label to our column name
VAR_NAME_MAP = {
    "ev-percentage": "ev_percentage",
    "num-charging-stations": "num_charging_stations",
    "charging-bays": "charging_bays",
    "num-cars": "num_cars",
    "ticks-per-cycle": "ticks_per_cycle",
}


def parse_behaviorspace_spreadsheet(path: str) -> pd.DataFrame:
    """Reshapes a NetLogo BehaviorSpace "spreadsheet" export into a tidy
    dataframe, one row per run. Any of the 5 model features this
    experiment didn't sweep is filled with its DEFAULTS value."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    run_row_idx = next(i for i, r in enumerate(rows) if r and r[0] == "[run number]")

    constants = {}  # column name -> list of per-column values (blank outside block start)
    i = run_row_idx + 1
    while rows[i] and rows[i][0] in VAR_NAME_MAP:
        col_name = VAR_NAME_MAP[rows[i][0]]
        constants[col_name] = rows[i][1:]
        i += 1

    i += 1  # skip the "[total steps]" row
    header_idx = i + 1  # skip the blank separator row
    data_idx = header_idx + 1
    header_row = rows[header_idx]
    data_row = rows[data_idx][1:]  # strip the row label, same convention as run_values
    run_values = rows[run_row_idx][1:]

    n_cols = len(data_row)
    n_runs = len(set(run_values))
    block_size = n_cols // n_runs
    metric_order = header_row[2 : 1 + block_size]  # header_row[0]=label, [1]="[step]"

    records = []
    for b in range(n_runs):
        s = b * block_size
        run_id = run_values[s]
        metric_values = data_row[s + 1 : s + 1 + len(metric_order)]  # skip [step]

        record = {"run": int(run_id)}
        for col_name in FEATURE_COLS:
            if col_name in constants:
                # value is only populated at the block's first column, blank after
                raw = constants[col_name][s]
                record[col_name] = int(float(raw)) if raw != "" else record.get(col_name)
            else:
                record[col_name] = DEFAULTS[col_name]

        for metric_name, value in zip(metric_order, metric_values):
            if "wait-time" in metric_name:
                record["mean_wait_time"] = float(value)
            elif "speed" in metric_name:
                record["mean_speed"] = float(value)
            elif "charging?" in metric_name:
                record["count_charging"] = int(float(value))
            elif "num-cars-stopped" in metric_name:
                record["num_cars_stopped"] = int(float(value))
            elif "battery" in metric_name:
                record["mean_ev_battery"] = float(value)
        records.append(record)

    return pd.DataFrame(records)


def confidence_label(r2: float) -> str:
    if r2 >= 0.4:
        return "Well predicted"
    elif r2 >= 0.15:
        return "Weak relationship"
    return "Not meaningfully predictable from these inputs"


def main(csv_paths: list[str]):
    all_agg = []
    for path in csv_paths:
        df = parse_behaviorspace_spreadsheet(path)
        print(f"{path}: parsed {len(df)} individual runs")
        agg = df.groupby(FEATURE_COLS)[TARGET_COLS].mean().reset_index()
        print(f"{path}: aggregated to {len(agg)} unique parameter combinations")
        all_agg.append(agg)

    agg = pd.concat(all_agg, ignore_index=True)
    print(f"\nPooled dataset: {len(agg)} unique parameter combinations across {len(csv_paths)} file(s)")

    X = agg[FEATURE_COLS]
    y = agg[TARGET_COLS].values

    loo_preds = cross_val_predict(LinearRegression(), X, y, cv=LeaveOneOut())
    diagnostics = {}
    for i, t in enumerate(TARGET_COLS):
        r2 = float(r2_score(y[:, i], loo_preds[:, i]))
        diagnostics[t] = {
            "r2": round(r2, 3),
            "mae": round(float(mean_absolute_error(y[:, i], loo_preds[:, i])), 3),
            "target_min": round(float(agg[t].min()), 2),
            "target_max": round(float(agg[t].max()), 2),
            "confidence": confidence_label(r2),
        }

    print(json.dumps(diagnostics, indent=2))

    final_model = LinearRegression()
    final_model.fit(X, y)

    joblib.dump(
        {
            "model": final_model,
            "feature_cols": FEATURE_COLS,
            "target_cols": TARGET_COLS,
            "diagnostics": diagnostics,
            "feature_ranges": {c: [int(agg[c].min()), int(agg[c].max())] for c in FEATURE_COLS},
        },
        "ev_model.pkl",
    )
    print("\nSaved ev_model.pkl")


if __name__ == "__main__":
    paths = sys.argv[1:] if len(sys.argv) > 1 else ["trafficEV.csv"]
    main(paths)
