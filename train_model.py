"""
Trains the EV Traffic Grid surrogate model from a NetLogo BehaviorSpace
"spreadsheet" export.

Usage:
    python train_model.py path/to/trafficEV.csv

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

FEATURE_COLS = ["ev_percentage", "num_charging_stations", "charging_bays"]
TARGET_COLS = [
    "mean_wait_time",
    "mean_speed",
    "count_charging",
    "num_cars_stopped",
    "mean_ev_battery",
]


def parse_behaviorspace_spreadsheet(path: str) -> pd.DataFrame:
    """Reshapes NetLogo's wide BehaviorSpace spreadsheet export into a tidy
    dataframe, one row per run. Expects the standard "final value" spreadsheet
    layout with 6 output columns per run (step, wait-time, speed, charging
    count, cars-stopped, battery) and constants recorded once per run block
    (blank in the remaining columns of that block)."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    run_row = rows[6][1:]
    ev_row = rows[7][1:]
    stations_row = rows[8][1:]
    bays_row = rows[9][1:]
    data_row = rows[13][1:]

    n_cols = len(data_row)
    assert n_cols % 6 == 0, "Unexpected column count, layout may have changed"
    n_runs = n_cols // 6

    records = []
    for b in range(n_runs):
        s = b * 6
        step, wait_time, speed, charging_count, cars_stopped, battery = data_row[s : s + 6]
        records.append(
            {
                "run": int(run_row[s]),
                "ev_percentage": int(ev_row[s]),
                "num_charging_stations": int(stations_row[s]),
                "charging_bays": int(bays_row[s]),
                "mean_wait_time": float(wait_time),
                "mean_speed": float(speed),
                "count_charging": int(float(charging_count)),
                "num_cars_stopped": int(float(cars_stopped)),
                "mean_ev_battery": float(battery),
            }
        )
    return pd.DataFrame(records)


def confidence_label(r2: float) -> str:
    if r2 >= 0.4:
        return "Well predicted"
    elif r2 >= 0.15:
        return "Weak relationship"
    return "Not meaningfully predictable from these inputs"


def load_any(path: str) -> pd.DataFrame:
    head = pd.read_csv(path, nrows=0).columns.tolist()
    if {"ev_percentage", "num_charging_stations", "charging_bays"} <= set(head):
        df = pd.read_csv(path)
        return df.groupby(FEATURE_COLS)[TARGET_COLS].mean().reset_index()
    return parse_behaviorspace_spreadsheet(path)


def main(csv_path: str):
    agg = load_any(csv_path)
    print(f"Loaded {len(agg)} unique parameter combinations")

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

    loo_residuals = y - loo_preds
    target_std    = y.std(axis=0)
    unique_values = {c: sorted(agg[c].unique().tolist()) for c in FEATURE_COLS}

    joblib.dump(
        {
            "model": final_model,
            "feature_cols": FEATURE_COLS,
            "target_cols": TARGET_COLS,
            "diagnostics": diagnostics,
            "feature_ranges": {c: [int(agg[c].min()), int(agg[c].max())] for c in FEATURE_COLS},
            "train_X": X.values,
            "train_y": y,
            "loo_residuals": loo_residuals,
            "target_std": target_std,
            "unique_values": unique_values,
        },
        "ev_model.pkl",
    )
    print("\nSaved ev_model.pkl")


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "trafficEV.csv"
    main(csv_path)
