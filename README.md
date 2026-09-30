# EV Traffic Grid Predictor

A Streamlit app that predicts the outcome of the NetLogo EV Traffic Grid
simulation (wait time, speed, charging activity, congestion, EV battery
level) for any combination of 8 parameters, without re-running the
simulation: EV percentage, charging stations, bays per station, number of
cars, traffic light cycle length, grid size (rows & columns), and speed limit.

It's trained on three pooled BehaviorSpace exports:
`ev_tidy_192runs.csv` (EV share and charging setup), `congestion_45runs.csv`
(car count and light timing), and `network_54runs.csv` (grid size and
speed limit).

## What's in here

| File | Purpose |
|---|---|
| `app.py` | The Streamlit app |
| `ev_model.pkl` | Trained model, saved with feature names, target names, and validation diagnostics |
| `train_model.py` | Reproducible training script, retrains from one or more BehaviorSpace exports |
| `requirements.txt` | Python dependencies |
| `.streamlit/config.toml` | Dark dashboard theme (colors pulled from the sim's own asphalt/road/charging-station palette) |
| `ev_tidy_192runs.csv`, `ev_aggregated_64combos.csv`, `congestion_45runs.csv`, `network_54runs.csv` | The underlying run data, useful for your thesis appendix |

**Important:** `.streamlit` is a folder whose name starts with a dot. Make sure
it lands in the same folder as `app.py`, not renamed or flattened, or the
custom colors won't apply and Streamlit will fall back to its default theme.

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

It opens at `http://localhost:8501`.

## Retrain on new data

```bash
python train_model.py ev_tidy_192runs.csv congestion_45runs.csv network_54runs.csv
```

Pass as many BehaviorSpace CSVs as you have; the script pools all of them.
Any of the 8 parameters an experiment didn't sweep is filled with its
interface slider default (see `DEFAULTS` in `train_model.py`). This
overwrites `ev_model.pkl`, the app picks it up next time it starts.

## Deploy it (free, public URL)

**Streamlit Community Cloud:**

1. Push this folder to a GitHub repo. Include `app.py`, `ev_model.pkl`, `.streamlit/config.toml`, and `requirements.txt` at minimum.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. New app → pick your repo/branch → main file path: `app.py` → Deploy.

Alternatives: Hugging Face Spaces (free, native Streamlit support) or
Render's free web service tier.

## A note on model quality

The underlying simulation is stochastic, cars spawn with random starting
battery and random positions, so repeated runs of the exact same settings
still land in different places. The model is trained on the average outcome
across the 3 repetitions of each of the 97 unique parameter combinations
pooled across all three experiments, validated with leave-one-out
cross-validation.

Current validation:

| Metric | R² | Confidence |
|---|---|---|
| Cars stopped | 0.85 | 🟢 Well predicted |
| EV battery | 0.73 | 🟢 Well predicted |
| Speed | 0.66 | 🟢 Well predicted |
| Charging activity | 0.44 | 🟢 Well predicted |
| Wait time | 0.24 | 🟡 Weak relationship |

**Wait time is the one that didn't budge.** Car count and light timing
(experiment 2) pushed other metrics up substantially, but a follow-up
experiment on grid size and speed limit (experiment 3) didn't move wait
time's R² up, if anything it dipped slightly (0.32 → 0.24) once pooled in,
likely just LOOCV variance from a small dataset rather than the new factors
actively hurting it. Two honest explanations, and probably both apply:

1. **Genuine floor.** In this model, an individual car's wait time depends
   heavily on the random phase of the traffic light when it happens to
   arrive, run-to-run luck more than any of the 8 swept parameters.
   Averaging 3 repetitions only partially cancels that out.
2. **Untested interactions.** The three experiments each varied their own
   parameter family while holding the rest fixed (a "star" design, see
   below), so the model has never seen, say, high EV% *and* a small grid
   *and* a fast light cycle all at once. If wait time depends on an
   interaction between families, a model trained this way can't see it.

If you want to push further: either increase `repetitions` in BehaviorSpace
(e.g. 10 instead of 3) to test whether more averaging shrinks the gap
(supports explanation 1), or design one experiment that varies a few
parameters from *different* families together instead of one family at a
time (supports explanation 2, but grows combinatorially fast, watch the
`num-cars > count roads` capacity constraint from before). Otherwise, "wait
time is largely noise-driven in this model" is itself a defensible finding
worth stating plainly in your writeup rather than something to keep
chasing.

One limitation worth stating explicitly regardless: the three experiments
were never **jointly** swept. The model assumes their effects combine
additively, a reasonable assumption for the four green metrics (the
validation backs it up), less certain for wait time. The app clips
predictions to physically valid ranges (no negative speeds, battery
capped at 100%, etc.) since a linear model can otherwise extrapolate past
what's physically possible at extreme, untested combinations.

The app surfaces per-metric confidence with a badge (🟢🟡🔴) rather than
presenting every prediction as equally trustworthy.
